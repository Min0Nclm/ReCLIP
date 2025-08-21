import os
import random
from enum import Enum
import PIL
import torch
from torchvision import transforms
import json
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F

from medsyn.tasks import CutPastePatchBlender,SmoothIntensityChangeTask,GaussIntensityChangeTask,SinkDeformationTask,SourceDeformationTask,IdentityTask

# =================================================================================
# K-Means based Support Set Selection
# =================================================================================

def _select_most_representative(features):
    """Selects the single most representative sample from a feature set."""
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    mean_sim = sim_matrix.mean(dim=1)
    return torch.argmax(mean_sim).item()

def _k_center_greedy_init(features, k):
    """Initializes K-Means centers greedily to ensure diversity."""
    selected_indices = [_select_most_representative(features)]
    
    while len(selected_indices) < k:
        dists = []
        for i in range(len(features)):
            if i in selected_indices:
                continue
            # Find the minimum distance from the current point to any already selected center
            min_dist_to_selected = min(torch.norm(features[i] - features[j]) for j in selected_indices)
            dists.append((min_dist_to_selected.item(), i))
        
        # Select the point that is farthest from its nearest center
        if not dists:
            break # Should not happen if k < len(features)
        _, next_idx = max(dists)
        selected_indices.append(next_idx)
        
    return selected_indices

def _select_support_samples_by_clustering(features, k):
    """
    Selects k representative support samples using K-Means clustering.
    This ensures diversity in the support set.
    """
    if k >= len(features):
        print(f"Warning: k ({k}) is >= number of samples ({len(features)}). Using all samples as support.")
        return list(range(len(features)))

    feats_np = features.numpy()
    
    # 1. Initialize cluster centers greedily to promote diversity
    init_ids = _k_center_greedy_init(features, k)
    
    # 2. Run K-Means clustering
    kmeans = KMeans(n_clusters=k, init=feats_np[init_ids], n_init=1, random_state=0).fit(feats_np)
    
    # 3. Find the sample closest to each cluster center to be the final support sample
    support_indices = []
    for i in range(k):
        cluster_member_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_member_indices) == 0:
            # If a cluster is empty (rare), use the initial seed point as the support
            support_indices.append(init_ids[i])
            continue
        
        cluster_members_feats = feats_np[cluster_member_indices]
        center_feat = kmeans.cluster_centers_[i]
        
        # Find the member closest to the center (in Euclidean distance)
        dists = np.linalg.norm(cluster_members_feats - center_feat, axis=1)
        closest_in_cluster_idx = np.argmin(dists)
        support_indices.append(cluster_member_indices[closest_in_cluster_idx])
        
    return support_indices

# =================================================================================

class TrainDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        k_shot=-1,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.k_shot = k_shot
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()

        # Load pre-computed features
        feature_path = os.path.join(self.source, 'sam_features.pt')
        if os.path.exists(feature_path):
            print(f"Loading pre-computed features from {feature_path}")
            features_data = torch.load(feature_path, map_location='cpu')
            self.sam_features = features_data['features'].to(torch.float32) # Ensure float32
            self.sam_image_paths = features_data['image_paths']
            print("Features loaded successfully.")
        else:
            raise FileNotFoundError(f"Feature file not found: {feature_path}. Please run precompute_features.py first.")

        # Ensure the number of images in JSON matches the feature file
        assert len(self.data_to_iterate) == len(self.sam_image_paths)

        # --- Support Set Selection via Clustering (Done ONCE) ---
        num_support = getattr(self.args, 'num_support_samples', 5)
        print(f"Selecting {num_support} diverse support samples using K-Means clustering...")
        self.support_indices = _select_support_samples_by_clustering(self.sam_features, num_support)
        
        # Load the selected support images once to be used for all training items
        self.support_images = [self.read_image(self.sam_image_paths[i]) for i in self.support_indices]
        print(f"Selected support sample indices: {self.support_indices}")
        # --- End of Selection Logic ---

        # Pre-load augmentations
        self.augs, self.augs_pro = self.load_anomaly_syn(self.support_images)
        assert sum(self.augs_pro) == 1.0

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source, 'images', info['filename'])
        image = self.read_image(image_path)

        # The support set is now pre-selected and shared across all items.
        # We just need to pick an augmentation and apply it.
        choice_aug = np.random.choice(a=self.augs, p=self.augs_pro, size=(1,), replace=False)[0]
        
        # To prevent an image from being its own support in CutPaste, 
        # we can temporarily filter it out if the choice is CutPaste.
        support_images_for_item = self.support_images
        if isinstance(choice_aug, CutPastePatchBlender):
            current_image_path_norm = os.path.normpath(image_path)
            support_paths_norm = [os.path.normpath(self.sam_image_paths[i]) for i in self.support_indices]
            if current_image_path_norm in support_paths_norm:
                # If the current image is in the support set, provide a filtered list
                # to the augmentation task.
                filtered_support_images = [
                    img for i, img in enumerate(self.support_images) 
                    if os.path.normpath(self.sam_image_paths[self.support_indices[i]]) != current_image_path_norm
                ]
                # If filtering results in an empty list, just use the original set.
                if filtered_support_images:
                    choice_aug.support_imgs = filtered_support_images

        image, mask = choice_aug(image)
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def read_image(self, path):
        image = PIL.Image.open(path).resize((self.args.image_size, self.args.image_size),
                                            PIL.Image.Resampling.BILINEAR).convert("L")
        image = np.array(image).astype(np.uint8)
        return image

    def get_image_data(self):
        data_to_iterate = []
        json_path = os.path.join(self.source, 'samples', "train.json")
        with open(json_path, "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        if self.k_shot != -1:
            data_to_iterate = random.sample(data_to_iterate, self.k_shot)
        return data_to_iterate

    def load_anomaly_syn(self, support_images_for_cutpaste):
        tasks = []
        task_probability = []
        # Ensure anomaly_tasks is not None and is a dict
        if not hasattr(self.args, 'anomaly_tasks') or not isinstance(self.args.anomaly_tasks, dict):
             raise ValueError("args.anomaly_tasks must be a dictionary of tasks.")

        for task_name in self.args.anomaly_tasks.keys():
            if task_name == 'CutpasteTask':
                # Use the dynamically provided similar images
                task = CutPastePatchBlender(support_images_for_cutpaste)
            elif task_name == 'SmoothIntensityTask':
                task = SmoothIntensityChangeTask(30.0)
            elif task_name == 'GaussIntensityChangeTask':
                task = GaussIntensityChangeTask()
            elif task_name == 'SinkTask':
                task = SinkDeformationTask()
            elif task_name == 'SourceTask':
                # task = SourceDeformationTask()
                task = ElasticDeformationTask()
            elif task_name == 'IdentityTask':
                task = IdentityTask()
            else:
                raise NotImplementedError(f"Task {task_name} not implemented.")

            tasks.append(task)
            task_probability.append(self.args.anomaly_tasks[task_name])
        return tasks, task_probability




class ChexpertTestDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source

        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()


    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source,'images',info['filename'])
        image = PIL.Image.open(image_path).convert("RGB").resize((self.args.image_size,self.args.image_size),PIL.Image.Resampling.BILINEAR)
        mask = np.zeros((self.args.image_size,self.args.image_size)).astype(np.float64)
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask" : mask,
            "classname": info['clsname'],
            "is_anomaly": info['label'],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate


class BrainMRITestDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):

        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()


    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source,'images',info['filename'])
        image = PIL.Image.open(image_path).convert("RGB").resize((self.args.image_size,self.args.image_size),PIL.Image.Resampling.BILINEAR)
        mask = np.zeros((self.args.image_size,self.args.image_size)).astype(np.float64)
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask" : mask,
            "classname": info['clsname'],
            "is_anomaly": info['label'],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)


    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate



class BusiTestDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):

        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()


    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source,'images',info['filename'])
        image = PIL.Image.open(image_path).convert("RGB").resize((self.args.image_size,self.args.image_size),PIL.Image.Resampling.BILINEAR)

        if info.get("mask", None):
            mask = os.path.join(self.source,'images',info['mask'])
            mask = PIL.Image.open(mask).convert("L").resize((self.args.image_size,self.args.image_size),PIL.Image.Resampling.NEAREST)
            mask = np.array(mask).astype(np.float64)/255.0
            mask [mask!=0.0] = 1.0
        else:
            mask = np.zeros((self.args.image_size,self.args.image_size)).astype(np.float64)

        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "classname": info['clsname'],
            "is_anomaly": info['label'],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate