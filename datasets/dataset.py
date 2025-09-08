import os
import random
from enum import Enum
import PIL
import torch
from torchvision import transforms
import json
from PIL import Image
import numpy as np

import torch.nn.functional as F

from medsyn.tasks import CutPastePatchBlender,SmoothIntensityChangeTask,GaussIntensityChangeTask,SinkDeformationTask,SourceDeformationTask,IdentityTask

# =================================================================================
# K-Means based Support Set Selection
# =================================================================================

def _select_most_representative(features):
    """
    Selects the single most representative sample from a feature set.
    This is approximated by finding the sample closest to the mean of all samples.
    """
    if features.numel() == 0:
        raise ValueError("Feature set cannot be empty.")

    # Calculate the centroid (mean) of all features
    centroid = torch.mean(features, dim=0)

    # Calculate cosine similarity between each feature and the centroid
    sim_to_centroid = F.cosine_similarity(features, centroid.unsqueeze(0))

    # The most representative sample is the one with the highest similarity to the centroid
    return torch.argmax(sim_to_centroid).item()


def _select_support_samples_by_clustering(features, k):
    """
    Selects the single most representative support sample.
    The k-means clustering logic has been removed as per user request
    to default to k=1 logic.
    """
    # The user requested to default to k=1 logic, which is to select the single most representative sample.
    # The k parameter is now ignored.
    if k > 1:
        print(f"Warning: The 'num_support_samples' parameter is ignored and only the single most representative support sample will be used.")
    
    most_representative_idx = _select_most_representative(features)
    return [most_representative_idx]


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
        print(f"Selecting the most representative support sample...")
        self.support_indices = _select_support_samples_by_clustering(self.sam_features, num_support)
        
        # Load the selected support images once to be used for all training items
        self.support_images = [self.read_image(self.sam_image_paths[i]) for i in self.support_indices]
        print(f"Selected support sample indices: {self.support_indices}")

        # --- Add images from data/tubes ---
        try:
            tubes_dir = os.path.join(os.path.dirname(self.source), 'tubes')
            if os.path.isdir(tubes_dir):
                tube_images_paths = [os.path.join(tubes_dir, f) for f in os.listdir(tubes_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if len(tube_images_paths) >= 2:
                    selected_tubes = random.sample(tube_images_paths, 2)
                    print(f"Adding 2 random images from data/tubes: {selected_tubes}")
                    for tube_path in selected_tubes:
                        self.support_images.append(self.read_image(tube_path))
                else:
                    print(f"Warning: Found fewer than 2 images in {tubes_dir}. Not adding tubes.")
            else:
                print(f"Warning: 'data/tubes' directory not found at {tubes_dir}")
        except Exception as e:
            print(f"Error loading images from data/tubes: {e}")
        # --- End of adding tubes images ---
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
        
        is_cutpaste = isinstance(choice_aug, CutPastePatchBlender)

        # To prevent an image from being its own support in CutPaste, 
        # we can temporarily filter it out if the choice is CutPaste.
        if is_cutpaste:
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
            "is_cutpaste": is_cutpaste,
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
        tasks = [
            CutPastePatchBlender(support_images_for_cutpaste),
            IdentityTask()
        ]
        task_probability = [0.5, 0.5]
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