import os
import random
from enum import Enum
import PIL
import torch
from torchvision import transforms
import json
from PIL import Image
import numpy as np

from medsyn.tasks import CutPastePatchBlender,SmoothIntensityChangeTask,GaussIntensityChangeTask,SinkDeformationTask,SourceDeformationTask,IdentityTask


class TrainDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        k_shot = -1,
        foreign_sources=None,
        **kwargs,
    ):

        super().__init__()
        self.args = args
        self.source = source
        self.k_shot = k_shot
        self.foreign_sources = foreign_sources
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()
        self.augs,self.augs_pro = self.load_anomaly_syn()
        assert sum(self.augs_pro)==1.0

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path=os.path.join(self.source,'images',info['filename'])
        image = self.read_image(image_path)
        choice_aug = np.random.choice(a=[aug for aug in self.augs],
                                         p = [pro for pro in self.augs_pro],
                                         size=(1,), replace=False)[0]
        image, mask = choice_aug(image)
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)
        return {
            "image": image,
            "mask" : mask,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def read_image(self,path):
        image = PIL.Image.open(path).resize((self.args.image_size,self.args.image_size),
                                            PIL.Image.Resampling.BILINEAR).convert("L")
        image = np.array(image).astype(np.uint8)
        return image

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"train.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        if self.k_shot != -1:
            data_to_iterate = random.sample(
                data_to_iterate, self.k_shot
            )
        return data_to_iterate


    def load_anomaly_syn(self):
        tasks = []
        task_probability = []
        for task_name in self.args.anomaly_tasks.keys():
            if task_name =='CutpasteTask':
                # Load in-domain images
                support_images = [self.read_image(os.path.join(self.source,'images',data['filename'])) for data in self.data_to_iterate]
                
                # Load out-of-domain images if provided
                if self.foreign_sources is not None:
                    foreign_images = []
                    # For k_shot=16, this gives 16 in-domain and 4 out-of-domain images (20% foreign ratio)
                    num_foreign_per_source = 2 
                    for source_path in self.foreign_sources:
                        foreign_data_to_iterate = []
                        json_path = os.path.join(source_path, 'samples', "train.json")
                        if os.path.exists(json_path):
                            with open(json_path, "r") as f_r:
                                for line in f_r:
                                    foreign_data_to_iterate.append(json.loads(line))
                            
                            # Take N samples from the foreign dataset
                            if len(foreign_data_to_iterate) > num_foreign_per_source:
                                foreign_samples = random.sample(foreign_data_to_iterate, num_foreign_per_source)
                            else:
                                foreign_samples = foreign_data_to_iterate
                            
                            for data in foreign_samples:
                                image_path = os.path.join(source_path, 'images', data['filename'])
                                if os.path.exists(image_path):
                                    foreign_images.append(self.read_image(image_path))
                    
                    support_images.extend(foreign_images)

                # --- Add images from data/mpdd ---
                try:
                    mpdd_dir = os.path.join(os.path.dirname(self.source), 'mpdd', 'image')
                    if os.path.isdir(mpdd_dir):
                        mpdd_images_paths = [os.path.join(mpdd_dir, f) for f in os.listdir(mpdd_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                        if len(mpdd_images_paths) >= 2:
                            selected_mpdd = random.sample(mpdd_images_paths, 2)
                            print(f"Adding 2 random images from data/mpdd: {selected_mpdd}")
                            for mpdd_path in selected_mpdd:
                                support_images.append(self.read_image(mpdd_path))
                        else:
                            print(f"Warning: Found fewer than 2 images in {mpdd_dir}. Not adding mpdd images.")
                    else:
                        print(f"Warning: 'data/mpdd/image' directory not found at {mpdd_dir}")
                except Exception as e:
                    print(f"Error loading images from data/mpdd: {e}")
                # --- End of adding mpdd images ---

                task = CutPastePatchBlender(support_images)
            elif task_name == 'SmoothIntensityTask':
                task = SmoothIntensityChangeTask(30.0)
            elif task_name == 'GaussIntensityChangeTask':
                task = GaussIntensityChangeTask()
            elif task_name == 'SinkTask':
                task = SinkDeformationTask()
            elif task_name == 'SourceTask':
                task = SourceDeformationTask()
            elif task_name == 'IdentityTask':
                task = IdentityTask()
            else:
                raise NotImplementedError("task must in [CutpasteTask, "
                                          "SmoothIntensityTask, "
                                          "GaussIntensityChangeTask,"
                                          "SinkTask, SourceTask, IdentityTask]")

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
        mask = np.zeros((self.args.image_size,self.args.image_size)).astype(float)
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
        mask = np.zeros((self.args.image_size,self.args.image_size)).astype(float)
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
            mask = np.array(mask).astype(float)/255.0
            mask [mask!=0.0] = 1.0
        else:
            mask = np.zeros((self.args.image_size,self.args.image_size)).astype(float)

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