import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from PIL import Image
import json
from tqdm import tqdm

# Assuming open_clip is in the path
import open_clip

def get_image_paths(data_root, dataset_name):
    """Gets all training image paths for a given dataset."""
    image_paths = []
    json_path = os.path.join(data_root, dataset_name, 'samples', 'train.json')
    image_dir = os.path.join(data_root, dataset_name, 'images')
    
    with open(json_path, 'r') as f:
        for line in f:
            meta = json.loads(line)
            # We only precompute features for the training images
            if 'train' in meta['filename']:
                image_paths.append(os.path.join(image_dir, meta['filename']))
    return image_paths

def main(args):
    """
    Main function to precompute image features using a CLIP model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained='laion2b_s34b_b79k')
    model.to(device)
    model.eval()

    # Get image paths
    print(f"Processing dataset: {args.dataset}")
    image_paths = get_image_paths(args.data_root, args.dataset)
    if not image_paths:
        print("No training images found. Exiting.")
        return

    all_features = []
    all_image_paths = []

    # Define a simple dataset for batching
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, paths, preprocess):
            self.paths = paths
            self.preprocess = preprocess
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            img_path = self.paths[idx]
            image = self.preprocess(Image.open(img_path).convert("RGB"))
            return image, img_path

    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Found {len(image_paths)} images. Starting feature extraction...")
    with torch.no_grad():
        for images, paths in tqdm(dataloader):
            images = images.to(device)
            features = model.encode_image(images)
            
            # Normalize features
            features /= features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu())
            all_image_paths.extend(paths)

    # Concatenate all features
    features_tensor = torch.cat(all_features)

    # Save features and paths
    output_path = os.path.join(args.data_root, args.dataset, 'sam_features.pt')
    print(f"Saving {features_tensor.shape[0]} features to {output_path}...")
    
    torch.save({
        'features': features_tensor,
        'image_paths': all_image_paths
    }, output_path)

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute image features for ReCLIP.')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory of the datasets.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to process (e.g., chexpert, busi).')
    parser.add_argument('--model_name', type=str, default='ViT-L-14', help='Name of the CLIP model to use.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader.')
    
    args = parser.parse_args()
    main(args)