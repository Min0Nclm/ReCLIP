
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
import json

# SAM imports
from segment_anything import sam_model_registry

# Configure environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

def get_args():
    parser = argparse.ArgumentParser(description="Pre-compute image features using SAM")
    parser.add_argument("--data_source", type=str, default="./data/brainmri", help="Path to the dataset source directory")
    parser.add_argument("--output_file", type=str, default="sam_features.pt", help="Name of the output feature file")
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam_vit_b_01ec64.pth", help="Path to the SAM checkpoint")
    parser.add_argument("--sam_model_type", type=str, default="vit_b", help="SAM model type")
    parser.add_argument("--image_size", type=int, default=1024, help="Image size expected by SAM encoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    return parser.parse_args()

def load_image_paths(source_dir):
    """Loads image paths from the train.json file."""
    image_paths = []
    json_path = os.path.join(source_dir, 'samples', 'train.json')
    with open(json_path, "r") as f:
        for line in f:
            meta = json.loads(line)
            # Join, then normalize to remove redundant separators like './'
            path = os.path.join(source_dir, 'images', meta['filename'])
            norm_path = os.path.normpath(path)
            # Finally, replace separators for consistency
            image_paths.append(norm_path.replace('\\', '/'))
    return image_paths

def preprocess_image(image_path, image_size):
    """Loads and preprocesses an image for SAM."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]
    return img

@torch.no_grad()
def extract_features(image_paths, sam_encoder, image_size, device):
    """Extracts features for a list of images."""
    all_features = []
    print(f"Extracting features from {len(image_paths)} images...")
    for path in tqdm(image_paths):
        img = preprocess_image(path, image_size)
        img = img.unsqueeze(0).to(device)  # Add batch dimension
        feat = sam_encoder(img)
        # Flatten the feature map and move to CPU
        feat_flat = feat.squeeze(0).reshape(-1).cpu()
        all_features.append(feat_flat)
    return torch.stack(all_features)

def main():
    args = get_args()
    print("Starting feature pre-computation...")

    # 1. Load SAM model
    print(f"Loading SAM model ({args.sam_model_type}) from {args.sam_checkpoint}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    sam.eval()
    sam_encoder = sam.image_encoder

    # 2. Get image paths
    image_paths = load_image_paths(args.data_source)
    if not image_paths:
        print(f"Error: No images found in {os.path.join(args.data_source, 'samples', 'train.json')}")
        return

    # 3. Extract features
    features = extract_features(image_paths, sam_encoder, args.image_size, args.device)

    # 4. Save features
    output_path = os.path.join(args.data_source, args.output_file)
    data_to_save = {
        'image_paths': image_paths,
        'features': features
    }
    torch.save(data_to_save, output_path)
    print(f"Successfully saved features to {output_path}")
    print(f"Feature tensor shape: {features.shape}")

if __name__ == "__main__":
    main()
