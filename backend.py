# backend.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import glob
import warnings
import random


from model import MRI_Synthesis_Net

warnings.filterwarnings("ignore", message="The given buffer is not writable, and PyTorch does not support non-writable tensors.")

# =====================================================================================
# ## Helper Functions & Classes
# =====================================================================================

class FlexibleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, file_paths=None):
        self.transform = transform
        if file_paths:
            self.image_paths = file_paths
        else:
            self.image_paths = []
            supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            for ext in supported_extensions:
                self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        if not self.image_paths:
            raise FileNotFoundError(f"No supported images found in: {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)

def calculate_batch_histograms(images_batch, num_bins, mask=None):
    B, C, H, W = images_batch.shape
    device = images_batch.device
    images_denorm = images_batch * 0.5 + 0.5
    if mask is None:
        mask = torch.ones_like(images_batch, dtype=torch.bool)
    bin_indices = (images_denorm * (num_bins - 1)).long()
    batch_offsets = torch.arange(B, device=device) * num_bins
    offset_indices = bin_indices + batch_offsets.view(B, 1, 1, 1)
    flat_hist = torch.zeros(B * num_bins, device=device)
    flat_indices_to_scatter = offset_indices[mask]
    flat_hist.scatter_add_(0, flat_indices_to_scatter, torch.ones_like(flat_indices_to_scatter, dtype=flat_hist.dtype))
    return flat_hist.view(B, num_bins)

def create_range_translation_guidance_map(input_image, perms, num_chunks, dark_threshold, shuffle_dark_chunk):
    B, _, H, W = input_image.shape
    device = input_image.device
    images_denorm = input_image * 0.5 + 0.5
    
    if shuffle_dark_chunk:
        # --- Logic for fully shufflable histogram ---
        original_chunk_idx = (images_denorm * num_chunks).long()
        original_chunk_idx = torch.clamp(original_chunk_idx, 0, num_chunks - 1)
        chunk_width = 1.0 / num_chunks
        chunk_lower_bound = original_chunk_idx.float() * chunk_width
        relative_pos = (images_denorm - chunk_lower_bound) / chunk_width
        new_value_base = 0.0
        new_value_range = 1.0
    else:
        # --- Logic for histogram with a fixed dark chunk ---
        background_mask = (images_denorm < dark_threshold)
        fg_pixels_01 = (images_denorm - dark_threshold) / (1.0 - dark_threshold)
        fg_pixels_01 = torch.clamp(fg_pixels_01, 0.0, 1.0)
        original_chunk_idx = (fg_pixels_01 * num_chunks).long()
        original_chunk_idx = torch.clamp(original_chunk_idx, 0, num_chunks - 1)
        chunk_width = 1.0 / num_chunks
        chunk_lower_bound = original_chunk_idx.float() * chunk_width
        relative_pos = (fg_pixels_01 - chunk_lower_bound) / chunk_width
        new_value_base = dark_threshold
        new_value_range = 1.0 - dark_threshold

    relative_pos = torch.clamp(relative_pos, 0.0, 1.0)
    batch_indices = torch.arange(B, device=device).view(B, 1, 1)
    target_chunk_idx = perms[batch_indices, original_chunk_idx.squeeze(1)].unsqueeze(1)
    target_chunk_lower_bound = target_chunk_idx.float() * chunk_width
    
    new_fg_value_01 = target_chunk_lower_bound + relative_pos * chunk_width
    new_value_denorm = new_fg_value_01 * new_value_range + new_value_base
    
    if shuffle_dark_chunk:
        final_map_denorm = new_value_denorm
    else:
        final_map_denorm = torch.where(background_mask, images_denorm, new_value_denorm)

    guidance_map = final_map_denorm * 2.0 - 1.0
    return guidance_map
    
def apply_permutation(hist_shufflable, perms, num_chunks):
    B, _ = hist_shufflable.shape
    chunk_size = hist_shufflable.shape[1] // num_chunks
    original_chunks = hist_shufflable.view(B, num_chunks, chunk_size)
    perms_expanded = perms.unsqueeze(-1).expand(-1, -1, chunk_size)
    shuffled_chunks = torch.gather(original_chunks, dim=1, index=perms_expanded)
    return shuffled_chunks.view(B, -1)

# =====================================================================================
# ## Histogram Calculation for Editor
# =====================================================================================
def get_base_histogram(image_paths, num_bins, dark_threshold, update_callback=None, shuffle_dark_chunk=False):
    if not image_paths:
        return None, None, "No image paths provided."

    try:
        if update_callback: update_callback("Calculating base histogram...")
        mri_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        dataset = FlexibleImageDataset(image_dir=None, transform=mri_transform, file_paths=image_paths)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_hist_fixed = torch.zeros(1, num_bins)
        total_hist_shufflable = torch.zeros(1, num_bins)

        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                if update_callback: update_callback(f"Processing batch {i+1}/{len(loader)} for histogram...")
                images = images.to(device)
                
                if shuffle_dark_chunk:
                    batch_hist = calculate_batch_histograms(images, num_bins, mask=None)
                    total_hist_shufflable += batch_hist.sum(dim=0, keepdim=True).cpu()
                else:
                    images_denorm = images * 0.5 + 0.5
                    background_mask = (images_denorm < dark_threshold)
                    batch_hist_fixed = calculate_batch_histograms(images, num_bins, mask=background_mask)
                    batch_hist_shufflable = calculate_batch_histograms(images, num_bins, mask=~background_mask)
                    total_hist_fixed += batch_hist_fixed.sum(dim=0, keepdim=True).cpu()
                    total_hist_shufflable += batch_hist_shufflable.sum(dim=0, keepdim=True).cpu()
        
        avg_hist_fixed = total_hist_fixed / len(dataset)
        avg_hist_shufflable = total_hist_shufflable / len(dataset)
        
        if update_callback: update_callback("Base histogram calculated successfully.")
        return avg_hist_fixed, avg_hist_shufflable, "Success"

    except Exception as e:
        return None, None, f"Error calculating histogram: {e}"

# =====================================================================================
# ## Main Inference Function
# =====================================================================================
def generate_contrasts(
    input_dir: str,
    output_dir: str,
    generation_mode: str,
    num_bins: int,
    hist_chunks: int,
    dark_threshold: float,
    update_callback,
    num_contrasts: int = 5,
    fixed_contrasts: bool = True,
    custom_permutations: list = None
):
    try:
        MODEL_PATH = "mri_contrast_generator.pth"
        if update_callback: update_callback("Setting up device and model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MRI_Synthesis_Net(scale_factor=1, num_hist_bins=num_bins).to(device)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        else:
            if update_callback: update_callback(f"⚠️ Warning: Model file '{MODEL_PATH}' not found. Using placeholder model.")
        model.eval()

        if update_callback: update_callback("Loading dataset...")
        dataset = FlexibleImageDataset(image_dir=input_dir, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]))
        
        total_images = len(dataset)
        with torch.no_grad():
            for i, (image, original_fname) in enumerate(dataset):
                status = f"Processing image {i+1}/{total_images}: {original_fname}"
                if update_callback: update_callback(status)
                original_image_batch = image.unsqueeze(0).to(device)

                if generation_mode == 'random':
                    images_denorm = original_image_batch * 0.5 + 0.5
                    background_mask = (images_denorm < dark_threshold)
                    hist_fixed = calculate_batch_histograms(original_image_batch, num_bins, mask=background_mask)
                    hist_shufflable = calculate_batch_histograms(original_image_batch, num_bins, mask=~background_mask)
                    
                    perms_to_run = [torch.rand(1, hist_chunks, device=device).argsort(dim=1) for _ in range(num_contrasts)]
                    if fixed_contrasts and i > 0: # Use the same perms as the first image
                        perms_to_run = first_image_perms
                    if i == 0:
                        first_image_perms = perms_to_run

                    for j, perm in enumerate(perms_to_run):
                        shuffled_part = apply_permutation(hist_shufflable, perm, hist_chunks)
                        target_hist = hist_fixed + shuffled_part
                        guidance_map = create_range_translation_guidance_map(original_image_batch, perm, hist_chunks, dark_threshold, shuffle_dark_chunk=False)
                        generated_output, _ = model(original_image_batch, target_hist, guidance_map)
                        
                        base_name, _ = os.path.splitext(original_fname)
                        contrast_name = f"random_contrast_{j+1:02d}"
                        save_dir = os.path.join(output_dir, contrast_name)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{base_name}.png")
                        save_image(generated_output.squeeze(0), save_path, normalize=True)

                elif generation_mode == 'custom':
                    for j, (perm_tuple) in enumerate(custom_permutations):
                        perm, shuffle_dark = perm_tuple
                        perm = perm.to(device)

                        if shuffle_dark:
                            hist_shufflable = calculate_batch_histograms(original_image_batch, num_bins, mask=None)
                            target_hist = apply_permutation(hist_shufflable, perm, hist_chunks)
                        else:
                            images_denorm = original_image_batch * 0.5 + 0.5
                            background_mask = (images_denorm < dark_threshold)
                            hist_fixed = calculate_batch_histograms(original_image_batch, num_bins, mask=background_mask)
                            hist_shufflable = calculate_batch_histograms(original_image_batch, num_bins, mask=~background_mask)
                            shuffled_part = apply_permutation(hist_shufflable, perm, hist_chunks)
                            target_hist = hist_fixed + shuffled_part

                        guidance_map = create_range_translation_guidance_map(original_image_batch, perm, hist_chunks, dark_threshold, shuffle_dark_chunk=shuffle_dark)
                        generated_output, _ = model(original_image_batch, target_hist, guidance_map)

                        base_name, _ = os.path.splitext(original_fname)
                        perm_str = '_'.join(map(str, perm.cpu().numpy().flatten()))
                        contrast_name = f"custom_contrast_{j+1}_{perm_str[:15]}"
                        save_dir = os.path.join(output_dir, contrast_name)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{base_name}.png")
                        save_image(generated_output.squeeze(0), save_path, normalize=True)

        if update_callback: update_callback(f"✅ Generation complete! Results saved.")
        return True, "Success"

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"An unexpected error occurred: {e}"
        if update_callback: update_callback(f"❌ Error: {error_msg}")
        return False, error_msg
