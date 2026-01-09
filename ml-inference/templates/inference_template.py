#!/usr/bin/env python3
"""
Universal Inference Template for Image Processing Models

This template provides a complete inference pipeline that can be customized
for various image processing tasks (denoising, super-resolution, restoration, etc.)

Usage:
    python inference_template.py --input image.png --output result.png --weights model.pth
    python inference_template.py --input_dir ./inputs --output_dir ./outputs --weights model.pth

Customization Points (marked with # CUSTOMIZE):
    1. Model import and initialization
    2. Preprocessing (normalization, color space, padding)
    3. Postprocessing (denormalization, clipping)
    4. Tile processing parameters (for large images)
"""

import argparse
import gc
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# =============================================================================
# CUSTOMIZE: Model Import
# =============================================================================
# Uncomment and modify based on your repository structure:

# Option 1: Import from models directory
# from models.network import YourModel

# Option 2: Import from specific file
# from model import YourModel

# Option 3: Add repo to path first
# REPO_ROOT = Path(__file__).parent
# sys.path.insert(0, str(REPO_ROOT))
# from models.architecture import YourModel

# Placeholder for demonstration
class PlaceholderModel(nn.Module):
    """Replace this with your actual model import."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


# =============================================================================
# Configuration
# =============================================================================

class InferenceConfig:
    """Configuration for inference pipeline."""

    # CUSTOMIZE: Model configuration
    MODEL_CLASS = PlaceholderModel  # Replace with your model class
    MODEL_KWARGS = {}  # Add initialization kwargs if needed

    # CUSTOMIZE: Multi-network architecture (optional, e.g., DASR-style)
    # Uncomment and set these for repos with predictor + generator pattern
    # AUX_MODEL_CLASS = PredictorModel  # Auxiliary/predictor network class
    # AUX_MODEL_KWARGS = {}  # Auxiliary model initialization kwargs

    # CUSTOMIZE: Preprocessing
    # Options: 'zero_one' ([0,1]), 'neg_one_one' ([-1,1]), 'imagenet'
    NORMALIZATION = 'zero_one'

    # ImageNet mean/std (used if NORMALIZATION == 'imagenet')
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # CUSTOMIZE: Color space
    # True if model expects BGR input (e.g., trained with OpenCV)
    USE_BGR = False

    # CUSTOMIZE: Padding
    # Set to 8, 16, 32, etc. if model requires input size to be multiple
    PAD_MULTIPLE = None

    # CUSTOMIZE: Tile processing for large images
    TILE_SIZE = 512  # Tile size for processing large images
    TILE_OVERLAP = 32  # Overlap between tiles

    # CUSTOMIZE: Scale factor (for super-resolution models)
    SCALE_FACTOR = 1  # Set to 2, 4, etc. for SR models

    # Supported image extensions
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    weights_path: str,
    device: str = 'cuda',
    config: InferenceConfig = InferenceConfig(),
    weights_path_aux: Optional[str] = None
) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
    """
    Load model with pretrained weights.

    Args:
        weights_path: Path to model weights (or primary network weights for multi-network)
        device: Device to load model on
        config: Inference configuration
        weights_path_aux: Path to auxiliary network weights (for multi-network architectures)

    Returns:
        Loaded model in eval mode, or tuple (model, aux_model) for multi-network
    """
    # Initialize model
    model = config.MODEL_CLASS(**config.MODEL_KWARGS)

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)

    # Extract state dict from checkpoint
    if isinstance(checkpoint, dict):
        # Try common keys
        state_dict = None
        for key in ['state_dict', 'model_state_dict', 'model', 'params', 'net', 'G', 'generator']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Handle DataParallel saved models
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Multi-network support (e.g., DASR with predictor + generator)
    if weights_path_aux is not None and hasattr(config, 'AUX_MODEL_CLASS'):
        aux_model = config.AUX_MODEL_CLASS(**config.AUX_MODEL_KWARGS)
        aux_checkpoint = torch.load(weights_path_aux, map_location=device)

        # Extract state dict
        if isinstance(aux_checkpoint, dict):
            aux_state_dict = None
            for key in ['state_dict', 'model_state_dict', 'model', 'params']:
                if key in aux_checkpoint:
                    aux_state_dict = aux_checkpoint[key]
                    break
            if aux_state_dict is None:
                aux_state_dict = aux_checkpoint
        else:
            aux_state_dict = aux_checkpoint

        # Handle DataParallel
        if list(aux_state_dict.keys())[0].startswith('module.'):
            aux_state_dict = {k[7:]: v for k, v in aux_state_dict.items()}

        aux_model.load_state_dict(aux_state_dict, strict=True)
        aux_model.to(device)
        aux_model.eval()

        return model, aux_model

    return model


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess(
    image_path: str,
    device: str = 'cuda',
    config: InferenceConfig = InferenceConfig()
) -> Tuple[torch.Tensor, dict]:
    """
    Preprocess input image for inference.

    Args:
        image_path: Path to input image
        device: Target device
        config: Inference configuration

    Returns:
        Tuple of (preprocessed tensor, metadata dict)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    # Convert to numpy
    img_np = np.array(img).astype(np.float32)

    # Color space conversion
    if config.USE_BGR:
        img_np = img_np[:, :, ::-1].copy()

    # Normalization
    if config.NORMALIZATION == 'zero_one':
        img_np = img_np / 255.0
    elif config.NORMALIZATION == 'neg_one_one':
        img_np = img_np / 255.0 * 2 - 1
    elif config.NORMALIZATION == 'imagenet':
        img_np = img_np / 255.0
        mean = np.array(config.IMAGENET_MEAN)
        std = np.array(config.IMAGENET_STD)
        img_np = (img_np - mean) / std
    else:
        raise ValueError(f"Unknown normalization: {config.NORMALIZATION}")

    # Convert to tensor: [H, W, C] -> [1, C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()

    # Padding
    pad_info = None
    if config.PAD_MULTIPLE is not None:
        h, w = img_tensor.shape[2:]
        pad_h = (config.PAD_MULTIPLE - h % config.PAD_MULTIPLE) % config.PAD_MULTIPLE
        pad_w = (config.PAD_MULTIPLE - w % config.PAD_MULTIPLE) % config.PAD_MULTIPLE
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(
                img_tensor, (0, pad_w, 0, pad_h), mode='reflect'
            )
            pad_info = {'h': pad_h, 'w': pad_w}

    metadata = {
        'original_size': original_size,
        'pad_info': pad_info,
    }

    return img_tensor.to(device), metadata


# =============================================================================
# Postprocessing
# =============================================================================

def postprocess(
    output_tensor: torch.Tensor,
    metadata: dict,
    config: InferenceConfig = InferenceConfig()
) -> Image.Image:
    """
    Postprocess model output to PIL Image.

    Args:
        output_tensor: Model output tensor [B, C, H, W]
        metadata: Preprocessing metadata
        config: Inference configuration

    Returns:
        PIL Image
    """
    # Remove batch dimension and move to CPU
    output = output_tensor.squeeze(0).cpu().float()

    # Convert to numpy: [C, H, W] -> [H, W, C]
    output_np = output.permute(1, 2, 0).numpy()

    # Remove padding
    if metadata.get('pad_info') is not None:
        pad_info = metadata['pad_info']
        h, w = output_np.shape[:2]
        scaled_pad_h = pad_info['h'] * config.SCALE_FACTOR
        scaled_pad_w = pad_info['w'] * config.SCALE_FACTOR
        if scaled_pad_h > 0:
            output_np = output_np[:-scaled_pad_h, :, :]
        if scaled_pad_w > 0:
            output_np = output_np[:, :-scaled_pad_w, :]

    # Denormalization
    if config.NORMALIZATION == 'zero_one':
        output_np = output_np
    elif config.NORMALIZATION == 'neg_one_one':
        output_np = (output_np + 1) / 2
    elif config.NORMALIZATION == 'imagenet':
        mean = np.array(config.IMAGENET_MEAN)
        std = np.array(config.IMAGENET_STD)
        output_np = output_np * std + mean

    # Color space conversion back
    if config.USE_BGR:
        output_np = output_np[:, :, ::-1].copy()

    # Clip and convert to uint8
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    result = Image.fromarray(output_np)

    return result


# =============================================================================
# Tile Processing (for large images)
# =============================================================================

def process_tiles(
    img_tensor: torch.Tensor,
    model: nn.Module,
    config: InferenceConfig = InferenceConfig()
) -> torch.Tensor:
    """
    Process large images in tiles to avoid memory issues.

    Args:
        img_tensor: Input tensor [1, C, H, W]
        model: Model for inference
        config: Inference configuration

    Returns:
        Output tensor
    """
    _, c, h, w = img_tensor.shape
    tile_size = config.TILE_SIZE
    overlap = config.TILE_OVERLAP
    scale = config.SCALE_FACTOR

    # Output dimensions
    out_h = h * scale
    out_w = w * scale
    out_tile_size = tile_size * scale
    out_overlap = overlap * scale

    # Initialize output
    output = torch.zeros(1, c, out_h, out_w, device=img_tensor.device)
    count = torch.zeros(1, 1, out_h, out_w, device=img_tensor.device)

    # Create weight mask for blending
    weight = create_weight_mask(out_tile_size, out_overlap, img_tensor.device)

    # Process tiles
    stride = tile_size - overlap
    n_tiles_h = math.ceil((h - overlap) / stride)
    n_tiles_w = math.ceil((w - overlap) / stride)

    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Input tile coordinates
            y1 = min(i * stride, h - tile_size)
            x1 = min(j * stride, w - tile_size)
            y2 = y1 + tile_size
            x2 = x1 + tile_size

            # Extract and process tile
            tile = img_tensor[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                tile_out = model(tile)

            # Output coordinates
            out_y1 = y1 * scale
            out_x1 = x1 * scale
            out_y2 = out_y1 + out_tile_size
            out_x2 = out_x1 + out_tile_size

            # Add to output with blending
            output[:, :, out_y1:out_y2, out_x1:out_x2] += tile_out * weight
            count[:, :, out_y1:out_y2, out_x1:out_x2] += weight

    # Normalize by count
    output = output / count.clamp(min=1e-8)

    return output


def create_weight_mask(size: int, overlap: int, device: str) -> torch.Tensor:
    """Create weight mask for tile blending."""
    weight = torch.ones(1, 1, size, size, device=device)

    if overlap > 0:
        # Create linear ramp for edges
        ramp = torch.linspace(0, 1, overlap, device=device)

        # Apply to all edges
        weight[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
        weight[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
        weight[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
        weight[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)

    return weight


# =============================================================================
# Main Inference Functions
# =============================================================================

def process_single_image(
    model: nn.Module,
    input_path: str,
    output_path: str,
    device: str = 'cuda',
    use_tiles: bool = False,
    config: InferenceConfig = InferenceConfig()
) -> None:
    """
    Process a single image.

    Args:
        model: Loaded model
        input_path: Input image path
        output_path: Output image path
        device: Device for inference
        use_tiles: Whether to use tile processing
        config: Inference configuration
    """
    # Preprocess
    input_tensor, metadata = preprocess(input_path, device, config)

    # Inference
    with torch.no_grad():
        if use_tiles:
            output_tensor = process_tiles(input_tensor, model, config)
        else:
            output_tensor = model(input_tensor)

    # Postprocess and save
    result = postprocess(output_tensor, metadata, config)
    result.save(output_path)


def process_directory(
    model: nn.Module,
    input_dir: str,
    output_dir: str,
    device: str = 'cuda',
    use_tiles: bool = False,
    config: InferenceConfig = InferenceConfig()
) -> None:
    """
    Process all images in a directory.

    Args:
        model: Loaded model
        input_dir: Input directory path
        output_dir: Output directory path
        device: Device for inference
        use_tiles: Whether to use tile processing
        config: Inference configuration
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = []
    for ext in config.IMAGE_EXTENSIONS:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    for img_file in tqdm(image_files, desc='Processing'):
        output_file = output_path / img_file.name
        try:
            process_single_image(
                model, str(img_file), str(output_file),
                device, use_tiles, config
            )
        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            continue

        # Memory management
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Image Processing Model Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    python %(prog)s --input img.png --output result.png --weights model.pth

  Directory:
    python %(prog)s --input_dir ./inputs --output_dir ./outputs --weights model.pth

  With tile processing (for large images):
    python %(prog)s --input img.png --output result.png --weights model.pth --tiles
        """
    )

    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', '-i', type=str,
                          help='Input image path')
    io_group.add_argument('--output', '-o', type=str,
                          help='Output image path')
    io_group.add_argument('--input_dir', type=str,
                          help='Input directory for batch processing')
    io_group.add_argument('--output_dir', type=str,
                          help='Output directory for batch processing')

    # Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--weights', '-w', type=str, required=True,
                             help='Path to pretrained weights')
    model_group.add_argument('--weights_aux', type=str,
                             help='Path to auxiliary network weights (for multi-network architectures)')

    # Processing
    proc_group = parser.add_argument_group('Processing')
    proc_group.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'],
                            help='Device to use (default: cuda)')
    proc_group.add_argument('--tiles', action='store_true',
                            help='Use tile processing for large images')
    proc_group.add_argument('--tile_size', type=int, default=512,
                            help='Tile size for tile processing (default: 512)')
    proc_group.add_argument('--tile_overlap', type=int, default=32,
                            help='Tile overlap (default: 32)')

    args = parser.parse_args()

    # Validate arguments
    if args.input and args.input_dir:
        parser.error("Specify either --input or --input_dir, not both")
    if not args.input and not args.input_dir:
        parser.error("Specify either --input or --input_dir")
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Update config
    config = InferenceConfig()
    config.TILE_SIZE = args.tile_size
    config.TILE_OVERLAP = args.tile_overlap

    # Load model
    print(f"Loading model from {args.weights}...")
    model = load_model(args.weights, args.device, config, args.weights_aux)
    if isinstance(model, tuple):
        print(f"Multi-network model loaded on {args.device} (primary + auxiliary)")
    else:
        print(f"Model loaded on {args.device}")

    # Process
    if args.input:
        print(f"Processing {args.input}...")
        process_single_image(
            model, args.input, args.output,
            args.device, args.tiles, config
        )
        print(f"Result saved to {args.output}")
    else:
        process_directory(
            model, args.input_dir, args.output_dir,
            args.device, args.tiles, config
        )
        print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
