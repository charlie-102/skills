#!/usr/bin/env python3
"""
Generate an inference script for an ML image processing repository.

Usage:
    python generate_inference.py --repo /path/to/repo --output inference.py
    python generate_inference.py --repo /path/to/repo --model-class ModelName --output inference.py
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


INFERENCE_TEMPLATE = '''#!/usr/bin/env python3
"""
Inference script for {model_name}

Auto-generated inference script. Please review and customize as needed.

Usage:
    python {output_name} --input image.png --output result.png --weights model.pth
    python {output_name} --input_dir ./inputs --output_dir ./outputs --weights model.pth
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add repository root to path if needed
REPO_ROOT = Path(__file__).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import model - CUSTOMIZE THIS
{model_import}


def load_model(weights_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load model with pretrained weights."""
    # Initialize model - CUSTOMIZE PARAMETERS AS NEEDED
    model = {model_init}

    # Load weights
    state_dict = torch.load(weights_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state_dict, dict):
        # Try common keys for state dict
        for key in ['state_dict', 'model_state_dict', 'model', 'params', 'net', 'G']:
            if key in state_dict:
                state_dict = state_dict[key]
                break

    # Handle DataParallel prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {{k[7:]: v for k, v in state_dict.items()}}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def preprocess(image_path: str, device: str = 'cuda') -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess input image for inference.

    CUSTOMIZE THIS FUNCTION based on the model's expected input:
    - Normalization: [0,1], [-1,1], or ImageNet
    - Color space: RGB vs BGR
    - Size requirements: padding, resizing, etc.
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    # Convert to numpy
    img_np = np.array(img).astype(np.float32)

    # Normalize to [0, 1] - CUSTOMIZE IF NEEDED
    img_np = img_np / 255.0

    # For [-1, 1] normalization, uncomment:
    # img_np = img_np * 2 - 1

    # For ImageNet normalization, uncomment:
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img_np = (img_np - mean) / std

    # Convert to tensor: [H, W, C] -> [1, C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Pad to multiple if needed (common for UNet-style architectures)
    # h, w = img_tensor.shape[2:]
    # pad_h = (8 - h % 8) % 8
    # pad_w = (8 - w % 8) % 8
    # if pad_h > 0 or pad_w > 0:
    #     img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    return img_tensor.to(device), original_size


def postprocess(output_tensor: torch.Tensor, original_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Postprocess model output to PIL Image.

    CUSTOMIZE THIS FUNCTION based on the model's output format.
    """
    # Remove batch dimension and move to CPU
    output = output_tensor.squeeze(0).cpu()

    # Convert to numpy: [C, H, W] -> [H, W, C]
    output_np = output.permute(1, 2, 0).numpy()

    # Denormalize from [0, 1] - CUSTOMIZE IF NEEDED
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

    # For [-1, 1] denormalization, uncomment:
    # output_np = (output_np + 1) / 2
    # output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

    # For ImageNet denormalization, uncomment:
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # output_np = output_np * std + mean
    # output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    result = Image.fromarray(output_np)

    # Resize back to original size if needed
    if original_size is not None:
        result = result.resize(original_size, Image.LANCZOS)

    return result


def process_single_image(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    device: str = 'cuda'
) -> None:
    """Process a single image."""
    # Preprocess
    input_tensor, original_size = preprocess(input_path, device)

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess and save
    result = postprocess(output_tensor, original_size)
    result.save(output_path)


def process_directory(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    device: str = 'cuda',
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
) -> None:
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{{ext}}'))
        image_files.extend(input_path.glob(f'*{{ext.upper()}}'))

    if not image_files:
        print(f"No images found in {{input_dir}}")
        return

    print(f"Processing {{len(image_files)}} images...")

    for img_file in tqdm(image_files, desc='Processing'):
        output_file = output_path / img_file.name
        try:
            process_single_image(model, str(img_file), str(output_file), device)
        except Exception as e:
            print(f"Error processing {{img_file.name}}: {{e}}")
            continue

        # Clear GPU cache periodically
        if device == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='{model_name} Inference')

    # Input/Output
    parser.add_argument('--input', '-i', type=str, help='Input image path')
    parser.add_argument('--output', '-o', type=str, help='Output image path')
    parser.add_argument('--input_dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output_dir', type=str, help='Output directory for batch processing')

    # Model
    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='Path to pretrained weights')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')

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

    # Load model
    print(f"Loading model from {{args.weights}}...")
    model = load_model(args.weights, args.device)
    print("Model loaded successfully")

    # Process
    if args.input:
        print(f"Processing {{args.input}}...")
        process_single_image(model, args.input, args.output, args.device)
        print(f"Result saved to {{args.output}}")
    else:
        process_directory(model, args.input_dir, args.output_dir, args.device)
        print(f"Results saved to {{args.output_dir}}")


if __name__ == '__main__':
    main()
'''


def find_model_classes(repo_path: Path) -> List[Dict]:
    """Find model class definitions in the repository."""
    model_dirs = ['models', 'model', 'networks', 'network', 'arch', 'archs', 'src/model', 'src/models']
    results = []

    for dir_name in model_dirs:
        model_dir = repo_path / dir_name
        if model_dir.exists():
            for f in model_dir.rglob('*.py'):
                if '__pycache__' in str(f) or '__init__' in f.name:
                    continue

                content = f.read_text(errors='ignore')
                classes = re.findall(r'class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\)', content)

                for cls in classes:
                    # Try to find __init__ parameters
                    init_match = re.search(
                        rf'class\s+{cls}\s*\([^)]*\):\s*.*?def\s+__init__\s*\(self([^)]*)\)',
                        content, re.DOTALL
                    )
                    params = init_match.group(1) if init_match else ''

                    results.append({
                        'class': cls,
                        'file': str(f.relative_to(repo_path)),
                        'params': params.strip().strip(',').strip()
                    })

    return results


def generate_model_import(model_info: Dict, repo_name: str) -> str:
    """Generate import statement for a model."""
    file_path = model_info['file']
    module_path = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
    class_name = model_info['class']

    return f"from {module_path} import {class_name}"


def generate_model_init(model_info: Dict) -> str:
    """Generate model initialization code."""
    class_name = model_info['class']
    params = model_info.get('params', '')

    if params:
        # Simple heuristic - add placeholder values
        return f"{class_name}()  # TODO: Add required parameters"
    else:
        return f"{class_name}()"


def generate_inference_script(
    repo_path: Path,
    output_path: str,
    model_class: Optional[str] = None
) -> str:
    """Generate an inference script for the repository."""
    repo_path = Path(repo_path).resolve()

    # Find model classes
    model_classes = find_model_classes(repo_path)

    if not model_classes:
        print("Warning: No model classes found. Using placeholder.")
        model_info = {
            'class': 'Model',
            'file': 'models/model.py',
            'params': ''
        }
    elif model_class:
        # Find specific class
        matches = [m for m in model_classes if m['class'] == model_class]
        if not matches:
            print(f"Warning: Model class '{model_class}' not found. Available: {[m['class'] for m in model_classes]}")
            model_info = model_classes[0]
        else:
            model_info = matches[0]
    else:
        # Use first found class (usually the main model)
        model_info = model_classes[0]
        print(f"Using model class: {model_info['class']} from {model_info['file']}")

    # Generate script
    model_import = generate_model_import(model_info, repo_path.name)
    model_init = generate_model_init(model_info)
    output_name = Path(output_path).name

    script = INFERENCE_TEMPLATE.format(
        model_name=model_info['class'],
        output_name=output_name,
        model_import=model_import,
        model_init=model_init
    )

    return script


def main():
    parser = argparse.ArgumentParser(description='Generate inference script for ML repository')
    parser.add_argument('--repo', '-r', type=str, required=True,
                        help='Path to the ML repository')
    parser.add_argument('--output', '-o', type=str, default='inference.py',
                        help='Output script path')
    parser.add_argument('--model-class', '-m', type=str,
                        help='Specific model class to use')
    parser.add_argument('--list-models', '-l', action='store_true',
                        help='List available model classes')

    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    if args.list_models:
        print("Available model classes:")
        for model in find_model_classes(repo_path):
            print(f"  - {model['class']} ({model['file']})")
        return

    # Generate script
    script = generate_inference_script(repo_path, args.output, args.model_class)

    # Save
    output_path = repo_path / args.output
    output_path.write_text(script)
    print(f"Generated inference script: {output_path}")
    print("\nNext steps:")
    print("1. Review and customize the generated script")
    print("2. Update model import and initialization")
    print("3. Adjust preprocessing/postprocessing as needed")
    print("4. Test with: python {args.output} --input test.png --output result.png --weights model.pth")


if __name__ == '__main__':
    main()
