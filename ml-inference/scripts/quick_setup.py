#!/usr/bin/env python3
"""
Quick Setup - Single command to analyze repo, fix common issues, and generate inference script.

This script consolidates multiple steps to minimize token usage:
1. Analyze repository structure
2. Auto-fix common import issues (thop, ptflops, etc.)
3. Generate a working inference script
4. Create Dockerfile for containerized testing

Usage:
    python quick_setup.py /path/to/repo
    python quick_setup.py /path/to/repo --docker-only
    python quick_setup.py /path/to/repo --fix-imports
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Auto-fix Common Import Issues
# =============================================================================

OPTIONAL_IMPORTS = {
    'thop': 'from thop import profile',
    'ptflops': 'from ptflops import get_model_complexity_info',
    'fvcore': 'from fvcore',
}

FIX_TEMPLATES = {
    'thop': '''try:
    from thop import profile
except ImportError:
    profile = None  # thop is optional, only used for profiling''',

    'ptflops': '''try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None  # ptflops is optional''',

    'fvcore': '''try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None  # fvcore is optional''',
}


def find_and_fix_imports(repo_path: Path, dry_run: bool = False) -> List[Dict]:
    """Find and fix optional import issues."""
    fixes = []

    for py_file in repo_path.rglob('*.py'):
        if '__pycache__' in str(py_file) or '.git' in str(py_file):
            continue

        try:
            content = py_file.read_text()
            original = content

            for pkg, import_pattern in OPTIONAL_IMPORTS.items():
                if import_pattern in content and f'try:\n    {import_pattern}' not in content:
                    # Check if this import would fail
                    fix_template = FIX_TEMPLATES.get(pkg)
                    if fix_template:
                        # Simple replacement - find the import line and wrap it
                        lines = content.split('\n')
                        new_lines = []
                        i = 0
                        while i < len(lines):
                            line = lines[i]
                            if import_pattern in line and not line.strip().startswith('#'):
                                # Replace with try/except block
                                indent = len(line) - len(line.lstrip())
                                indent_str = ' ' * indent
                                new_lines.append(f'{indent_str}try:')
                                new_lines.append(f'{indent_str}    {line.strip()}')
                                new_lines.append(f'{indent_str}except ImportError:')
                                # Determine variable name
                                if 'import' in line:
                                    if 'from' in line and 'import' in line:
                                        var_name = line.split('import')[-1].strip().split()[0].rstrip(',')
                                    else:
                                        var_name = line.split('import')[-1].strip().split()[0]
                                    new_lines.append(f'{indent_str}    {var_name} = None  # {pkg} is optional')
                                fixes.append({
                                    'file': str(py_file.relative_to(repo_path)),
                                    'package': pkg,
                                    'line': i + 1
                                })
                            else:
                                new_lines.append(line)
                            i += 1
                        content = '\n'.join(new_lines)

            if content != original and not dry_run:
                py_file.write_text(content)

        except Exception as e:
            continue

    return fixes


# =============================================================================
# Repository Analysis (Condensed)
# =============================================================================

def analyze_repo_quick(repo_path: Path) -> Dict:
    """Quick repository analysis - returns essential info only."""
    result = {
        'name': repo_path.name,
        'framework': None,
        'model_files': [],
        'inference_scripts': [],
        'weight_locations': [],
        'preprocessing': None,
        'multi_network': False,
        'diffusion_model': False,
        'cuda_hardcoded': False,
        'install_method': None,
    }

    # Detect framework
    if (repo_path / 'basicsr').exists():
        result['framework'] = 'basicsr'
        result['install_method'] = 'pip install -e .'
    elif (repo_path / 'mmedit').exists() or (repo_path / 'mmcv').exists():
        result['framework'] = 'mmcv'
        result['install_method'] = 'pip install -e .'
    elif (repo_path / 'setup.py').exists():
        result['install_method'] = 'pip install -e .'
    else:
        result['install_method'] = 'pip install -r requirements.txt'

    # Find model files (limit search depth)
    model_dirs = ['models', 'model', 'archs', 'arch', 'networks', 'basicsr/archs']
    for dir_name in model_dirs:
        model_dir = repo_path / dir_name
        if model_dir.exists():
            for f in model_dir.glob('*.py'):
                if '__init__' not in f.name:
                    content = f.read_text(errors='ignore')
                    classes = re.findall(r'class\s+(\w+)\s*\([^)]*nn\.Module', content)
                    if classes:
                        result['model_files'].append({
                            'path': str(f.relative_to(repo_path)),
                            'classes': classes[:5]  # Limit to 5
                        })

    # Find inference scripts (check root level first, then subdirs)
    inference_patterns = [
        'inference*.py', 'inference_*.py', 'test*.py', 'demo*.py', 'app.py',  # root level
        '**/inference*.py', '**/inference_*.py', '**/test*.py', '**/demo*.py', '**/main_test*.py'
    ]
    for pattern in inference_patterns:
        for f in repo_path.glob(pattern):
            if '__pycache__' not in str(f) and str(f.relative_to(repo_path)) not in result['inference_scripts']:
                content = f.read_text(errors='ignore')
                if 'load_state_dict' in content or 'torch.load' in content or 'argparse' in content:
                    result['inference_scripts'].append(str(f.relative_to(repo_path)))
                    if len(result['inference_scripts']) >= 5:
                        break

    # Find weight locations
    for dir_name in ['model_zoo', 'pretrained', 'weights', 'checkpoints', 'experiments/pretrained_models']:
        if (repo_path / dir_name).exists():
            result['weight_locations'].append(dir_name)

    # Check for multi-network, diffusion models, and CUDA hardcoding
    for f in repo_path.rglob('*.py'):
        if '__pycache__' in str(f):
            continue
        try:
            content = f.read_text(errors='ignore')
            if re.search(r'net_[gp]|self\.net_g.*self\.net_p', content):
                result['multi_network'] = True
            # Detect diffusion models
            if re.search(r'diffusion|Diffusion|DDPM|DDIM|sampler.*step|noise_schedule', content):
                result['diffusion_model'] = True
            # Detect CUDA hardcoding
            if '.cuda()' in content and 'if.*cuda.*available' not in content:
                result['cuda_hardcoded'] = True
        except:
            continue

    # Detect preprocessing (sample first inference script)
    for script in result['inference_scripts'][:1]:
        try:
            content = (repo_path / script).read_text(errors='ignore')
            if '/ 255' in content:
                result['preprocessing'] = '[0, 1]'
            elif '* 2 - 1' in content:
                result['preprocessing'] = '[-1, 1]'
        except:
            pass

    return result


# =============================================================================
# Dockerfile Generation
# =============================================================================

DOCKERFILE_TEMPLATE = '''FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    libgl1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version for smaller image, change for GPU)
RUN pip install --no-cache-dir \\
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install common ML dependencies
RUN pip install --no-cache-dir \\
    numpy \\
    pillow \\
    opencv-python-headless \\
    scipy \\
    tqdm \\
    pyyaml \\
    einops \\
    timm

# Copy repository
COPY . /app/

{install_cmd}

# Create directories for weights and I/O
RUN mkdir -p /app/weights /app/input /app/output

ENV PYTHONPATH=/app:$PYTHONPATH

# Default command - can be overridden
CMD ["python", "--version"]
'''

def generate_dockerfile(repo_path: Path, analysis: Dict) -> str:
    """Generate Dockerfile based on analysis."""
    install_cmd = ''

    if analysis['install_method'] == 'pip install -e .':
        install_cmd = '# Install as package\nRUN pip install --no-cache-dir -e .'
    elif (repo_path / 'requirements.txt').exists():
        install_cmd = '# Install requirements\nRUN pip install --no-cache-dir -r requirements.txt || true'

    return DOCKERFILE_TEMPLATE.format(install_cmd=install_cmd)


# =============================================================================
# Inference Script Generation
# =============================================================================

def generate_inference_script(repo_path: Path, analysis: Dict) -> str:
    """Generate a minimal inference script based on analysis."""

    # Find the main model class
    model_import = "# TODO: Update model import"
    model_class = "Model"

    if analysis['model_files']:
        first_model = analysis['model_files'][0]
        model_path = first_model['path'].replace('/', '.').replace('.py', '')
        if first_model['classes']:
            model_class = first_model['classes'][0]
            model_import = f"from {model_path} import {model_class}"

    # Determine preprocessing
    preprocess = "img / 255.0"
    postprocess = "np.clip(output * 255.0, 0, 255).astype(np.uint8)"
    if analysis['preprocessing'] == '[-1, 1]':
        preprocess = "img / 255.0 * 2 - 1"
        postprocess = "np.clip((output + 1) / 2 * 255.0, 0, 255).astype(np.uint8)"

    script = f'''#!/usr/bin/env python3
"""Auto-generated inference script for {analysis['name']}"""

import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# Model import - verify this is correct
{model_import}


def load_model(weights_path: str, device: str = 'cuda'):
    """Load model with pretrained weights."""
    model = {model_class}()  # TODO: Add constructor args if needed

    checkpoint = torch.load(weights_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        for key in ['state_dict', 'model_state_dict', 'model', 'params']:
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    # Handle DataParallel prefix
    if list(checkpoint.keys())[0].startswith('module.'):
        checkpoint = {{k[7:]: v for k, v in checkpoint.items()}}

    model.load_state_dict(checkpoint, strict=True)
    model.to(device).eval()
    return model


def inference(model, image_path: str, device: str = 'cuda'):
    """Run inference on a single image."""
    # Load and preprocess
    img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)
    img = {preprocess}
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

    # Postprocess
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = {postprocess}
    return Image.fromarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input image')
    parser.add_argument('--output', '-o', required=True, help='Output image')
    parser.add_argument('--weights', '-w', required=True, help='Model weights')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    result = inference(model, args.input, args.device)
    result.save(args.output)
    print(f"Saved to {{args.output}}")


if __name__ == '__main__':
    main()
'''
    return script


# =============================================================================
# Docker Commands Generator
# =============================================================================

def generate_docker_commands(repo_name: str) -> str:
    """Generate Docker build and run commands."""
    return f'''# Docker Commands for {repo_name}

# Build the image
docker build -t {repo_name.lower()}-inference .

# Run inference (mount volumes for weights, input, output)
docker run --rm \\
    -v $(pwd)/weights:/app/weights \\
    -v $(pwd)/input:/app/input \\
    -v $(pwd)/output:/app/output \\
    {repo_name.lower()}-inference \\
    python inference_generated.py \\
    --input /app/input/test.png \\
    --output /app/output/result.png \\
    --weights /app/weights/model.pth \\
    --device cpu

# Interactive shell for debugging
docker run -it --rm \\
    -v $(pwd)/weights:/app/weights \\
    -v $(pwd)/input:/app/input \\
    -v $(pwd)/output:/app/output \\
    {repo_name.lower()}-inference \\
    /bin/bash
'''


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Quick setup for ML inference repositories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Full setup:
    python quick_setup.py /path/to/repo

  Fix imports only:
    python quick_setup.py /path/to/repo --fix-imports

  Generate Docker only:
    python quick_setup.py /path/to/repo --docker-only

  Dry run (no changes):
    python quick_setup.py /path/to/repo --dry-run
        '''
    )
    parser.add_argument('repo_path', help='Path to the repository')
    parser.add_argument('--fix-imports', action='store_true', help='Fix optional imports only')
    parser.add_argument('--docker-only', action='store_true', help='Generate Dockerfile only')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--output-json', action='store_true', help='Output analysis as JSON')

    args = parser.parse_args()
    repo_path = Path(args.repo_path).resolve()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    print(f"=" * 60)
    print(f"Quick Setup: {repo_path.name}")
    print(f"=" * 60)

    # Step 1: Analyze
    print("\n[1/4] Analyzing repository...")
    analysis = analyze_repo_quick(repo_path)

    if args.output_json:
        print(json.dumps(analysis, indent=2))
        return

    print(f"  Framework: {analysis['framework'] or 'standard'}")
    print(f"  Model files: {len(analysis['model_files'])}")
    print(f"  Inference scripts: {len(analysis['inference_scripts'])}")
    if analysis['inference_scripts']:
        for script in analysis['inference_scripts'][:3]:
            print(f"    - {script}")
    print(f"  Multi-network: {analysis['multi_network']}")
    print(f"  Diffusion model: {analysis['diffusion_model']}")
    print(f"  Install method: {analysis['install_method']}")

    # Warnings
    if analysis['cuda_hardcoded']:
        print("\n  ⚠️  WARNING: CUDA hardcoded - may need patches for CPU")
    if analysis['diffusion_model']:
        print("  ⚠️  WARNING: Diffusion model detected - use existing inference script, not generated one")

    if args.docker_only:
        # Just generate Dockerfile
        dockerfile = generate_dockerfile(repo_path, analysis)
        if not args.dry_run:
            (repo_path / 'Dockerfile').write_text(dockerfile)
            print(f"\n✓ Generated Dockerfile")
        else:
            print(f"\n[DRY RUN] Would generate Dockerfile")
        return

    # Step 2: Fix imports
    print("\n[2/4] Fixing optional imports...")
    fixes = find_and_fix_imports(repo_path, dry_run=args.dry_run)
    if fixes:
        for fix in fixes:
            print(f"  Fixed {fix['package']} in {fix['file']}")
    else:
        print("  No fixes needed")

    if args.fix_imports:
        return

    # Step 3: Generate inference script
    print("\n[3/4] Generating inference script...")
    inference_script = generate_inference_script(repo_path, analysis)
    if not args.dry_run:
        (repo_path / 'inference_generated.py').write_text(inference_script)
        print(f"  Created inference_generated.py")
    else:
        print(f"  [DRY RUN] Would create inference_generated.py")

    # Step 4: Generate Dockerfile
    print("\n[4/4] Generating Dockerfile...")
    dockerfile = generate_dockerfile(repo_path, analysis)
    if not args.dry_run:
        (repo_path / 'Dockerfile').write_text(dockerfile)
        print(f"  Created Dockerfile")

        # Also save docker commands
        docker_cmds = generate_docker_commands(analysis['name'])
        (repo_path / 'DOCKER_COMMANDS.md').write_text(docker_cmds)
        print(f"  Created DOCKER_COMMANDS.md")
    else:
        print(f"  [DRY RUN] Would create Dockerfile and DOCKER_COMMANDS.md")

    print("\n" + "=" * 60)
    print("Setup complete! Next steps:")
    print("=" * 60)
    print(f"""
1. Download pretrained weights to: {analysis['weight_locations'][0] if analysis['weight_locations'] else 'weights/'}

2. Build Docker image:
   cd {repo_path}
   docker build -t {analysis['name'].lower()}-inference .

3. Run inference:
   docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \\
       {analysis['name'].lower()}-inference python inference_generated.py \\
       --input /app/input/test.png --output /app/output/result.png \\
       --weights /app/weights/model.pth --device cpu
""")


if __name__ == '__main__':
    main()
