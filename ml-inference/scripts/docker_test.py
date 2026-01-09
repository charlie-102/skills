#!/usr/bin/env python3
"""
Docker Test Runner - Build and test ML inference in containers.

This script handles the full Docker workflow:
1. Build container from repo
2. Download weights (if URL provided)
3. Run inference test
4. Report results

Usage:
    python docker_test.py /path/to/repo --test-image test.png
    python docker_test.py /path/to/repo --weights-url https://example.com/model.pth
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
import hashlib


def run_cmd(cmd: list, cwd: str = None, capture: bool = False, timeout: int = 600) -> tuple:
    """Run command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout if capture else ""
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def get_image_name(repo_path: Path) -> str:
    """Generate Docker image name from repo path."""
    return f"ml-inference-{repo_path.name.lower()}"


def build_image(repo_path: Path, no_cache: bool = False) -> bool:
    """Build Docker image."""
    image_name = get_image_name(repo_path)

    # Check if Dockerfile exists
    dockerfile = repo_path / 'Dockerfile'
    if not dockerfile.exists():
        print("  Dockerfile not found. Generating...")
        # Run quick_setup to generate Dockerfile
        script_dir = Path(__file__).parent
        quick_setup = script_dir / 'quick_setup.py'
        if quick_setup.exists():
            success, _ = run_cmd(
                [sys.executable, str(quick_setup), str(repo_path), '--docker-only'],
                capture=True
            )
            if not success:
                print("  Failed to generate Dockerfile")
                return False

    cmd = ['docker', 'build', '-t', image_name, '.']
    if no_cache:
        cmd.insert(2, '--no-cache')

    print(f"  Building image: {image_name}")
    success, _ = run_cmd(cmd, cwd=str(repo_path), timeout=1800)
    return success


def download_weights(url: str, output_dir: Path) -> Optional[Path]:
    """Download weights file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename from URL
    filename = url.split('/')[-1].split('?')[0]
    if not filename.endswith(('.pth', '.pt', '.ckpt')):
        filename = 'model.pth'

    output_path = output_dir / filename

    if output_path.exists():
        print(f"  Weights already exist: {output_path}")
        return output_path

    print(f"  Downloading: {url}")

    # Try wget first, then curl
    success, _ = run_cmd(['wget', '-q', '-O', str(output_path), url], timeout=300)
    if not success:
        success, _ = run_cmd(['curl', '-L', '-o', str(output_path), url], timeout=300)

    if success and output_path.exists():
        return output_path
    return None


def run_inference_test(
    repo_path: Path,
    test_image: Path,
    weights_path: Path,
    output_dir: Path
) -> bool:
    """Run inference test in Docker container."""
    image_name = get_image_name(repo_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the inference script
    inference_script = 'inference_generated.py'
    if not (repo_path / inference_script).exists():
        # Look for other inference scripts
        for pattern in ['inference*.py', 'test*.py', 'main_test*.py']:
            matches = list(repo_path.glob(pattern))
            if matches:
                inference_script = matches[0].name
                break

    output_file = output_dir / f"result_{test_image.stem}.png"

    # Build docker run command
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{test_image.parent}:/app/input:ro',
        '-v', f'{weights_path.parent}:/app/weights:ro',
        '-v', f'{output_dir}:/app/output',
        image_name,
        'python', inference_script,
        '--input', f'/app/input/{test_image.name}',
        '--output', f'/app/output/{output_file.name}',
        '--weights', f'/app/weights/{weights_path.name}',
        '--device', 'cpu'
    ]

    print(f"  Running inference...")
    success, output = run_cmd(cmd, capture=True, timeout=300)

    if success and output_file.exists():
        print(f"  ✓ Output saved to: {output_file}")
        return True
    else:
        print(f"  ✗ Inference failed")
        if output:
            print(f"  Error: {output[:500]}")
        return False


def create_test_image(output_path: Path, size: tuple = (256, 256)) -> Path:
    """Create a simple test image."""
    try:
        from PIL import Image
        import numpy as np

        # Create gradient test image
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for i in range(size[1]):
            for j in range(size[0]):
                arr[i, j] = [
                    int(255 * i / size[1]),
                    int(255 * j / size[0]),
                    128
                ]

        img = Image.fromarray(arr)
        img.save(output_path)
        return output_path
    except ImportError:
        print("  PIL not available, cannot create test image")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Docker-based ML inference testing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('repo_path', help='Path to the repository')
    parser.add_argument('--test-image', help='Path to test image')
    parser.add_argument('--weights', help='Path to weights file')
    parser.add_argument('--weights-url', help='URL to download weights from')
    parser.add_argument('--output-dir', default='./docker_test_output', help='Output directory')
    parser.add_argument('--no-cache', action='store_true', help='Build without Docker cache')
    parser.add_argument('--skip-build', action='store_true', help='Skip Docker build step')

    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not repo_path.exists():
        print(f"Error: Repository not found: {repo_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"Docker Test: {repo_path.name}")
    print("=" * 60)

    # Step 1: Build Docker image
    if not args.skip_build:
        print("\n[1/3] Building Docker image...")
        if not build_image(repo_path, args.no_cache):
            print("  ✗ Build failed")
            sys.exit(1)
        print("  ✓ Build successful")

    # Step 2: Get weights
    print("\n[2/3] Preparing weights...")
    weights_dir = output_dir / 'weights'
    weights_path = None

    if args.weights:
        weights_path = Path(args.weights).resolve()
        if not weights_path.exists():
            print(f"  ✗ Weights not found: {weights_path}")
            sys.exit(1)
    elif args.weights_url:
        weights_path = download_weights(args.weights_url, weights_dir)
        if not weights_path:
            print("  ✗ Failed to download weights")
            sys.exit(1)
    else:
        # Look for existing weights in repo
        for pattern in ['*.pth', '*.pt', 'model_zoo/*.pth', 'pretrained/*.pth', 'weights/*.pth']:
            matches = list(repo_path.glob(pattern))
            if matches:
                weights_path = matches[0]
                print(f"  Found weights: {weights_path}")
                break

        if not weights_path:
            print("  ✗ No weights found. Use --weights or --weights-url")
            sys.exit(1)

    print(f"  ✓ Using weights: {weights_path}")

    # Step 3: Run inference
    print("\n[3/3] Running inference test...")

    # Get or create test image
    if args.test_image:
        test_image = Path(args.test_image).resolve()
        if not test_image.exists():
            print(f"  ✗ Test image not found: {test_image}")
            sys.exit(1)
    else:
        test_image = output_dir / 'test_input.png'
        if not test_image.exists():
            print("  Creating test image...")
            test_image = create_test_image(test_image)
            if not test_image:
                sys.exit(1)

    success = run_inference_test(repo_path, test_image, weights_path, output_dir)

    print("\n" + "=" * 60)
    if success:
        print("✓ TEST PASSED")
    else:
        print("✗ TEST FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
