# Reproducible Unit (RU) Specification

A Reproducible Unit is a standardized package for ML inference that ensures:
- **Reproducibility**: Same input → same output across environments
- **Portability**: Runs anywhere with Docker
- **Simplicity**: Single command to run inference

## Directory Structure

```
<name>_RU/
├── adapter.py          # Unified inference interface (REQUIRED)
├── metadata.json       # RU metadata and settings (REQUIRED)
├── Dockerfile          # Container definition (REQUIRED)
├── requirements.txt    # Python dependencies (REQUIRED)
├── download_weights.sh # Weight download script (REQUIRED)
├── weights/            # Model weights directory
│   └── .gitkeep        # (weights downloaded separately)
├── <model_source>/     # Minimal model source files
├── test_input/         # Sample test images
├── test_output/        # Expected outputs
└── README.md           # Usage instructions
```

**Note**: `metadata.json` replaces `config.yaml` for unified configuration across all task types.
See [METADATA_SCHEMA.md](METADATA_SCHEMA.md) for full schema documentation.

## adapter.py Interface

```python
class Adapter:
    """Unified inference interface for Reproducible Units."""

    def __init__(self, weights_path: str = None, device: str = 'cpu'):
        """Initialize model with weights."""
        pass

    def preprocess(self, image_path: str) -> torch.Tensor:
        """Load and preprocess input image."""
        pass

    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run model inference."""
        pass

    def postprocess(self, output_tensor: torch.Tensor) -> Image:
        """Convert output tensor to PIL Image."""
        pass

    def run(self, input_path: str, output_path: str) -> None:
        """Full inference pipeline."""
        tensor = self.preprocess(input_path)
        output = self.inference(tensor)
        result = self.postprocess(output)
        result.save(output_path)

# CLI interface
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--weights', '-w', default='weights/model.pth')
    parser.add_argument('--device', '-d', default='cpu')
    args = parser.parse_args()

    adapter = Adapter(args.weights, args.device)
    adapter.run(args.input, args.output)
```

## metadata.json Format

See [METADATA_SCHEMA.md](METADATA_SCHEMA.md) for full schema. Minimal example:

```json
{
  "name": "ModelName",
  "version": "1.0.0",
  "task": {
    "type": "super_resolution",
    "params": { "scale": 4 }
  },
  "model": {
    "type": "single",
    "architecture": "SRNet"
  },
  "weights": {
    "default": "model.pth",
    "options": {
      "model.pth": { "url": "https://..." }
    }
  },
  "input": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1]
  },
  "output": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1],
    "scale_factor": 4
  },
  "inference": {
    "device": "cpu"
  },
  "source": {
    "repo": "https://github.com/user/repo"
  }
}
```

## Dockerfile Template

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy adapter and config
COPY adapter.py config.yaml ./
COPY weights/ ./weights/

# Default command
ENTRYPOINT ["python", "adapter.py"]
CMD ["--help"]
```

## Usage

### Local
```bash
python adapter.py -i input.png -o output.png -w weights/model.pth
```

### Docker
```bash
# Build
docker build -t model-ru .

# Run
docker run --rm \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    model-ru -i /input/image.png -o /output/result.png
```

### Validation
```bash
# Test with sample input
python adapter.py -i test_input/sample.png -o test_output/result.png
# Compare with expected output
```

## RU Types

### Type 1: Single Model (SCUNet)
- One model file, straightforward inference
- Structure: `adapter.py` + `models/network.py`

### Type 2: Multi-Network (DASR)
- Multiple models that work together (predictor + generator)
- Structure: `adapter.py` + `basicsr/archs/` (with registry stub)

### Type 3: Diffusion Model (ResShift)
- Sampler + UNet + Autoencoder
- Multiple inference steps
- Config-driven
- Structure: `adapter.py` + `src/` (sampler, models, configs, utils)

---

## Self-Containment Requirements

An RU must be **fully self-contained** - all imports must resolve within the RU directory.

### Common Issues and Fixes

#### External Framework Dependencies (BasicSR)
Create stub files for framework utilities:

```python
# basicsr/utils/registry.py
class Registry:
    def __init__(self, name):
        self._name = name

    def register(self, obj=None):
        if obj is None:
            return lambda func_or_cls: func_or_cls
        return obj

ARCH_REGISTRY = Registry('arch')
```

```python
# basicsr/utils/__init__.py
from .registry import ARCH_REGISTRY

def get_root_logger():
    import logging
    return logging.getLogger('basicsr')

def scandir(dir_path):
    import os
    for entry in os.scandir(dir_path):
        if entry.is_file():
            yield entry.name
```

#### Training-Only Imports
Use lazy imports for modules only needed for training:

```python
# At module level (not inside functions)
create_dataset = None
def _get_create_dataset():
    global create_dataset
    if create_dataset is None:
        from datapipe.datasets import create_dataset as _cd
        create_dataset = _cd
    return create_dataset

# When needed
dataset = _get_create_dataset()(config)
```

#### CUDA Hardcoding
Patch for CPU support:

```python
# Before
self.device = 'cuda'
tensor.cuda()
torch.load(path, map_location=f"cuda:{rank}")

# After
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor.to(self.device)
torch.load(path, map_location=self.device)
```

#### Profiling Imports
Make optional with try/except:

```python
try:
    from thop import profile
except ImportError:
    profile = None  # Only used for profiling
```

---

## Testing

### test_all.py

Place in RU parent directory to test all RUs:

```python
#!/usr/bin/env python3
"""Test all RUs - verifies model creation, inference shape, output validity."""

import sys
from pathlib import Path
import torch

def test_ru(ru_path: Path):
    # 1. Test model creation (without weights)
    # 2. Test inference shape with random input
    # 3. Check output validity (no Inf, NaN ok with random weights)
    pass

def main():
    ru_dir = Path(__file__).parent
    for ru_path in ru_dir.glob('*_RU'):
        if (ru_path / 'adapter.py').exists():
            test_ru(ru_path)
```

### Test Criteria

| Check | Pass Condition |
|-------|----------------|
| Model creation | No import errors, model instantiates |
| Inference shape | Output matches expected size (same for denoising, scaled for SR) |
| Output validity | No Inf values (NaN acceptable with random weights) |

### Running Tests

```bash
cd /path/to/RU_parent
python test_all.py

# Expected output:
# SCUNet_RU: PASS
#   Model creation: OK
#   Inference shape: OK (64x64 -> 64x64)
#   Output validity: OK
```

---

## Best Practices

1. **Minimize source files**: Copy only what's needed for inference
2. **Stub external deps**: Don't copy entire frameworks, create minimal stubs
3. **Lazy load training deps**: Use lazy imports for non-inference modules
4. **Test without weights**: Model creation should work without pretrained weights
5. **Document weight URLs**: Include download script and URLs in config.yaml
