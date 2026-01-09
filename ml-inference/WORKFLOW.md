# GitHub Repo → Reproducible Unit (RU) Workflow

This guide provides the step-by-step workflow for converting a GitHub ML inference repository into a self-contained Reproducible Unit.

---

## Overview

```
GitHub URL → Clone → Analyze → Test Locally → Create RU → Test RU → Update Skills → Done
```

**Output**: `<name>_RU/` folder that can run inference independently, plus skill updates.

---

## Phase 1: Clone & Analyze

### Step 1.1: Clone Repository

```bash
git clone <repo_url> <repo_name>
cd <repo_name>
```

### Step 1.2: Identify Model Type

Read README and explore structure to determine:

| Type | Indicators | Example |
|------|------------|---------|
| Single Model | One network class, one weight file | SCUNet |
| Multi-Network | Multiple networks working together | DASR (predictor + generator) |
| Diffusion | Sampler, DDPM/DDIM, noise scheduling | ResShift |

### Step 1.3: Find Key Files

```bash
# Model architecture
find . -type f -name "*arch*.py" -o -name "*network*.py" -o -name "*model*.py" | head -20

# Inference scripts
ls -la *.py | grep -E "(inference|test|demo|predict)"

# Weight download info
grep -r "drive.google.com\|dropbox\|releases/download" README.md
```

### Step 1.4: Check Framework

| Framework | Indicators | Setup |
|-----------|------------|-------|
| BasicSR | `basicsr/` dir, `ARCH_REGISTRY` | `pip install -e .` |
| Custom | Direct model classes | Add to PYTHONPATH |
| Diffusion | `sampler.py`, configs/ | May need patching for CPU |

---

## Phase 2: Test Locally

### Step 2.1: Set Up Environment

```bash
# Prefer conda for proxy issues
source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>
conda install -c conda-forge <packages> -y

# Or pip
pip install -r requirements.txt
```

### Step 2.2: Download Weights

Check README for download links. Common sources:
- GitHub releases
- Google Drive
- Dropbox

**If download fails**: Try once, then ask user to download manually.

### Step 2.3: Run Existing Inference

```bash
python inference.py --input test.png --output result.png
# or
python main_test_*.py
```

### Step 2.4: Fix Common Issues

**CUDA hardcoding**:
```python
# Replace .cuda() with .to(device)
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Profiling imports**:
```python
try:
    from thop import profile
except ImportError:
    profile = None
```

---

## Phase 3: Create RU Structure

### Step 3.1: Create Directory

```bash
mkdir -p <name>_RU/{weights,test_input,test_output}
```

### Step 3.2: Copy Minimal Source

Copy ONLY inference-required files:

| Copy | Don't Copy |
|------|------------|
| Model architecture | Training scripts |
| Required utilities | Datasets |
| Config files | Evaluation metrics |

### Step 3.3: Make Imports Self-Contained

**Registry stub** (for BasicSR):
```python
# basicsr/utils/registry.py
class Registry:
    def register(self, obj=None):
        if obj is None:
            return lambda x: x
        return obj
ARCH_REGISTRY = Registry('arch')
```

**Lazy imports** (for training-only deps):
```python
create_dataset = None
def _get_create_dataset():
    global create_dataset
    if create_dataset is None:
        from datapipe.datasets import create_dataset as _cd
        create_dataset = _cd
    return create_dataset
```

---

## Phase 4: Create adapter.py

Standard interface:

```python
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from models.network import Model  # Adjust import


class Adapter:
    def __init__(self, weights_path: str = None, device: str = 'cpu'):
        self.device = device
        self.model = Model()  # Adjust initialization

        weights_path = weights_path or 'weights/model.pth'
        state = torch.load(weights_path, map_location=device)
        # Handle different checkpoint formats
        if 'params' in state:
            state = state['params']
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()

    def preprocess(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

    def postprocess(self, output_tensor: torch.Tensor) -> Image.Image:
        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(output)

    def run(self, input_path: str, output_path: str) -> None:
        tensor = self.preprocess(input_path)
        output = self.inference(tensor)
        result = self.postprocess(output)
        result.save(output_path)
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--weights', '-w', default='weights/model.pth')
    parser.add_argument('--device', '-d', default='cpu')
    args = parser.parse_args()

    adapter = Adapter(args.weights, args.device)
    adapter.run(args.input, args.output)


if __name__ == '__main__':
    main()
```

---

## Phase 5: Create Supporting Files

### config.yaml

```yaml
name: <name>
task: <denoising|super_resolution|restoration>
scale: <1|2|4>  # for SR tasks

weights:
  default: weights/model.pth
  url: <download_url>

input:
  channels: 3
  format: RGB

output:
  channels: 3
  format: RGB
```

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /input /output

ENTRYPOINT ["python", "adapter.py"]
CMD ["--help"]
```

### requirements.txt

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
pillow>=8.0.0
# Add model-specific deps
```

### download_weights.sh

```bash
#!/bin/bash
mkdir -p weights
wget -q <url> -O weights/model.pth
echo "Downloaded weights to weights/model.pth"
```

---

## Phase 6: Test RU

### Step 6.1: Run test_all.py

Place `test_all.py` in the RU parent directory:

```bash
cd /path/to/RU_parent
python test_all.py
```

### Step 6.2: Verify Output

All checks should pass:
```
SCUNet_RU: PASS
  Model creation: OK
  Inference shape: OK (64x64 -> 64x64)
  Output validity: OK
```

### Step 6.3: Test with Real Weights

```bash
cd <name>_RU
./download_weights.sh
python adapter.py -i test_input/sample.png -o test_output/result.png
```

---

## Phase 7: Update Skills with Learnings

After successful RU creation, capture what was learned.

### Step 7.1: Identify New Patterns

Review what fixes were needed:
- New import stub patterns?
- New preprocessing/postprocessing?
- New model architecture type?
- New troubleshooting steps?

### Step 7.2: Update Skill Files

| Learning Type | Update Location |
|---------------|-----------------|
| New fix pattern | SKILL.md → "Common Fixes" section |
| New model type | RU_SPEC.md → "RU Types" section |
| New repo pattern | COMMON_PATTERNS.md |
| Working script | `templates/` directory |
| Troubleshooting | SKILL.md → "Troubleshooting" section |

### Step 7.3: Add to Tested Repositories

Update the "Tested Repositories" table in SKILL.md:

```markdown
| Repo | Type | Task | RU Status |
|------|------|------|-----------|
| [NewRepo](url) | Type | Task | Verified |
```

### Example Learnings

From SCUNet:
- Profiling imports (thop) can be made optional with try/except

From DASR:
- BasicSR registry needs stub file
- Multi-network models need multiple weight files

From ResShift:
- Diffusion models need lazy imports for training-only deps
- CUDA hardcoding common in samplers

---

## Checklist

Before finalizing RU:

- [ ] All imports resolve within RU (no external deps)
- [ ] Model creation works without weights
- [ ] Inference produces correct output shape
- [ ] adapter.py runs with `--help`
- [ ] Dockerfile builds successfully
- [ ] download_weights.sh has correct URL
- [ ] config.yaml has accurate metadata
- [ ] test_all.py passes
- [ ] Skills updated with new learnings
- [ ] Tested Repositories table updated

---

## Quick Reference: RU Types

### Single Model (SCUNet-style)
```
<name>_RU/
├── adapter.py
├── models/
│   └── network.py
└── weights/
    └── model.pth
```

### Multi-Network (DASR-style)
```
<name>_RU/
├── adapter.py
├── basicsr/
│   ├── archs/
│   │   ├── generator.py
│   │   └── predictor.py
│   └── utils/
│       └── registry.py  # stub
└── weights/
    ├── net_g.pth
    └── net_p.pth
```

### Diffusion (ResShift-style)
```
<name>_RU/
├── adapter.py
├── src/
│   ├── sampler.py  # patched for CPU
│   ├── configs/
│   ├── models/
│   └── utils/
└── weights/
    ├── diffusion.pth
    └── autoencoder.pth
```
