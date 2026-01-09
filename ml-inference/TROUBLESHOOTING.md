# ML Inference Troubleshooting Guide

Common issues and solutions when converting ML repositories to Reproducible Units.

---

## Import Errors

### Issue: `ModuleNotFoundError: No module named 'basicsr'`

**Cause**: Repository uses BasicSR framework but environment not properly activated.

**Solutions**:
```bash
# Option 1: Source shell config to activate conda environment
source ~/.zshrc  # or ~/.bashrc
conda activate <env_name>

# Option 2: Install BasicSR in current environment
pip install basicsr

# Option 3: Install repo as package
cd /path/to/repo && pip install -e .

# Option 4: Create stub files (for RU self-containment)
# See RU_SPEC.md for registry stub template
```

### Issue: `ImportError: cannot import name 'ARCH_REGISTRY'`

**Cause**: Missing registry stub in RU.

**Solution**: Create `basicsr/utils/registry.py`:
```python
class Registry:
    def __init__(self, name):
        self._name = name

    def register(self, obj=None):
        if obj is None:
            return lambda func_or_cls: func_or_cls
        return obj

ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
```

### Issue: `ModuleNotFoundError: No module named 'thop'`

**Cause**: Profiling dependency not installed (usually not needed for inference).

**Solution**: Wrap with try/except:
```python
try:
    from thop import profile
except ImportError:
    profile = None
```

### Issue: `ModuleNotFoundError: No module named 'datapipe'` or training-only deps

**Cause**: Training-only dependencies imported at module level.

**Solution**: Use lazy imports:
```python
# Bad - imported at module load
from datapipe.datasets import create_dataset

# Good - lazy import
_create_dataset = None
def get_create_dataset():
    global _create_dataset
    if _create_dataset is None:
        from datapipe.datasets import create_dataset
        _create_dataset = create_dataset
    return _create_dataset
```

---

## Weight Loading Errors

### Issue: `RuntimeError: Error(s) in loading state_dict`

**Cause**: Mismatched keys between model and checkpoint.

**Solutions**:

1. **Check for nested checkpoint structure**:
```python
checkpoint = torch.load(weights_path, map_location=device)

# Try common nested keys
if isinstance(checkpoint, dict):
    for key in ['params', 'state_dict', 'model_state_dict', 'model', 'net', 'G']:
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
```

2. **Remove DataParallel prefix**:
```python
if list(checkpoint.keys())[0].startswith('module.'):
    checkpoint = {k[7:]: v for k, v in checkpoint.items()}
```

3. **Try non-strict loading** (debugging only):
```python
missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print(f"Missing: {missing}")
print(f"Unexpected: {unexpected}")
```

### Issue: `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Model and input on different devices.

**Solution**:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
input_tensor = input_tensor.to(device)
```

### Issue: `torch.cuda.OutOfMemoryError`

**Cause**: Model or input too large for GPU memory.

**Solutions**:

1. **Use CPU**:
```python
device = 'cpu'
```

2. **Use tile processing**:
```python
# Process image in overlapping tiles
for tile in tiles:
    output_tile = model(tile)
    # Blend tiles together
```

3. **Use half precision**:
```python
model = model.half()
input_tensor = input_tensor.half()
```

4. **Reduce batch size**: Use batch_size=1

---

## CUDA/Device Errors

### Issue: `.cuda()` fails on CPU-only machine

**Cause**: Hardcoded CUDA calls in repository.

**Solution**: Patch to use device variable:
```python
# Before
self.model = Model().cuda()
tensor = tensor.cuda()

# After
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
self.model = Model().to(self.device)
tensor = tensor.to(self.device)
```

### Issue: `RuntimeError: CUDA error: device-side assert triggered`

**Cause**: Usually input dimension or value range issues.

**Solutions**:
1. Check input is in expected range (e.g., [0,1] vs [-1,1])
2. Check input dimensions match model expectations
3. Run on CPU to get more detailed error message

---

## Shape/Dimension Errors

### Issue: `RuntimeError: Given groups=1, weight of size [64, 3, 3, 3], expected input[1, 1, H, W]`

**Cause**: Model expects 3 channels, got 1 (grayscale).

**Solution**:
```python
# Convert grayscale to RGB
img = Image.open(path).convert('RGB')
```

### Issue: `RuntimeError: Sizes of tensors must match`

**Cause**: Input size not divisible by required multiple (usually 8, 16, 32, or 64).

**Solution**: Pad input to multiple:
```python
def pad_to_multiple(tensor, multiple=64):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

# Remember to crop output back to original size
```

### Issue: Output is wrong size for super-resolution

**Cause**: Scale factor not applied correctly.

**Check**:
```python
# For 4x SR: output should be 4x input size
expected_h = input_h * scale
expected_w = input_w * scale
assert output.shape[2] == expected_h
```

---

## Color/Image Quality Issues

### Issue: Output image has wrong colors

**Cause**: Color space mismatch (RGB vs BGR).

**Solution**:
```python
# If model trained with OpenCV (BGR), convert:
# Input: RGB -> BGR
img_bgr = img_rgb[:, :, ::-1]

# Output: BGR -> RGB
output_rgb = output_bgr[:, :, ::-1]
```

### Issue: Output is too dark or bright

**Cause**: Wrong normalization range.

**Check preprocessing**:
```python
# Most common: [0, 1]
img = img / 255.0

# Some models: [-1, 1]
img = img / 255.0 * 2 - 1

# Match postprocessing!
```

### Issue: Output has artifacts at tile boundaries

**Cause**: Insufficient tile overlap or no blending.

**Solution**: Increase overlap and use weighted blending:
```python
TILE_OVERLAP = 64  # Increase from 32

# Use cosine blending for smoother transitions
def create_blend_mask(size, overlap):
    ramp = np.linspace(0, np.pi, overlap)
    weights = (1 - np.cos(ramp)) / 2
    # Apply to edges...
```

---

## Diffusion Model Issues

### Issue: Diffusion inference extremely slow on CPU

**Cause**: Diffusion models run many denoising steps.

**Solutions**:
1. Reduce number of steps (may reduce quality):
```python
sampler = Sampler(num_steps=15)  # Instead of 1000
```

2. Use DDIM instead of DDPM sampler (faster)
3. Use smaller tile size
4. Accept that CPU inference will be slow

### Issue: `KeyError: 'ema_model'` or similar

**Cause**: Diffusion checkpoint has EMA (Exponential Moving Average) weights.

**Solution**:
```python
checkpoint = torch.load(weights_path)
if 'ema_model' in checkpoint:
    state_dict = checkpoint['ema_model']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
```

### Issue: Config file not found

**Cause**: Diffusion models typically need YAML config files.

**Solution**: Ensure config file is copied to RU:
```
RU/
├── src/
│   └── configs/
│       └── realsr_swinunet_realesrgan256.yaml  # Copy this!
```

---

## Docker Issues

### Issue: `ImportError: libGL.so.1: cannot open shared object file`

**Cause**: OpenCV needs OpenGL libraries.

**Solution** in Dockerfile:
```dockerfile
RUN apt-get update && apt-get install -y \
    libgl1 \          # Correct package name
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# NOTE: Use libgl1, NOT libgl1-mesa-glx (old package name)
```

### Issue: Docker build fails with pip errors

**Cause**: Dependencies conflict or proxy issues.

**Solutions**:
1. Pin versions in requirements.txt
2. Use conda base image for tricky dependencies
3. Split pip install into multiple RUN commands to identify failing package

### Issue: Model works locally but fails in Docker

**Cause**: Usually path issues or missing files.

**Checklist**:
- [ ] All source files copied to Docker image
- [ ] Weight files mounted or downloaded
- [ ] PYTHONPATH set correctly
- [ ] Working directory correct

---

## Multi-Network Architecture Issues

### Issue: `AttributeError: 'tuple' object has no attribute 'to'`

**Cause**: Model returns multiple outputs, code expects single tensor.

**Solution**:
```python
output = model(input)
if isinstance(output, tuple):
    output = output[0]  # Usually first element is the main output
```

### Issue: Predictor + Generator networks not synchronized

**Cause**: Wrong weight files loaded for each network.

**Solution**: Be explicit about which weights go where:
```python
# DASR-style
net_g.load_state_dict(torch.load('net_g.pth'))
net_p.load_state_dict(torch.load('net_p.pth'))

# Check metadata.json for correct mapping
```

---

## Quick Diagnostic Commands

```bash
# Check PyTorch/CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test model import
python -c "from models.network import Model; print('Import OK')"

# Check checkpoint keys
python -c "import torch; ckpt = torch.load('model.pth', map_location='cpu'); print(ckpt.keys() if isinstance(ckpt, dict) else type(ckpt))"

# Check model parameter count
python -c "from models.network import Model; m = Model(); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"

# Quick inference test
python -c "
import torch
from models.network import Model
m = Model().eval()
x = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    y = m(x)
print(f'Input: {x.shape} -> Output: {y.shape}')
"
```

---

## When All Else Fails

1. **Check the original repo's inference script** - It often has the exact preprocessing/postprocessing needed

2. **Look for issues/discussions** in the GitHub repo - Others may have encountered the same problem

3. **Test with the original repo first** - Ensure the original code works before creating RU

4. **Check model architecture parameters** - Wrong constructor args cause subtle bugs

5. **Compare intermediate outputs** - Add print statements to trace where things diverge

6. **Read the paper** - Implementation details often in supplementary material
