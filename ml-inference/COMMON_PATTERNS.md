# Common Patterns in ML Image Processing Repositories

This document catalogs common patterns found in deep learning image processing repositories to help quickly understand and adapt new codebases.

## Repository Structures

### Pattern A: Simple Flat Structure
```
repo/
├── model.py          # Model definition
├── test.py           # Inference script
├── train.py          # Training script
├── utils.py          # Utilities
├── requirements.txt
└── README.md
```
**Examples**: Small research repos, demo projects

### Pattern B: Modular Structure
```
repo/
├── models/
│   ├── __init__.py
│   ├── network.py
│   └── layers.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── img_utils.py
├── configs/
│   └── config.yaml
├── scripts/
│   ├── test.py
│   └── train.py
├── model_zoo/
└── README.md
```
**Examples**: SCUNet, SwinIR

### Pattern C: MMEditing/OpenMMLab Style
```
repo/
├── mmedit/
│   ├── models/
│   ├── datasets/
│   └── apis/
├── configs/
│   ├── restorers/
│   └── synthesizers/
├── tools/
│   ├── test.py
│   └── train.py
├── demo/
└── README.md
```
**Examples**: MMEditing, BasicSR-based repos

### Pattern D: Research Paper Structure
```
repo/
├── src/
│   ├── model/
│   ├── data/
│   └── trainer/
├── experiments/
│   └── exp_name/
├── pretrained/
├── main.py
└── README.md
```
**Examples**: Academic research code

### Pattern E: BasicSR Framework Structure
```
repo/
├── basicsr/
│   ├── archs/              # Network architectures (registry-based)
│   │   ├── arch_util.py
│   │   └── my_arch.py
│   ├── models/             # Model wrappers (training logic)
│   │   └── sr_model.py
│   ├── data/               # Dataset definitions
│   └── utils/
├── options/                # YAML config files
│   ├── train/
│   └── test/
├── experiments/            # Checkpoints stored here
│   └── pretrained_models/
├── inference/              # Simple inference scripts
├── setup.py                # Installs as package
└── README.md
```
**Examples**: DASR, Real-ESRGAN, BasicSR

**Key characteristics**:
- Registry pattern: `@ARCH_REGISTRY.register_class`
- Config-driven testing: YAML files define model, weights, datasets
- Multi-network support: Separate net_g (generator) and net_p (predictor)
- Installed as package via `pip install -e .`

### Pattern F: Diffusion Model Structure
```
repo/
├── configs/                # YAML configs for different tasks
│   ├── realsr_*.yaml
│   └── deblur_*.yaml
├── models/
│   ├── unet.py            # UNet backbone
│   └── diffusion.py       # Diffusion process
├── sampler.py             # Diffusion sampler (DDPM/DDIM)
├── inference_<name>.py    # Main inference script
├── weights/               # Pretrained models
├── app.py                 # Gradio demo
└── README.md
```
**Examples**: ResShift, StableSR, DiffBIR

**Key characteristics**:
- Sampler class with `build_model()`, noise scheduling
- Multiple weight files (main model + autoencoder/VQGAN)
- Often hardcodes `.cuda()` - needs CPU fallback patches
- Config-driven with OmegaConf/YAML
- Use existing inference script, don't generate new one

**Common CPU fallback patches needed**:
```python
# In sampler.py:
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Replace: .cuda() → .to(self.device)
# Replace: map_location=f"cuda:{rank}" → map_location=self.device
```

## Inference Script Patterns

### Pattern 1: Argparse-based CLI
```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.model)
    result = inference(model, args.input)
    save_result(result, args.output)

if __name__ == '__main__':
    main()
```

### Pattern 2: Config-based
```python
import yaml

def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    model = build_model(config['model'])
    model.load_state_dict(torch.load(config['weights']))

    for img_path in config['test_images']:
        result = inference(model, img_path)
        save_result(result, config['output_dir'])
```

### Pattern 3: Class-based Inference
```python
class Inferencer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, path):
        model = Network()
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        model.eval()
        return model

    def __call__(self, image):
        with torch.no_grad():
            return self.model(image)
```

### Pattern 4: Multi-Network Inference (DASR-style)
```python
class MultiNetworkInference:
    """For architectures with multiple networks (predictor + generator)."""

    def __init__(self, weights_g: str, weights_p: str, device: str = 'cuda'):
        self.device = device

        # Load generator network
        self.net_g = GeneratorNetwork()
        self.net_g.load_state_dict(torch.load(weights_g, map_location=device))
        self.net_g.to(device).eval()

        # Load predictor/auxiliary network
        self.net_p = PredictorNetwork()
        self.net_p.load_state_dict(torch.load(weights_p, map_location=device))
        self.net_p.to(device).eval()

    def __call__(self, image):
        with torch.no_grad():
            # Predictor provides parameters/features for generator
            params = self.net_p(image)
            # Generator uses predictor output
            output = self.net_g(image, params)
        return output
```
**Examples**: DASR (degradation predictor + super-resolution generator)

## Model Loading Patterns

### Direct State Dict
```python
model = Network()
model.load_state_dict(torch.load('model.pth'))
```

### Checkpoint with Metadata
```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])
# Also available: checkpoint['epoch'], checkpoint['optimizer'], etc.
```

### Common State Dict Keys
```python
# Different repos use different keys
state_dict_keys = [
    'state_dict',      # Most common
    'model_state_dict',
    'model',
    'params',
    'net',
    'network',
    'G',               # For GANs (Generator)
    'generator',
]
```

### Handling DataParallel Prefix
```python
state_dict = torch.load('model.pth')

# Remove 'module.' prefix if present
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
```

### Strict Loading
```python
# Strict (default) - all keys must match
model.load_state_dict(state_dict, strict=True)

# Non-strict - allows missing/extra keys
model.load_state_dict(state_dict, strict=False)
```

## Preprocessing Patterns

### Pattern A: [0, 1] Normalization
```python
img = img.astype(np.float32) / 255.0
# Output: float32 array in range [0, 1]
```

### Pattern B: [-1, 1] Normalization
```python
img = img.astype(np.float32) / 255.0
img = img * 2 - 1
# Output: float32 array in range [-1, 1]
```

### Pattern C: ImageNet Normalization
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img = (img - mean) / std
```

### Pattern D: No Normalization (uint8)
```python
# Some models work with uint8 directly
img = img.astype(np.float32)  # [0, 255]
```

### Color Channel Handling
```python
# OpenCV loads as BGR, PIL loads as RGB
# Most PyTorch models expect RGB

# OpenCV to RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Or use PIL directly
from PIL import Image
img = Image.open(path).convert('RGB')
img = np.array(img)
```

### Dimension Ordering
```python
# NumPy/OpenCV: [H, W, C]
# PyTorch: [B, C, H, W]

# Convert HWC to BCHW
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

# Convert BCHW to HWC
img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
```

### Padding to Multiple
```python
def pad_to_multiple(img, multiple=8):
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
```

## Postprocessing Patterns

### Basic Clipping
```python
output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
```

### From [-1, 1] to [0, 255]
```python
output = (output + 1) / 2  # to [0, 1]
output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
```

### Denormalize ImageNet
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
output = output * std + mean
output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
```

### Remove Padding
```python
def remove_padding(img, original_h, original_w):
    return img[:original_h, :original_w]
```

## Tiling/Patch Processing

### Simple Tiling
```python
def process_tiles(img, model, tile_size=256, overlap=32):
    h, w = img.shape[:2]
    result = np.zeros_like(img)
    count = np.zeros_like(img)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img[y:y_end, x:x_end]

            processed = model(tile)

            result[y:y_end, x:x_end] += processed
            count[y:y_end, x:x_end] += 1

    return result / count
```

### Scale Factor Handling (Super-Resolution)
```python
def tile_sr(img, model, scale=4, tile_size=256, overlap=32):
    # Output size is scaled
    h, w = img.shape[:2]
    out_h, out_w = h * scale, w * scale
    result = np.zeros((out_h, out_w, 3))

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # Input tile
            tile = img[y:y+tile_size, x:x+tile_size]
            processed = model(tile)  # Output is scaled

            # Output position
            out_y, out_x = y * scale, x * scale
            # ... blend into result
```

## Device Handling

### Auto Device Selection
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)
```

### Multi-GPU
```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)
```

### Half Precision (FP16)
```python
# For faster inference with less memory
model = model.half()
input_tensor = input_tensor.half()

# Convert back for saving
output = output.float()
```

## Common Dependencies

### Core (Almost Always Required)
```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
pillow>=8.0.0
opencv-python>=4.4.0
```

### Optional (Task-Specific)
```
scipy          # Signal processing, metrics
scikit-image   # Image metrics (PSNR, SSIM)
tqdm           # Progress bars
pyyaml         # Config files
einops         # Tensor operations (transformers)
timm           # Pretrained vision models
lpips          # Perceptual metrics
```

### Model-Specific
```
# Transformer-based models
einops
timm

# GAN-based models
lpips
pytorch-fid

# OpenMMLab-based
mmcv-full
mmedit
```

### Profiling-Only (Often Not Needed for Inference)
```
thop           # Model FLOPs/params calculation
ptflops        # Alternative FLOPs calculator
fvcore         # Facebook's profiling tools
```

**Note**: Many repos include profiling imports that fail if packages aren't installed.
These can be made optional with try/except blocks:

```python
# Fix for missing profiling dependencies
try:
    from thop import profile
except ImportError:
    profile = None  # Only used for profiling, not inference

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None
```

## Checkpoint Formats

### PyTorch Native (.pth, .pt)
```python
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

### Full Checkpoint (.pth, .ckpt)
```python
checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'config': config,
}
torch.save(checkpoint, 'checkpoint.pth')
```

### ONNX (.onnx)
```python
# Export
torch.onnx.export(model, dummy_input, 'model.onnx')

# Import
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
output = session.run(None, {'input': input_np})
```

### TorchScript (.pt)
```python
# Export
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')

# Import
model = torch.jit.load('model_scripted.pt')
```
