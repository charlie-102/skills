# RU Metadata Schema (metadata.json)

A unified schema for all image restoration task variations.

## Schema Overview

```json
{
  "name": "string",
  "version": "string",
  "task": { },
  "model": { },
  "weights": { },
  "input": { },
  "output": { },
  "inference": { },
  "source": { }
}
```

## Full Schema

```json
{
  "name": "SCUNet",
  "version": "1.0.0",

  "task": {
    "type": "denoising",
    "description": "Real-world image denoising",
    "variants": ["gaussian", "real"]
  },

  "model": {
    "type": "single",
    "architecture": "SCUNet",
    "framework": "pytorch"
  },

  "weights": {
    "default": "scunet_color_real_psnr.pth",
    "options": {
      "scunet_color_real_psnr.pth": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        "sha256": "...",
        "description": "Real denoising, optimized for PSNR"
      },
      "scunet_color_real_gan.pth": {
        "url": "https://...",
        "description": "Real denoising, GAN-based"
      }
    }
  },

  "input": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1],
    "dtype": "float32"
  },

  "output": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1],
    "dtype": "float32"
  },

  "inference": {
    "device": "cpu",
    "tile_size": null,
    "tile_overlap": 32,
    "batch_size": 1
  },

  "source": {
    "repo": "https://github.com/cszn/SCUNet",
    "paper": "https://arxiv.org/abs/2203.13278",
    "license": "Apache-2.0"
  }
}
```

---

## Task Types

### Denoising
```json
{
  "task": {
    "type": "denoising",
    "variants": ["gaussian", "real", "poisson"],
    "params": {
      "noise_level": [15, 25, 50]
    }
  }
}
```

### Super-Resolution
```json
{
  "task": {
    "type": "super_resolution",
    "variants": ["bicubic", "real", "blind"],
    "params": {
      "scale": 4,
      "supported_scales": [2, 4, 8]
    }
  }
}
```

### Deblurring
```json
{
  "task": {
    "type": "deblurring",
    "variants": ["motion", "gaussian", "defocus"],
    "params": {
      "blur_kernel_size": null
    }
  }
}
```

### Inpainting
```json
{
  "task": {
    "type": "inpainting",
    "variants": ["free_form", "rectangular"],
    "params": {
      "mask_required": true,
      "mask_format": "binary"
    }
  }
}
```

### JPEG Artifact Removal
```json
{
  "task": {
    "type": "jpeg_artifact_removal",
    "params": {
      "quality_factor": [10, 20, 40]
    }
  }
}
```

### Face Restoration
```json
{
  "task": {
    "type": "face_restoration",
    "variants": ["blind", "guided"],
    "params": {
      "face_detection": true,
      "background_enhance": false
    }
  }
}
```

### Colorization
```json
{
  "task": {
    "type": "colorization",
    "params": {
      "input_grayscale": true
    }
  }
}
```

---

## Model Types

### Single Model
```json
{
  "model": {
    "type": "single",
    "architecture": "SCUNet",
    "num_params": "15.2M"
  }
}
```

### Multi-Network
```json
{
  "model": {
    "type": "multi_network",
    "architecture": "DASR",
    "networks": {
      "generator": "MSRResNetDynamic",
      "predictor": "Degradation_Predictor"
    }
  }
}
```

### Diffusion
```json
{
  "model": {
    "type": "diffusion",
    "architecture": "ResShift",
    "networks": {
      "unet": "SwinUNet",
      "autoencoder": "AutoencoderKL"
    },
    "params": {
      "num_steps": 15,
      "sampler": "DDPM",
      "guidance_scale": 1.0
    }
  }
}
```

---

## Input/Output Specifications

### Standard Image
```json
{
  "input": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1],
    "dtype": "float32",
    "pad_multiple": 64
  },
  "output": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1],
    "dtype": "float32",
    "scale_factor": 1
  }
}
```

### With Mask (Inpainting)
```json
{
  "input": {
    "image": {
      "channels": 3,
      "color_space": "RGB",
      "value_range": [0, 1]
    },
    "mask": {
      "channels": 1,
      "value_range": [0, 1],
      "description": "1 = area to inpaint"
    }
  }
}
```

### Grayscale Input (Colorization)
```json
{
  "input": {
    "channels": 1,
    "color_space": "L",
    "value_range": [0, 1]
  },
  "output": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1]
  }
}
```

---

## Inference Parameters

```json
{
  "inference": {
    "device": "cpu",
    "tile_size": 512,
    "tile_overlap": 32,
    "batch_size": 1,
    "fp16": false,
    "deterministic": true,
    "seed": 42
  }
}
```

### Diffusion-Specific
```json
{
  "inference": {
    "device": "cpu",
    "num_steps": 15,
    "sampler": "DDPM",
    "guidance_scale": 1.0,
    "eta": 0.0,
    "seed": 42
  }
}
```

---

## Complete Examples

### SCUNet (Denoising)
```json
{
  "name": "SCUNet",
  "version": "1.0.0",
  "task": {
    "type": "denoising",
    "variants": ["real"]
  },
  "model": {
    "type": "single",
    "architecture": "SCUNet"
  },
  "weights": {
    "default": "scunet_color_real_psnr.pth",
    "options": {
      "scunet_color_real_psnr.pth": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
      }
    }
  },
  "input": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1],
    "pad_multiple": 64
  },
  "output": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1],
    "scale_factor": 1
  },
  "inference": {
    "device": "cpu",
    "tile_size": null,
    "batch_size": 1
  }
}
```

### DASR (4x Super-Resolution)
```json
{
  "name": "DASR",
  "version": "1.0.0",
  "task": {
    "type": "super_resolution",
    "variants": ["blind"],
    "params": {
      "scale": 4
    }
  },
  "model": {
    "type": "multi_network",
    "architecture": "DASR",
    "networks": {
      "generator": {
        "name": "net_g",
        "class": "MSRResNetDynamic"
      },
      "predictor": {
        "name": "net_p",
        "class": "Degradation_Predictor"
      }
    }
  },
  "weights": {
    "default": ["net_g.pth", "net_p.pth"],
    "options": {
      "net_g.pth": {
        "url": "https://drive.google.com/..."
      },
      "net_p.pth": {
        "url": "https://drive.google.com/..."
      }
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
    "device": "cpu",
    "batch_size": 1
  }
}
```

### ResShift (Diffusion SR)
```json
{
  "name": "ResShift",
  "version": "1.0.0",
  "task": {
    "type": "super_resolution",
    "variants": ["real", "bicubic"],
    "params": {
      "scale": 4
    }
  },
  "model": {
    "type": "diffusion",
    "architecture": "ResShift",
    "networks": {
      "unet": "SwinUNet",
      "autoencoder": "AutoencoderKL"
    },
    "params": {
      "num_steps": 15,
      "sampler": "DDPM"
    }
  },
  "weights": {
    "default": "resshift_realsrx4_s15.pth",
    "options": {
      "resshift_realsrx4_s15.pth": {
        "url": "https://drive.google.com/...",
        "config": "realsr_swinunet_realesrgan256_journal.yaml"
      },
      "resshift_bicx4_s15.pth": {
        "url": "https://drive.google.com/...",
        "config": "bicx4_swinunet_lpips.yaml"
      }
    }
  },
  "input": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [-1, 1]
  },
  "output": {
    "channels": 3,
    "color_space": "RGB",
    "value_range": [0, 1],
    "scale_factor": 4
  },
  "inference": {
    "device": "cpu",
    "num_steps": 15,
    "tile_size": 512,
    "tile_overlap": 64,
    "seed": 12345
  }
}
```

---

## Using metadata.json in adapter.py

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

class Adapter:
    def __init__(self, metadata_path: str = 'metadata.json',
                 weights: str = None, device: str = None):
        # Load metadata
        with open(metadata_path) as f:
            self.meta = json.load(f)

        # Override with CLI args if provided
        self.device = device or self.meta['inference'].get('device', 'cpu')

        # Get weight path
        if weights:
            self.weights_path = weights
        else:
            default = self.meta['weights']['default']
            if isinstance(default, list):
                self.weights_path = [f"weights/{w}" for w in default]
            else:
                self.weights_path = f"weights/{default}"

        # Task-specific setup
        self.task_type = self.meta['task']['type']
        self.scale = self.meta['task'].get('params', {}).get('scale', 1)

        # Initialize model based on type
        self._init_model()

    def _init_model(self):
        model_type = self.meta['model']['type']
        if model_type == 'single':
            self._init_single_model()
        elif model_type == 'multi_network':
            self._init_multi_network()
        elif model_type == 'diffusion':
            self._init_diffusion()

    def preprocess(self, image_path: str):
        # Use metadata for preprocessing
        value_range = self.meta['input']['value_range']
        # ... normalize to value_range

    def inference(self, input_tensor):
        # Task-specific inference
        pass

    def postprocess(self, output_tensor):
        # Use metadata for postprocessing
        value_range = self.meta['output']['value_range']
        scale = self.meta['output'].get('scale_factor', 1)
        # ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--metadata', '-m', default='metadata.json')
    parser.add_argument('--weights', '-w', default=None)
    parser.add_argument('--device', '-d', default=None)

    # Task-specific overrides
    parser.add_argument('--scale', type=int, default=None)
    parser.add_argument('--noise-level', type=int, default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--mask', default=None, help='Mask for inpainting')

    args = parser.parse_args()

    adapter = Adapter(args.metadata, args.weights, args.device)
    adapter.run(args.input, args.output,
                mask=args.mask,
                scale=args.scale,
                noise_level=args.noise_level,
                num_steps=args.num_steps)


if __name__ == '__main__':
    main()
```

---

## CLI Examples

```bash
# Basic usage (uses metadata.json defaults)
python adapter.py -i input.png -o output.png

# Override device
python adapter.py -i input.png -o output.png -d cuda

# Use specific weights
python adapter.py -i input.png -o output.png -w weights/model_gan.pth

# Task-specific: SR with scale
python adapter.py -i input.png -o output.png --scale 4

# Task-specific: Denoising with noise level
python adapter.py -i input.png -o output.png --noise-level 25

# Task-specific: Diffusion with steps
python adapter.py -i input.png -o output.png --num-steps 50

# Task-specific: Inpainting with mask
python adapter.py -i input.png -o output.png --mask mask.png
```
