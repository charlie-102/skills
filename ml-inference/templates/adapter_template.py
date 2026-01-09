#!/usr/bin/env python3
"""
Adapter Template - Uses metadata.json for configuration
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add RU directory to path
sys.path.insert(0, str(Path(__file__).parent))


class Adapter:
    """Unified inference interface using metadata.json configuration."""

    def __init__(self, metadata_path: str = 'metadata.json',
                 weights: str = None, device: str = None, **kwargs):
        # Load metadata
        with open(metadata_path) as f:
            self.meta = json.load(f)

        # Device selection (CLI > metadata > default)
        self.device = device or self.meta.get('inference', {}).get('device', 'cpu')

        # Task info
        self.task_type = self.meta['task']['type']
        self.scale = self.meta['task'].get('params', {}).get('scale', 1)

        # Input/output config
        self.input_range = self.meta['input'].get('value_range', [0, 1])
        self.output_range = self.meta['output'].get('value_range', [0, 1])
        self.pad_multiple = self.meta['input'].get('pad_multiple')

        # Weight path resolution
        self.weights_path = self._resolve_weights(weights)

        # Initialize model based on type
        model_type = self.meta['model']['type']
        if model_type == 'single':
            self._init_single_model()
        elif model_type == 'multi_network':
            self._init_multi_network()
        elif model_type == 'diffusion':
            self._init_diffusion(**kwargs)

    def _resolve_weights(self, weights_override):
        """Resolve weight path from override or metadata default."""
        if weights_override:
            return weights_override

        default = self.meta['weights']['default']
        if isinstance(default, list):
            return [f"weights/{w}" for w in default]
        return f"weights/{default}"

    def _init_single_model(self):
        """Initialize single model architecture."""
        # TODO: Import and initialize your model class
        # from models.network import Model
        # self.model = Model()
        # self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        # self.model.to(self.device).eval()
        raise NotImplementedError("Implement _init_single_model for your model")

    def _init_multi_network(self):
        """Initialize multi-network architecture."""
        # TODO: Import and initialize your network classes
        # self.networks = {}
        # for name, info in self.meta['model']['networks'].items():
        #     # Initialize and load each network
        #     pass
        raise NotImplementedError("Implement _init_multi_network for your model")

    def _init_diffusion(self, **kwargs):
        """Initialize diffusion model."""
        # TODO: Import and initialize sampler
        # num_steps = kwargs.get('num_steps') or self.meta['inference'].get('num_steps', 15)
        raise NotImplementedError("Implement _init_diffusion for your model")

    def preprocess(self, image_path: str) -> torch.Tensor:
        """Load and preprocess input image."""
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1]

        # Convert to input range if needed
        if self.input_range != [0, 1]:
            # e.g., [0,1] -> [-1,1]: (x - 0.5) / 0.5
            img_np = (img_np - 0.5) / 0.5

        # HWC -> BCHW
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

        # Pad if required
        if self.pad_multiple:
            h, w = tensor.shape[2:]
            pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
            pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
            if pad_h or pad_w:
                tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
            self._original_size = (h, w)
        else:
            self._original_size = None

        return tensor.to(self.device)

    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run model inference."""
        with torch.no_grad():
            # TODO: Call your model
            # output = self.model(input_tensor)
            raise NotImplementedError("Implement inference for your model")
        return output

    def postprocess(self, output_tensor: torch.Tensor) -> Image.Image:
        """Convert output tensor to PIL Image."""
        # Remove padding if applied
        if self._original_size:
            h, w = self._original_size
            h_out, w_out = h * self.scale, w * self.scale
            output_tensor = output_tensor[:, :, :h_out, :w_out]

        # Convert from output range to [0, 1]
        if self.output_range != [0, 1]:
            # e.g., [-1,1] -> [0,1]: x * 0.5 + 0.5
            output_tensor = output_tensor * 0.5 + 0.5

        # BCHW -> HWC
        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(output)

    def run(self, input_path: str, output_path: str, **kwargs) -> None:
        """Full inference pipeline."""
        tensor = self.preprocess(input_path)
        output = self.inference(tensor)
        result = self.postprocess(output)
        result.save(output_path)

        # Print info
        in_img = Image.open(input_path)
        print(f"Task: {self.task_type}")
        print(f"Input: {input_path} ({in_img.size[0]}x{in_img.size[1]})")
        print(f"Output: {output_path} ({result.size[0]}x{result.size[1]})")

    @classmethod
    def from_metadata(cls, metadata_path: str = 'metadata.json', **kwargs):
        """Factory method to create adapter from metadata."""
        return cls(metadata_path, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='RU Inference Adapter')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    parser.add_argument('--metadata', '-m', default='metadata.json', help='Metadata JSON path')
    parser.add_argument('--weights', '-w', default=None, help='Override weight path')
    parser.add_argument('--device', '-d', default=None, choices=['cpu', 'cuda', 'mps'])

    # Task-specific overrides
    parser.add_argument('--scale', type=int, default=None, help='Scale factor (SR)')
    parser.add_argument('--noise-level', type=int, default=None, help='Noise level (denoising)')
    parser.add_argument('--num-steps', type=int, default=None, help='Diffusion steps')
    parser.add_argument('--mask', default=None, help='Mask path (inpainting)')

    args = parser.parse_args()

    adapter = Adapter(
        metadata_path=args.metadata,
        weights=args.weights,
        device=args.device,
        num_steps=args.num_steps,
    )

    adapter.run(
        args.input,
        args.output,
        scale=args.scale,
        noise_level=args.noise_level,
        mask=args.mask,
    )


if __name__ == '__main__':
    main()
