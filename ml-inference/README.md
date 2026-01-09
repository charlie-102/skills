# ML Inference Skill

Convert GitHub ML inference repositories into self-contained Reproducible Units (RU).

## What This Skill Does

Takes a GitHub URL for an ML image processing model and creates a standalone RU folder that:
- Runs inference without the original repo's dependencies
- Auto-downloads weights if missing
- Works in Docker for reproducibility
- Has a unified `adapter.py` interface

## Quick Start

```
Input:  GitHub URL (e.g., https://github.com/cszn/SCUNet)
Output: <name>_RU/ folder ready for inference
```

## Supported Model Types

| Type | Example | Description |
|------|---------|-------------|
| Single Model | SCUNet | One network, one weight file |
| Multi-Network | DASR | Predictor + generator networks |
| Diffusion | ResShift | Sampler + UNet + autoencoder |

## Supported Tasks

- Denoising (gaussian, real-world)
- Super-resolution (2x, 4x, 8x)
- Deblurring (motion, defocus)
- Face restoration
- JPEG artifact removal
- Inpainting
- Colorization

## Files in This Skill

```
ml-inference/
├── SKILL.md              # Main skill instructions (Claude reads this)
├── README.md             # This file
├── WORKFLOW.md           # Step-by-step conversion workflow
├── RU_SPEC.md            # Reproducible Unit specification
├── METADATA_SCHEMA.md    # metadata.json schema reference
├── COMMON_PATTERNS.md    # Common repo patterns and fixes
├── TROUBLESHOOTING.md    # Common issues and solutions
├── templates/
│   ├── adapter_template.py      # Adapter class template
│   ├── inference_template.py    # Full inference script template
│   ├── metadata.schema.json     # JSON schema for validation
│   └── test_all.py              # RU test runner
└── scripts/
    ├── analyze_repo.py      # Analyze repo structure
    ├── quick_setup.py       # Auto-generate inference setup
    ├── docker_test.py       # Docker testing utilities
    └── generate_inference.py # Generate inference scripts
```

## Typical Workflow

1. **Clone & Analyze** - Identify model type, find existing inference scripts
2. **Test Locally** - Verify original repo works
3. **Create RU** - Copy minimal source, create adapter.py
4. **Add metadata.json** - Configure weights, input/output specs
5. **Test RU** - Run test_all.py
6. **Update Skill** - Add learnings back to skill docs

## Common Fixes

| Issue | Fix |
|-------|-----|
| `No module 'basicsr'` | `source ~/.zshrc && conda activate <env>` |
| Registry import error | Create stub in `basicsr/utils/registry.py` |
| `.cuda()` fails | Replace with `.to(device)` |
| Weight loading fails | Check nested keys: `params`, `state_dict`, `model` |
| `module.` prefix | Strip with `{k[7:]: v for k, v in state_dict.items()}` |
| Docker libGL error | Use `libgl1` not `libgl1-mesa-glx` |

## RU Output Structure

```
<name>_RU/
├── adapter.py          # Unified inference interface
├── metadata.json       # Configuration and weight URLs
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
├── download_weights.sh # Weight download script
├── weights/            # Model weights (downloaded separately)
├── <model_source>/     # Minimal model source files
├── test_input/         # Sample test images
└── test_output/        # Expected outputs
```

## Usage After RU Creation

```bash
# Local inference
python adapter.py -i input.png -o output.png

# Docker inference
docker build -t model-ru .
docker run --rm -v $(pwd)/io:/io model-ru -i /io/input.png -o /io/output.png
```

## Adding New Learnings

After converting a new model, update:
- `SKILL.md` - Add to "Common Fixes" if new pattern found
- `COMMON_PATTERNS.md` - Document new repo structures
- `TROUBLESHOOTING.md` - Add new issues encountered
