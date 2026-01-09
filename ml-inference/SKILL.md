---
name: ml-inference
description: Convert GitHub ML inference repos into self-contained Reproducible Units (RU).
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch, Task
---

# ML Inference -> Reproducible Unit (RU)

**Goal**: GitHub URL -> Self-contained RU with adapter.py that auto-downloads weights.

## Quick Start

```
Input:  GitHub URL (e.g., https://github.com/cszn/SCUNet)
Output: <name>_RU/ in zoo/units/
```

## Workflow

1. **Clone & Analyze**
   - Look for existing `inference_simple.py`, `demo.py`, or `test.py` first - use as reference
   - Identify model type (single/multi/diffusion), architecture files
   - Check README for weight URLs and usage examples

2. **Test Locally** - Verify original repo inference works

3. **Create RU** - Copy minimal source, create adapter.py with:
   - Robust weight loading (see pattern below)
   - Auto-download for missing weights
   - metadata.json configuration

4. **Create metadata.json** - Task type, weights URLs, input/output specs

5. **Test** - Run `python test_all.py`

6. **Update Skills** - Add new patterns learned

## Adapter Template

```python
def download_weights(url: str, dest: Path) -> bool:
    if dest.exists(): return True
    if 'drive.google.com' in url:
        print(f"Manual download: {url}\nSave to: {dest}")
        return False
    urllib.request.urlretrieve(url, str(dest))
    return True

def _load_weights(model, weights_path, device):
    """Robust weight loading for various checkpoint formats."""
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Handle nested checkpoint keys
    if isinstance(state_dict, dict):
        for key in ['params', 'state_dict', 'model_state_dict', 'model']:
            if key in state_dict:
                state_dict = state_dict[key]
                break

    # Handle DataParallel 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

class Adapter:
    def __init__(self, metadata_path='metadata.json', weights=None, device=None):
        with open(meta_file) as f: self.meta = json.load(f)
        # Auto-download weights if not found
        if not weights_path.exists():
            download_weights(self.meta['weights']['options'][default]['url'], weights_path)
        # Use _load_weights for robust loading
        _load_weights(self.model, weights_path, self.device)
```

## Common Fixes

- **Registry stub**: `ARCH_REGISTRY = Registry('arch')` for BasicSR repos
- **Lazy imports**: `create_dataset = None; def _get()...` for training-only deps
- **CPU fallback**: Replace `.cuda()` with `.to(device)`
- **Docker**: Use `libgl1` not `libgl1-mesa-glx`
- **Robust weight loading**: Handle nested keys + DataParallel prefix (see template above)

## Checklist Before Creating RU

1. [ ] Found existing inference script in repo? Use as reference
2. [ ] Identified all model networks (single vs multi-network)?
3. [ ] Found weight download URLs in README/releases?
4. [ ] Tested original inference works locally?
5. [ ] Created registry/utils stubs if BasicSR-based?
6. [ ] Added robust weight loading (nested keys + module. prefix)?
7. [ ] Added auto-download for weights?

## Reference

Full documentation in `zoo/docs/`:
- RU_SPEC.md - Full specification
- METADATA_SCHEMA.md - metadata.json schema
- COMMON_PATTERNS.md - Repository patterns
- WORKFLOW.md - Detailed workflow
