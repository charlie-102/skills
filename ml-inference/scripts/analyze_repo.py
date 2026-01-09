#!/usr/bin/env python3
"""
Analyze an ML repository to identify key components for inference.

Usage:
    python analyze_repo.py /path/to/repo
    python analyze_repo.py /path/to/repo --output report.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


def find_files_by_pattern(repo_path: Path, patterns: List[str]) -> List[Path]:
    """Find files matching any of the given patterns."""
    matches = []
    for pattern in patterns:
        matches.extend(repo_path.rglob(pattern))
    return sorted(set(matches))


def find_inference_scripts(repo_path: Path) -> List[Dict]:
    """Identify potential inference/test scripts."""
    patterns = [
        '**/test*.py', '**/main_test*.py', '**/inference*.py',
        '**/inference_*.py',  # ResShift pattern: inference_resshift.py
        '**/predict*.py', '**/demo*.py', '**/eval*.py',
        '**/run*.py', '**/main*.py', '**/app.py'  # Gradio apps
    ]
    scripts = []

    for f in find_files_by_pattern(repo_path, patterns):
        if '__pycache__' in str(f) or '.git' in str(f):
            continue

        # Check if it's likely an inference script
        content = f.read_text(errors='ignore')
        score = 0
        indicators = []

        if 'argparse' in content:
            score += 1
            indicators.append('CLI arguments')
        if re.search(r'load_state_dict|torch\.load', content):
            score += 2
            indicators.append('model loading')
        if re.search(r'\.eval\(\)', content):
            score += 1
            indicators.append('eval mode')
        if re.search(r'with torch\.no_grad|torch\.no_grad\(\)', content):
            score += 1
            indicators.append('inference mode')
        if re.search(r'cv2\.imread|Image\.open|imageio\.imread', content):
            score += 1
            indicators.append('image loading')
        if re.search(r'cv2\.imwrite|\.save\(|imageio\.imwrite', content):
            score += 1
            indicators.append('image saving')

        if score >= 2:
            scripts.append({
                'path': str(f.relative_to(repo_path)),
                'score': score,
                'indicators': indicators
            })

    return sorted(scripts, key=lambda x: x['score'], reverse=True)


def find_model_definitions(repo_path: Path) -> List[Dict]:
    """Find model/network definition files."""
    model_dirs = ['models', 'model', 'networks', 'network', 'arch', 'archs', 'src/model', 'src/models']
    results = []

    # Check common directories
    for dir_name in model_dirs:
        model_dir = repo_path / dir_name
        if model_dir.exists():
            for f in model_dir.rglob('*.py'):
                if '__pycache__' in str(f) or '__init__' in f.name:
                    continue
                content = f.read_text(errors='ignore')

                # Look for class definitions inheriting from nn.Module
                classes = re.findall(r'class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\)', content)
                if classes:
                    results.append({
                        'path': str(f.relative_to(repo_path)),
                        'classes': classes
                    })

    # Check for nested framework patterns (e.g., basicsr/archs/)
    for framework_dir in repo_path.iterdir():
        if framework_dir.is_dir() and not framework_dir.name.startswith('.'):
            for subdir in ['archs', 'models', 'arch', 'model']:
                nested_dir = framework_dir / subdir
                if nested_dir.exists():
                    for f in nested_dir.rglob('*.py'):
                        if '__pycache__' in str(f) or '__init__' in f.name:
                            continue
                        content = f.read_text(errors='ignore')
                        classes = re.findall(r'class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\)', content)
                        if classes:
                            results.append({
                                'path': str(f.relative_to(repo_path)),
                                'classes': classes
                            })

    # Also check root level
    for f in repo_path.glob('*.py'):
        content = f.read_text(errors='ignore')
        classes = re.findall(r'class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\)', content)
        if classes:
            results.append({
                'path': str(f.relative_to(repo_path)),
                'classes': classes
            })

    return results


def detect_framework(repo_path: Path) -> Dict:
    """Detect ML framework patterns (BasicSR, MMEditing, etc.)."""
    frameworks = {
        'basicsr': False,
        'mmcv': False,
        'mmedit': False,
        'registry_pattern': False,
        'config_driven': False,
        'multi_network': False
    }
    details = []

    # Check for BasicSR
    if (repo_path / 'basicsr').exists():
        frameworks['basicsr'] = True
        details.append('BasicSR framework detected')

    # Check for MMEditing/MMCV
    for name in ['mmedit', 'mmcv', 'mmsr']:
        if (repo_path / name).exists():
            frameworks['mmcv'] = True
            details.append(f'{name} framework detected')

    # Check for registry pattern
    for f in repo_path.rglob('*.py'):
        if '__pycache__' in str(f):
            continue
        try:
            content = f.read_text(errors='ignore')
            if '@.*REGISTRY.register' in content or 'ARCH_REGISTRY' in content or 'MODEL_REGISTRY' in content:
                frameworks['registry_pattern'] = True
                details.append('Registry pattern detected')
                break
        except:
            continue

    # Check for config-driven testing
    options_dir = repo_path / 'options'
    configs_dir = repo_path / 'configs'
    if options_dir.exists() or configs_dir.exists():
        frameworks['config_driven'] = True
        details.append('Config-driven testing detected')

    # Check for multi-network architecture (e.g., generator + discriminator, predictor + generator)
    for f in repo_path.rglob('*.py'):
        if '__pycache__' in str(f):
            continue
        try:
            content = f.read_text(errors='ignore')
            if re.search(r'net_[gp]|net_g.*net_p|network_g.*network_p', content):
                frameworks['multi_network'] = True
                details.append('Multi-network architecture detected')
                break
        except:
            continue

    return {
        'frameworks': frameworks,
        'details': list(set(details))
    }


def find_pretrained_weights(repo_path: Path) -> Dict:
    """Find locations for pretrained weights."""
    weight_dirs = ['model_zoo', 'pretrained', 'weights', 'checkpoints', 'ckpt', 'experiments']
    found_dirs = []
    found_files = []

    # Check directories
    for dir_name in weight_dirs:
        weight_dir = repo_path / dir_name
        if weight_dir.exists():
            found_dirs.append(dir_name)
            # Check for existing weight files
            for ext in ['*.pth', '*.pt', '*.ckpt', '*.pkl']:
                found_files.extend([str(f.relative_to(repo_path)) for f in weight_dir.rglob(ext)])

    # Check for download scripts
    download_scripts = []
    for pattern in ['*download*.py', '*setup*.py']:
        for f in repo_path.glob(pattern):
            content = f.read_text(errors='ignore')
            if 'download' in content.lower() or 'wget' in content or 'gdown' in content:
                download_scripts.append(str(f.relative_to(repo_path)))

    # Check README for download links
    readme_links = []
    for readme in repo_path.glob('README*'):
        content = readme.read_text(errors='ignore')
        # Look for common download links
        links = re.findall(r'https?://[^\s\)]+(?:\.pth|\.pt|drive\.google|dropbox|mega\.nz|\.zip)[^\s\)]*', content)
        readme_links.extend(links[:5])  # Limit to first 5

    return {
        'directories': found_dirs,
        'existing_files': found_files[:10],  # Limit output
        'download_scripts': download_scripts,
        'readme_links': readme_links
    }


def find_utilities(repo_path: Path) -> List[str]:
    """Find utility directories and files."""
    util_patterns = ['utils', 'util', 'lib', 'common', 'tools']
    found = []

    for pattern in util_patterns:
        util_dir = repo_path / pattern
        if util_dir.exists() and util_dir.is_dir():
            found.append(pattern)

    return found


def find_config_files(repo_path: Path) -> List[str]:
    """Find configuration files."""
    config_patterns = ['*.yaml', '*.yml', '*.json', 'configs/**/*.yaml', 'options/**/*.yaml']
    configs = []

    for pattern in config_patterns:
        for f in repo_path.glob(pattern):
            if '.git' not in str(f):
                configs.append(str(f.relative_to(repo_path)))

    return configs[:20]  # Limit output


def check_requirements(repo_path: Path) -> Dict:
    """Analyze requirements/dependencies."""
    req_files = []
    dependencies = set()

    # Check various requirement files
    for pattern in ['requirements*.txt', 'setup.py', 'environment*.yml', 'environment*.yaml']:
        for f in repo_path.glob(pattern):
            req_files.append(str(f.relative_to(repo_path)))

            content = f.read_text(errors='ignore')
            if f.suffix == '.txt':
                # Parse requirements.txt
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pkg = re.split(r'[>=<\[]', line)[0].strip()
                        if pkg:
                            dependencies.add(pkg)
            elif f.suffix in ['.yml', '.yaml']:
                # Simple YAML parsing for conda env
                for match in re.findall(r'^\s*-\s*(\w[\w-]*)', content, re.MULTILINE):
                    dependencies.add(match)

    return {
        'files': req_files,
        'key_dependencies': sorted(dependencies)[:20]
    }


def analyze_preprocessing(repo_path: Path) -> Dict:
    """Analyze preprocessing patterns used in the repo."""
    patterns_found = {
        'normalization': [],
        'color_space': [],
        'data_format': []
    }

    for f in repo_path.rglob('*.py'):
        if '__pycache__' in str(f) or '.git' in str(f):
            continue

        try:
            content = f.read_text(errors='ignore')
        except:
            continue

        # Check normalization
        if '/ 255' in content or '/255' in content:
            if '[0, 1]' not in patterns_found['normalization']:
                patterns_found['normalization'].append('[0, 1]')
        if '* 2 - 1' in content or '*2-1' in content:
            if '[-1, 1]' not in patterns_found['normalization']:
                patterns_found['normalization'].append('[-1, 1]')
        if 'mean=' in content and 'std=' in content:
            if 'ImageNet' not in patterns_found['normalization']:
                patterns_found['normalization'].append('ImageNet-style')

        # Check color space
        if 'BGR2RGB' in content or 'cvtColor' in content:
            if 'BGR conversion' not in patterns_found['color_space']:
                patterns_found['color_space'].append('BGR conversion')
        if "convert('RGB')" in content:
            if 'PIL RGB' not in patterns_found['color_space']:
                patterns_found['color_space'].append('PIL RGB')

        # Check data format
        if 'permute(2, 0, 1)' in content or 'transpose(2, 0, 1)' in content:
            if 'HWC to CHW' not in patterns_found['data_format']:
                patterns_found['data_format'].append('HWC to CHW')
        if 'unsqueeze(0)' in content:
            if 'batch dimension added' not in patterns_found['data_format']:
                patterns_found['data_format'].append('batch dimension added')

    return patterns_found


def generate_report(repo_path: Path) -> Dict:
    """Generate a comprehensive analysis report."""
    repo_path = Path(repo_path).resolve()

    if not repo_path.exists():
        return {'error': f'Repository path does not exist: {repo_path}'}

    report = {
        'repository': str(repo_path),
        'name': repo_path.name,
        'framework': detect_framework(repo_path),
        'inference_scripts': find_inference_scripts(repo_path),
        'model_definitions': find_model_definitions(repo_path),
        'pretrained_weights': find_pretrained_weights(repo_path),
        'utilities': find_utilities(repo_path),
        'config_files': find_config_files(repo_path),
        'requirements': check_requirements(repo_path),
        'preprocessing_patterns': analyze_preprocessing(repo_path)
    }

    return report


def print_report(report: Dict):
    """Print a formatted report to console."""
    print("=" * 60)
    print(f"ML Repository Analysis: {report.get('name', 'Unknown')}")
    print("=" * 60)

    # Framework detection
    framework = report.get('framework', {})
    if framework.get('details'):
        print("\n🏗️ FRAMEWORK DETECTION:")
        for detail in framework.get('details', []):
            print(f"  - {detail}")
        frameworks = framework.get('frameworks', {})
        active = [k for k, v in frameworks.items() if v]
        if active:
            print(f"  Patterns: {', '.join(active)}")

    print("\n📜 INFERENCE SCRIPTS (ranked by likelihood):")
    for script in report.get('inference_scripts', [])[:5]:
        print(f"  - {script['path']} (score: {script['score']})")
        print(f"    Indicators: {', '.join(script['indicators'])}")

    print("\n🧠 MODEL DEFINITIONS:")
    for model in report.get('model_definitions', [])[:10]:
        print(f"  - {model['path']}")
        print(f"    Classes: {', '.join(model['classes'][:5])}")

    print("\n⚖️ PRETRAINED WEIGHTS:")
    weights = report.get('pretrained_weights', {})
    if weights.get('directories'):
        print(f"  Directories: {', '.join(weights['directories'])}")
    if weights.get('existing_files'):
        print(f"  Found files: {len(weights['existing_files'])} weight file(s)")
    if weights.get('download_scripts'):
        print(f"  Download scripts: {', '.join(weights['download_scripts'])}")
    if weights.get('readme_links'):
        print(f"  README links: {len(weights['readme_links'])} download link(s) found")

    print("\n🔧 UTILITIES:")
    utils = report.get('utilities', [])
    print(f"  Directories: {', '.join(utils) if utils else 'None found'}")

    print("\n⚙️ CONFIG FILES:")
    configs = report.get('config_files', [])
    for cfg in configs[:5]:
        print(f"  - {cfg}")
    if len(configs) > 5:
        print(f"  ... and {len(configs) - 5} more")

    print("\n📦 REQUIREMENTS:")
    reqs = report.get('requirements', {})
    if reqs.get('files'):
        print(f"  Files: {', '.join(reqs['files'])}")
    if reqs.get('key_dependencies'):
        print(f"  Key packages: {', '.join(reqs['key_dependencies'][:10])}")

    print("\n🔄 PREPROCESSING PATTERNS:")
    preproc = report.get('preprocessing_patterns', {})
    if preproc.get('normalization'):
        print(f"  Normalization: {', '.join(preproc['normalization'])}")
    if preproc.get('color_space'):
        print(f"  Color space: {', '.join(preproc['color_space'])}")
    if preproc.get('data_format'):
        print(f"  Data format: {', '.join(preproc['data_format'])}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze ML repository for inference setup')
    parser.add_argument('repo_path', type=str, help='Path to the repository')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    parser.add_argument('--json', action='store_true', help='Print JSON to stdout')

    args = parser.parse_args()

    report = generate_report(args.repo_path)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    elif args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == '__main__':
    main()
