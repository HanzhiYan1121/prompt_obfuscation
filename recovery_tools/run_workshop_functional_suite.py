"""Run the frozen-model evaluation serially on one explicitly selected GPU.

This entry point never trains, discovers idle GPUs, or launches detached jobs.
All data/checkpoint locations come from release_config.py.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='Physical CUDA device index or UUID; exactly one device')
    parser.add_argument('--config', type=Path, help='JSON configuration; equivalent to RECOVERY_CONFIG')
    parser.add_argument('--data-root', type=Path, help='External historical assets; overrides RECOVERY_DATA_ROOT')
    parser.add_argument('--run-dir', type=Path, help='Output directory; overrides RECOVERY_RUN_DIR')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without reading models or creating outputs')
    args = parser.parse_args()
    if not args.gpu.strip() or ',' in args.gpu:
        parser.error('--gpu must name exactly one CUDA device')
    for key, value in [('RECOVERY_CONFIG', args.config), ('RECOVERY_DATA_ROOT', args.data_root), ('RECOVERY_RUN_DIR', args.run_dir)]:
        if value is not None:
            os.environ[key] = str(value.expanduser().resolve())
    from release_config import RUN_DIR, LOCAL_FILES_ONLY
    from analysis_tools.workshop_functional_benchmark import write

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(PROJECT_ROOT))
    if LOCAL_FILES_ONLY:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    commands = [
        ('prepare', ['analysis_tools/workshop_functional_benchmark.py', *(['--validate-existing'] if (RUN_DIR/'manifest.json').exists() else [])]),
        ('collect', ['recovery_tools/workshop_functional_victim.py', 'collect']),
        ('extract', ['recovery_tools/workshop_functional_victim.py', 'extract']),
        ('recover', ['recovery_tools/workshop_functional_attack.py', 'recover']),
        ('tgps', ['recovery_tools/workshop_functional_attack.py', 'tgps']),
        ('select', ['recovery_tools/workshop_functional_victim.py', 'select']),
        ('evaluate', ['recovery_tools/workshop_functional_victim.py', 'evaluate']),
        ('score', ['analysis_tools/report_workshop_functional_suite.py']),
    ]
    for name, command in commands:
        full = [sys.executable, '-u', *command]
        print(name, ':', full, flush=True)
        if args.dry_run:
            continue
        write(RUN_DIR/'status.json', {'status':'running', 'phase':name, 'no_training':True, 'time':time.time()})
        try:
            subprocess.run(full, cwd=PROJECT_ROOT, env=env, check=True)
        except BaseException as exc:
            write(RUN_DIR/'status.json', {'status':'failed', 'phase':name, 'no_training':True, 'error':str(exc), 'time':time.time()})
            raise
    if not args.dry_run:
        write(RUN_DIR/'status.json', {'status':'complete', 'no_training':True, 'time':time.time()})


if __name__ == '__main__':
    main()
