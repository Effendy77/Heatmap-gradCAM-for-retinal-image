#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml, os, json, hashlib, pathlib, time

def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main(cfg_path: str, panels_dir: str, out_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    entry = {
        'timestamp': time.time(),
        'experiment_name': cfg.get('experiment_name'),
        'checkpoints': {k:(sha256(v) if os.path.exists(v) else None) for k,v in cfg.get('checkpoints',{}).items()},
        'predictions_csv': {k:(sha256(v) if os.path.exists(v) else None) for k,v in cfg.get('predictions_csv',{}).items()},
        'device': cfg.get('device'),
        'backbone': cfg.get('backbone'),
        'target_layer': cfg.get('target_layer'),
        'threshold': cfg.get('threshold'),
        'panels': {}
    }
    pdir = pathlib.Path(panels_dir)
    for p in pdir.rglob('*.png'):
        entry['panels'][str(p)] = sha256(str(p))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(entry, f, indent=2)
    print('wrote', out_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--panels_dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.config, a.panels_dir, a.out)
    print(f'Provenance log written to {a.out}')
    print('You can now run `generate_gradcam.py` to generate Grad-CAM images.')
    print('After that, run `compose_panels.py` to create overview panels.')
    print('Finally, run `provenance_log.py` to log the experiment details.')  