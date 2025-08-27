#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml, os, re, pandas as pd
from pathlib import Path

REQ_COLS = ['eid','filename','label']

def check_file(path, msg):
    if not Path(path).exists():
        raise FileNotFoundError(f"{msg} not found: {path}")

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    pe, ck = cfg['predictions_csv'], cfg['checkpoints']
    left_pat = re.compile(cfg.get('left_eye_pattern','21015'))
    prob_col = cfg.get('probability_column','DL_raw_prob')

    for cohort in ['primary','secondary']:
        check_file(pe[cohort], f'{cohort} predictions CSV')
        check_file(ck[cohort], f'{cohort} checkpoint')
        df = pd.read_csv(pe[cohort])
        have = set(df.columns.str.lower())
        filename_ok = any(c in have for c in ['filename','image_filename','left_eye_filename','image'])
        if not filename_ok:
            raise ValueError(f"{cohort}: predictions missing a filename-like column")
        for c in ['eid','label']:
            if c not in have:
                raise ValueError(f"{cohort}: missing column '{c}'")
        if prob_col.lower() not in have:
            raise ValueError(f"{cohort}: missing probability column '{prob_col}'")

        fname_col = next((c for c in df.columns if c.lower() in ['left_eye_filename','image_filename','filename','image']), None)
        bad = df[~df[fname_col].astype(str).str.contains(left_pat, na=False)]
        if not bad.empty:
            print(f"[WARN] {cohort}: {len(bad)} filenames not matching left-eye pattern. Example: {bad[fname_col].iloc[0]}")

    # images_dir can be a single string or a dict per cohort
    img_root_cfg = cfg['images_dir']
    if isinstance(img_root_cfg, dict):
        for cohort in ['primary','secondary']:
            if cohort not in img_root_cfg:
                raise FileNotFoundError(f'images_dir missing key: {cohort}')
            p = Path(img_root_cfg[cohort])
            if not p.exists():
                raise FileNotFoundError(f'images_dir for {cohort} not found: {p}')
    else:
        p = Path(img_root_cfg)
        if not p.exists():
            raise FileNotFoundError(f'images_dir not found: {p}')

    print('[OK] Inputs look good.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    main(a.config)
