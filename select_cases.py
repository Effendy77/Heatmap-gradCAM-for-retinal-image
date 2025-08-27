#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml, os, pandas as pd, numpy as np, pathlib

def ensure_age_bins(df: pd.DataFrame, bins):
    if 'age_bin' in df.columns:
        return df
    if 'Age_at_baseline' not in df.columns:
        return df
    def pick(age):
        try:
            a = float(age)
        except Exception:
            return None
        for b in bins:
            if b['min'] <= a < b['max']:
                return b['label']
        return None
    df = df.copy()
    df['age_bin'] = df['Age_at_baseline'].apply(pick)
    return df

def add_outcome(df: pd.DataFrame, prob_col: str, label_col: str, thr: float):
    df = df.copy()
    p = df[prob_col].astype(float)
    y = df[label_col].astype(int)
    df['outcome'] = np.where((y==1)&(p>=thr),'TP', np.where((y==0)&(p>=thr),'FP', np.where((y==1)&(p<thr),'FN','TN')))
    df['margin'] = (p - thr).abs()
    return df

def select_margin(g: pd.DataFrame, k: int):
    return g.sort_values('margin').head(k)

def choose_filename_column(df: pd.DataFrame) -> str:
    for c in ['left_eye_filename','image_filename','filename','image']:
        if c in df.columns:
            return c
    raise ValueError('No filename-like column found')

def main(cfg_path: str, out_dir: str):
    cfg = yaml.safe_load(open(cfg_path))
    prob_col = cfg.get('probability_column','DL_raw_prob')
    thr = float(cfg.get('threshold',0.5))
    k = int(cfg.get('topk_per_outcome',12))
    strat = cfg['stratifications']

    out_root = pathlib.Path(out_dir); out_root.mkdir(parents=True, exist_ok=True)

    for cohort in ['primary','secondary']:
        df = pd.read_csv(cfg['predictions_csv'][cohort])
        fname_col = choose_filename_column(df)
        df = df.rename(columns={fname_col: 'filename'})

        # compute age bins if needed
        for s in strat[cohort]:
            if 'age_bin' in s.get('by', []):
                df = ensure_age_bins(df, s.get('bins', []))

        # ensure flags exist in secondary
        if cohort == 'secondary':
            for required in ['diabetes_prevalent','hypertension_prevalent']:
                if required not in df.columns:
                    df[required] = 'NA'

        df = add_outcome(df, prob_col, label_col='mace' if 'mace' in df.columns else 'label', thr=thr)

        # grouping columns
        group_cols = []
        for s in strat[cohort]:
            group_cols.extend(s['by'])
        for c in group_cols:
            if c not in df.columns:
                df[c] = 'NA'

        rows = []
        for key, g in df.groupby(group_cols + ['outcome'], dropna=False):
            sel = select_margin(g, k)
            if sel.empty:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            sel = sel.assign(cohort=cohort, stratum='_'.join(map(str, key)))
            rows.append(sel)
        out_df = pd.concat(rows, ignore_index=True) if rows else df.head(0)
        out_df.to_csv(out_root / f'{cohort}_selections.csv', index=False)
        print(f'[{cohort}] wrote selections: {len(out_df)} rows')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.config, a.out)
    print('[OK] Selections complete.')
    print(f'Output written to: {a.out}')
    print('You can now run `select_cases.py` to finalize the case selection.')