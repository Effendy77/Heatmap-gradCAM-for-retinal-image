#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml, os, pandas as pd, numpy as np, pathlib, re

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
    df['outcome'] = np.where((y==1)&(p>=thr),'TP',
                     np.where((y==0)&(p>=thr),'FP',
                     np.where((y==1)&(p<thr),'FN','TN')))
    df['margin'] = (p - thr).abs()
    return df

def choose_filename_column(df: pd.DataFrame):
    # prefer columns that look like real filenames with an extension
    candidates = [c for c in df.columns if c.lower() in
                  ['image_filename','image_filename_x','image_filename_y',
                   'left_eye_filename','filename','image']]
    for c in candidates:
        if df[c].astype(str).str.contains(r'\.(png|jpg|jpeg)$', case=False, na=False).mean() > 0.5:
            return c, 'as_is'
    # else return the best available and indicate we should synthesize
    for c in candidates:
        return c, 'synthesize'
    raise ValueError('No filename-like column found')

def select_margin(df: pd.DataFrame, k: int) -> pd.DataFrame:
    # Select top-k rows with largest margin per group
    return df.nlargest(k, 'margin')

def main(cfg_path: str, out_dir: str):
    cfg = yaml.safe_load(open(cfg_path))
    prob_col = cfg.get('probability_column','DL_raw_prob')
    thr = float(cfg.get('threshold',0.345))
    k = int(cfg.get('topk_per_outcome',12))
    strat = cfg['stratifications']
    left_pat = str(cfg.get('left_eye_pattern','21015'))

    out_root = pathlib.Path(out_dir); out_root.mkdir(parents=True, exist_ok=True)

    for cohort in ['primary','secondary']:
        df = pd.read_csv(cfg['predictions_csv'][cohort]).copy()

        fname_col, mode = choose_filename_column(df)
        if mode == 'as_is':
            df = df.rename(columns={fname_col: 'filename'})
        else:
            # Build filenames like "<eid>_<left_pat>_0_0.png"
            if 'eid' not in df.columns:
                raise ValueError(f"{cohort}: cannot synthesize filenames without an 'eid' column")
            df['filename'] = df['eid'].astype(str) + f'_{left_pat}_0_0.png'

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
        # new code: stratum is built from the subgroup ONLY (group_cols), not including 'outcome'
        # We still select top-k per outcome, and 'outcome' remains as a column used for TP/FP/FN/TN rows.
        topk_per_outcome = k  # use the value from config
        for key, g in df.groupby(group_cols + ['outcome'], dropna=False):
            sel = select_margin(g, topk_per_outcome)
            if sel.empty:
                continue
            # key is a tuple: (*group_cols, outcome)
            if not isinstance(key, tuple):
                key = (key,)

            group_key = key[:-1]          # drop the trailing 'outcome'
            outcome   = key[-1]           # keep outcome for the row label/category

            # Build a clean stratum name from the subgroup fields only
            # Convert NAs to 'NA' to avoid 'None' text in names
            group_key_safe = tuple('NA' if (x is None or (isinstance(x, float) and pd.isna(x))) else x for x in group_key)
            stratum = '_'.join(map(str, group_key_safe))

            # Ensure the 'outcome' column is present (some select_margin variants return a view)
            if 'outcome' not in sel.columns:
                sel = sel.assign(outcome=outcome)

            rows.append(sel.assign(cohort=cohort, stratum=stratum))

        out_df = pd.concat(rows, ignore_index=True) if rows else df.head(0)
        (out_root / f'{cohort}_selections.csv').parent.mkdir(parents=True, exist_ok=True)
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
