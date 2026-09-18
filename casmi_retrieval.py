"""Offline CASMI reference-library baseline. No de novo or accuracy claims.

Inputs are the official train.parquet, test.parquet, sample_submission.csv.
Run: python casmi_retrieval.py --data-dir /path/to/data --output-dir /path/to/output
Requires numpy, pandas, pyarrow, rdkit==2026.3.3. No network calls.
"""
import argparse
import json
import time
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
TAUTOMER = rdMolStandardize.TautomerEnumerator()
# Ionic mass added to neutral M, for singly charged test adducts.
ADDUCT_DELTA = {
    '[M+H]+': 1.007276466621, '[M-H]-': -1.007276466621,
    '[M+Na]+': 22.989218, '[M+K]+': 38.963158,
    '[M+NH4]+': 18.033823, '[M+Cl]-': 34.969402,
    '[M+CH2O2-H]-': 44.998201, '[M+HCOO]-': 44.998201,
}

@lru_cache(maxsize=250000)
def identity(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = TAUTOMER.Canonicalize(mol)
        key = Chem.MolToInchiKey(mol).split('-')[0]
        if len(key) != 14:
            return None
        return key, Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None

def prepare_peaks(mzs, intensities, precursor, losses=False):
    mz = np.asarray(mzs, dtype=float)
    intensity = np.asarray(intensities, dtype=float)
    if mz.shape != intensity.shape:
        raise ValueError('Mismatched peak/intensity lengths')
    keep = (np.isfinite(mz) & np.isfinite(intensity) & (mz > 0)
            & (intensity > 0) & (np.abs(mz - precursor) > 1.5))
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) == 0:
        return np.array([]), np.array([])
    keep = np.argsort(intensity)[-128:]
    mz, intensity = mz[keep], np.sqrt(intensity[keep])
    if losses:
        mz = precursor - mz
        keep = mz > 1.5
        mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz)
    return mz[order], intensity[order] / max(np.linalg.norm(intensity), 1e-12)

def cosine(a, b, tolerance=0.02):
    """Greedy one-to-one matching; all pairs within absolute m/z tolerance."""
    am, ai = a
    bm, bi = b
    if not len(am) or not len(bm):
        return 0.0
    left = np.searchsorted(bm, am - tolerance)
    right = np.searchsorted(bm, am + tolerance, side='right')
    pairs = [(ai[i] * bi[j], i, j) for i in np.flatnonzero(right > left)
             for j in range(left[i], right[i])]
    used_a, used_b, score = set(), set(), 0.0
    for value, i, j in sorted(pairs, reverse=True):
        if i not in used_a and j not in used_b:
            score += value
            used_a.add(i)
            used_b.add(j)
    return min(float(score), 1.0)

def validate_submission(frame, sample):
    if list(frame.columns) != ['molecule_id', 'smiles'] or frame.empty:
        raise ValueError('Invalid submission columns or empty output')
    if frame.isna().any().any() or frame.molecule_id.duplicated().any():
        raise ValueError('Null values or duplicate molecule IDs')
    if set(frame.molecule_id) != set(sample.molecule_id):
        raise ValueError('Submission molecule IDs differ from sample submission')
    for value in frame.smiles:
        guesses = value.split(';')
        keys = [identity(s) for s in guesses]
        if not 1 <= len(guesses) <= 25 or any(x is None for x in keys):
            raise ValueError('Invalid SMILES or candidate count')
        if len({x[0] for x in keys}) != len(keys):
            raise ValueError('Duplicate scoring identities in a row')

def main(data_dir, output_dir):
    start = time.monotonic()
    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError('Install rdkit==2026.3.3 for competition identity matching')
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(data_dir / 'test.parquet')
    sample = pd.read_csv(data_dir / 'sample_submission.csv', dtype={'molecule_id': str})
    if sample.molecule_id.isna().any() or sample.molecule_id.duplicated().any():
        raise ValueError('Invalid sample submission IDs')
    if set(test.molecule_id) != set(sample.molecule_id):
        raise ValueError('Test and sample molecule IDs do not match')
    unknown = sorted(set(test.adduct) - set(ADDUCT_DELTA))
    if unknown:
        raise ValueError(f'Add verified mass mappings for these adducts: {unknown}')
    test['neutral_mass'] = test.precursor_mz - test.adduct.map(ADDUCT_DELTA)
    neutral = test.groupby('molecule_id').neutral_mass.median()
    reference = pq.ParquetFile(data_dir / 'train.parquet')

    # Use every supplied training source; no test labels or external structures.
    all_smiles = set()
    for batch in reference.iter_batches(batch_size=100000, columns=['ingest_lib', 'normalized_smiles']):
        df = batch.to_pandas()
        all_smiles.update(df['normalized_smiles'].dropna())
    print('Reference structures:', len(all_smiles), flush=True)
    mass_records = []
    for s in sorted(all_smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None and Chem.GetFormalCharge(mol) == 0 and '.' not in s:
            mass_records.append((Descriptors.ExactMolWt(mol), s))
    mass_records.sort()
    masses = np.array([r[0] for r in mass_records])
    if not len(masses):
        raise ValueError('No neutral reference structures found')

    # Exact mass constrains the pool; a nearest-mass fallback is explicitly audited.
    pools, details, smiles_to_queries = {}, [], {}
    for mid in sample.molecule_id:
        target = float(neutral[mid])
        tolerance = max(target * 15e-6, 0.003)
        lo, hi = np.searchsorted(masses, [target - tolerance, target + tolerance])
        ix = np.arange(lo, hi)
        fallback = not len(ix)
        if fallback:
            center = np.searchsorted(masses, target)
            nearby = np.arange(max(0, center - 200), min(len(masses), center + 200))
            ix = nearby[np.argsort(np.abs(masses[nearby] - target))[:100]]
        pool = {}
        for j in ix:
            mass, s = mass_records[j]
            ident = identity(s)
            if ident is None:
                continue
            key, canonical = ident
            ppm = abs(mass - target) / target * 1e6
            pool.setdefault(key, {'smiles': canonical, 'ppm': ppm, 'spectra': {}})
            smiles_to_queries.setdefault(s, []).append((mid, key))
        if not pool:
            raise ValueError(f'No valid candidates for {mid}; do not fabricate a prediction')
        pools[mid] = pool
        details.append({'molecule_id': mid, 'neutral_mass': target,
                        'mass_fallback': bool(fallback), 'candidate_count': len(pool)})

    spectra = {}
    for mid, rows in test.groupby('molecule_id', sort=False):
        spectra[mid] = [(str(r.spectrum_id), str(r.adduct), str(r.ionization_mode),
                         prepare_peaks(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz),
                         prepare_peaks(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, True))
                        for r in rows.itertuples()]
    columns = ['ingest_lib', 'normalized_smiles', 'adduct', 'ionization_mode',
               'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
    compared = 0
    for batch_no, batch in enumerate(reference.iter_batches(batch_size=32768, columns=columns)):
        df = batch.to_pandas()
        df = df[df.normalized_smiles.isin(smiles_to_queries)]
        for r in df.itertuples():
            if not np.isfinite(r.precursor_mz):
                continue
            peaks = prepare_peaks(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
            losses = prepare_peaks(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, True)
            for mid, key in smiles_to_queries[r.normalized_smiles]:
                evidence = pools[mid][key]['spectra']
                for sid, adduct, mode, query, qloss in spectra[mid]:
                    if mode != r.ionization_mode:
                        continue
                    if adduct == r.adduct:
                        score = 0.85 * cosine(query, peaks) + 0.15 * cosine(qloss, losses)
                    else:
                        score = 0.5 * cosine(qloss, losses)
                    evidence[sid] = max(evidence.get(sid, 0.), score)
                    compared += 1
        if batch_no % 10 == 0:
            print('Reference batch', batch_no, 'comparisons', compared, flush=True)

    rows, audit = [], []
    for detail in details:
        mid = detail['molecule_id']
        ranked = []
        for key, c in pools[mid].items():
            support = np.array([c['spectra'].get(sid, 0.) for sid, *_ in spectra[mid]])
            spectral = 0.7 * support.max() + 0.3 * support.mean()
            score = spectral - 0.02 * min(c['ppm'] / 15., 100.)
            ranked.append((score, -c['ppm'], key, c['smiles'], spectral))
        ranked.sort(reverse=True)
        ranked = ranked[:25]
        rows.append((mid, ';'.join(r[3] for r in ranked)))
        audit.append({**detail, 'top_score': ranked[0][0], 'top_spectral_score': ranked[0][4],
                      'top_mass_error_ppm': -ranked[0][1], 'n_guesses': len(ranked)})
    submission = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
    validate_submission(submission, sample)
    submission.to_csv(output_dir / 'submission.csv', index=False)
    pd.DataFrame(audit).to_csv(output_dir / 'retrieval_audit.csv', index=False)
    report = {'method': 'Mass-constrained reference-spectrum retrieval; no de novo generation',
              'molecules': len(submission), 'spectra': len(test),
              'reference_unique_smiles': len(all_smiles), 'spectral_comparisons': compared,
              'mass_fallback_molecules': sum(d['mass_fallback'] for d in details),
              'rdkit_version': rdBase.rdkitVersion, 'elapsed_seconds': time.monotonic() - start,
              'validation_mrr25': None,
              'limitation': 'Unscored baseline; structures outside the candidate library cannot be recovered.'}
    (output_dir / 'run_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='/kaggle/working')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
