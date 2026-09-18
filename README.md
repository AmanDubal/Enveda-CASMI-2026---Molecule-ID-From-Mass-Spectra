# CASMI 2026 code and submission

## What is delivered

- `submission.csv`: real predictions for the 400 supplied test molecule IDs, produced by the reference-library retrieval code. This is **not a scored or best-accuracy submission**.
- `casmi_retrieval_submission.ipynb`: self-contained Kaggle notebook that reproduces that submission using the official attached competition data.
- `casmi_retrieval.py`: command-line version of the same method.
- `casmi_corrected_denovo.ipynb`: experimental GPU training notebook derived from inversion's tutorial, with substantive correctness fixes. It has not been trained or evaluated end-to-end in this environment.
- `retrieval_audit.csv`: per-molecule candidate counts, mass fallback flags, top spectral similarity and mass error. Similarities are not confidence probabilities.
- `run_report.json`: execution statistics. `validation_mrr25: null` means accuracy has not been measured, not that it is zero.

## Submit on Kaggle

1. Import `casmi_retrieval_submission.ipynb` into the competition's notebook editor.
2. Attach the official Enveda CASMI 2026 competition dataset. The notebook checks both common Kaggle mount paths; adjust `data_dir` if needed.
3. Provision numpy, pandas, pyarrow and `rdkit==2026.3.3` through Kaggle Dependency Manager before the offline commit.
4. Disable Internet, then Save Version / Run All. Check that the output includes `/kaggle/working/submission.csv`.
5. Submit the notebook output through the competition. The competition requires notebook submissions; downloading a CSV alone does not submit an entry.

No Google Drive access or external download is required in either submitted notebook. Original input files are not bundled because the training file is about 3 GB.

## Retrieval method and limits

The input training file has 2,539,608 spectra and 277,566 distinct supplied SMILES. The test file has 1,213 spectra for 400 molecules. Unlike the linked tutorial's data subset, retrieval uses all supplied training sources. Candidate molecules are neutral, single-component reference structures. Their exact neutral masses are compared with adduct-corrected test masses (15 ppm, with a 0.003 Da tolerance floor). Unknown test adducts stop execution rather than being silently treated as protonated ions.

Within the candidate pool, the code removes precursor peaks, retains up to 128 intense fragments, square-root transforms intensities, computes one-to-one fragment/neutral-loss cosine matches, and aggregates evidence across each test molecule's spectra. It deduplicates tautomer-canonical InChIKey14 identities using RDKit 2026.03.3 and returns at most 25 candidates. No test labels were used. Ranking weights and mass thresholds are initial settings, not optimized hyperparameters.

When no reference structure fits the mass tolerance, the method uses nearby reference masses and flags that molecule in the audit. These are weak fallback guesses. Charged compounds, mixtures, unusual adduct chemistry and structures absent from the library are limitations. A valid CSV is not evidence of predictive accuracy.

## What changed in the de novo experiment

Reference: https://www.kaggle.com/code/inversion/casmi-denovo-tutorial-notebook (version 9 retrieved through Kaggle's public download endpoint).

- Next-token targets replace the original same-token reconstruction objective.
- Generation passes the entire prefix. The original supplied only the most recent token despite having no KV cache.
- Added decoder positional embeddings and causal masking during both training and generation.
- Prevented padding, BOS and unknown-token sampling; discarded unfinished generations.
- Added explicit float32 peak tensors and safe empty-spectrum preprocessing.
- Replaced noncanonical top-one/Tanimoto model selection with tautomer-aware MRR@25.
- Added a connectivity-disjoint validation split, bounded streaming data sampling and best-checkpoint restoration.
- Reduced the architecture to 384 dimensions / four layers for a more practical starting point on a single GPU; this is an unvalidated compute tradeoff.
- Added separate training/prediction batch sizes, compatible mixed precision, a training timer, and a total-time guard.

The neural notebook has no trained weights. Its nine-hour feasibility and MRR@25 must be measured in Kaggle. Per-epoch validation uses one spectrum per held-out structure; final test inference pools up to three spectra per molecule, so validation does not fully reproduce the test setting. Generation has no KV cache and can be slow. A timeout raises an error rather than producing a silently incomplete CSV.

## Local reproduction

```bash
python -m pip install numpy pandas pyarrow rdkit==2026.3.3
python casmi_retrieval.py --data-dir /path/to/competition-data --output-dir ./output
```

To establish improvement, run the reference and corrected neural model on the same held-out connectivity split, use identical scoring canonicalization, and compare MRR@25 under the same runtime budget. Do not infer a leaderboard win from the code fixes or retrieval similarities.

## Checks performed

The retrieval pipeline ran end-to-end on the supplied files in about 314 seconds in this workspace. It made 854,268 spectral comparisons and needed zero out-of-tolerance mass fallbacks. RDKit 2026.03.3 was installed and used. Notebook schemas and every code cell's Python syntax were checked. A small CPU regression test passed for causal masking, finite decoder loss/backpropagation, positional gradients, full-prefix generation and correctly shifted teacher targets. These tests establish implementation behavior, not model accuracy.

## Important: supplied training/test overlap

All 1,213 supplied test spectra, covering all 400 molecules, match reference spectra from the training file's `enveda-180` source after the documented preprocessing. Every matching spectrum has one reference connectivity; all 400 top-ranked guesses agree with those matched reference connectivities. The check compares same-adduct spectra within 0.005 Da precursor difference, processed fragment masses rounded to six decimals, and normalized square-root intensities rounded to eight decimals. This is not an exact-byte claim about the original unprocessed arrays. See `overlap_report.json` and `overlap_audit.csv`.

This explains the perfect spectral similarities. It is evidence of supplied-data overlap, **not a measured leaderboard score or proof of de novo generalization**. The reference tutorial excludes this source; we have not established why. The retrieval notebook uses it because it is in the supplied training file; if competition rules specifically exclude that source, this notebook must be changed and rerun. Hidden test molecules may not overlap.
