# PCR-Map (ICPR code release)

Minimal reference implementation for the PCR-Map failure detection pipeline used in our ICPR submission.

## Structure
- src/pcrmap/: core implementation
- scripts/: entry scripts for generating result CSV and figures from local data

## Quick start
Install:
  pip install -r requirements.txt

Run (requires local fastMRI data; paths differ across machines):
  python scripts/run_batch.py --data_root "<YOUR_FASTMRI_ROOT>" --out_csv "outputs/results.csv"

Evaluate failure detection:
  python scripts/eval_failure.py --csv "outputs/results.csv" --out_csv "outputs/fd.csv"

## Notes
- No datasets are included.
- This release is intended for transparency and reference; exact reproduction may depend on environment and data availability.
