# Data: TRAC-1 Hindi (not stored in this repo)

**Content warning:** the data contains abusive text. No raw examples appear anywhere in this repo.

Licence: CC-BY-NC-SA 4.0 (non-commercial, share-alike). The data cannot be redistributed, so `data/` is git-ignored.

## How to get it
1. Fill the data request form linked from https://sites.google.com/view/trac1/shared-task
2. Extract the archives and copy these four files into `data/raw/`:
   - `agr_hi_train.csv`   (12,000 rows, Facebook, train)
   - `agr_hi_dev.csv`     (3,001 rows, Facebook, validation)
   - `agr_hi_fb_gold.csv` (970 rows, Facebook, test 1)
   - `agr_hi_tw_gold.csv` (1,194 rows, Twitter, test 2 / out-of-domain)
3. The files have no header: `id, text, label` with label in OAG / CAG / NAG.
4. Never open or save them in Excel (it garbles Devanagari). Read them with Python, UTF-8.

## Verify and build the splits
    python src/verify_data.py --write    # first time only, then commit data_checksums.json
    python src/verify_data.py            # any other machine: confirms identical files
    python src/split.py                  # writes splits/*.txt ids and reports/dedupe_report.json

Binary label: NAG = 0, OAG + CAG = 1 (decision D2b).
