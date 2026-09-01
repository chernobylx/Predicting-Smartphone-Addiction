# Tasks — Predicting Smartphone Addiction

Rewritten 2026-09-01 against what actually shipped. The previous version had drifted badly
in both directions: it still listed "baseline model + first submission" and "model tuning
and ensembling" as backlog after both had shipped, and it claimed `pixi run check` was
failing on `notebooks/eda.py` when the lint had already been clean for some time.

## 🔄 In Progress
- [ ] Portfolio polish: notebook narrative pass on `eda.py` / `eda_profile.py`
- [ ] Review the competition's 1st-place solution against this project's ledger
      (blocked: no `~/.kaggle/kaggle.json` on this machine, and no S6E8 write-up
      published or indexed as of 2026-09-01)

## 📋 Backlog
- [ ] Record the competition deadline in the README — still unconfirmed; Kaggle renders as
      a JS SPA, so it could not be read without API credentials
- [ ] Restore `data/raw/` on this machine (`pixi run data`) — currently empty, so nothing
      that touches real data can be run or re-verified here
- [ ] Run `pixi run members` end to end on real data and record the reduced stack's honest
      held-out number in the README, beside the historical 0.97013
- [ ] Regenerate the three data-dependent figures once data is restored: distinct-value
      counts vs the 255/2047 bin thresholds, raw vs target-encoded univariate AUC per
      column, and the member correlation heatmap

## ✅ Done
- [x] README restructure: leads with the thesis, figures carry the numbers, and an explicit
      section on what does and does not reproduce (2026-09-01)
- [x] Four exported figures drawn from the ledger, plus their Vega-Lite specs
      (`pixi run figures`, needs no competition data) (2026-09-01)
- [x] Corrected the CV→LB offset claim: measured +0.00125 ± 0.00015, falling monotonically,
      not "stable at +0.0012 ± 0.0001" (2026-09-01)
- [x] **Leakage-enforcement battery** — the project's stated thesis, finally tested. The
      full target-encoding path must score chance under a shuffled label
      (`tests/test_members.py`) (2026-09-01)
- [x] `members.py`: reproducible member generation. `stack.py` had been reading from a
      directory no committed code ever wrote (2026-09-01)
- [x] `features.py`: one shared representation layer, so every arm is a paired comparison;
      encoding keys now selected by column name rather than by position (2026-09-01)
- [x] 26-member hill-climb rank stack — LB 0.97013, rank 728/3353 (2026-08-31)
- [x] Full experiment ledger in the README (2026-08-30)
- [x] Representation pipeline: fine binning, value-level encoding, generator structure —
      OOF AUC 0.968069 (2026-08-30)
- [x] LightGBM baseline and first submission — OOF AUC 0.963947 (2026-08-29)
- [x] Profiling notebook on 10% stratified sample: quality triage, distributions, target
      associations (`notebooks/eda_profile.py`) (2026-08-29)
- [x] Download competition data via `pixi run data` (2026-08-29)
- [x] Project scaffold: pixi env, src package, tests, tooling (2026-08-24)

## ❌ Dropped
- Missingness mechanism check (can any feature predict another's nullness?) — answered by
  the ledger instead: missingness indicators and null counts have univariate AUC 0.5017,
  i.e. MCAR and inert. Nothing further to learn here.
- "Fix lint findings in notebooks/eda.py" — already clean; `ruff check src tests notebooks`
  passes.
