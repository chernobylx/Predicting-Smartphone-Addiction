# Tasks — Predicting Smartphone Addiction

Rewritten 2026-09-01 against what actually shipped. The previous version had drifted badly
in both directions: it still listed "baseline model + first submission" and "model tuning
and ensembling" as backlog after both had shipped, and it claimed `pixi run check` was
failing on `notebooks/eda.py` when the lint had already been clean for some time.

## 🔄 In Progress
_Nothing in flight._

## 📋 Backlog

### Opened by the winner review
- [ ] **Re-open missingness as value recovery, not indicator features.** The 0.5017 figure is
      correct about the *pattern* of missing cells and was wrongly generalised to the whole
      area. The generator constraint pins some missing drivers exactly; `features.structural`
      builds bounds but stops there. 1st place reports missing values as "one new large
      source of signal" — this is the biggest single gap the review found.
- [ ] Pair and feature-on-feature statistics — on 7th place's list of useful families, absent
      here, and cheap
- [ ] Widen the model zoo before widening the stack (7th place measured 21 level-1 families;
      this project used three). The ledger already says representation earns weight and
      hyperparameters do not, and a different architecture is a different representation.
- [ ] Allow negative weights in `stack.hillclimb` — it cannot currently subtract a member, so
      an over-represented error direction has no way out. 7th place moved OOF 0.970820 →
      0.970849 by subtracting 25% of one stack and 10% of another.
- [ ] Record the private-leaderboard rank beside the public one; the competition shook up on
      private and only the public rank is currently written down

### Standing
- [x] ~~Record the competition deadline~~ — resolved: the competition closed **2026-08-31**
      (the 1st-place write-up is dated that day)
- [ ] Restore `data/raw/` on this machine (`pixi run data`) — currently empty, so nothing
      that touches real data can be run or re-verified here
- [ ] Run `pixi run members` end to end on real data and record the reduced stack's honest
      held-out number in the README, beside the historical 0.97013
- [ ] Regenerate the three data-dependent figures once data is restored: distinct-value
      counts vs the 255/2047 bin thresholds, raw vs target-encoded univariate AUC per
      column, and the member correlation heatmap

## ✅ Done
- [x] Winner review against the 1st- and 7th-place write-ups: score comparison, three things
      this project got right, three it got wrong, and what the top of the board did
      differently (2026-09-01)
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
