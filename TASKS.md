# Tasks — Predicting Smartphone Addiction

## 🔄 In Progress

## 📋 Backlog
- [ ] Record the competition deadline in the README (metric is done: ROC-AUC)
- [ ] Build the leakage-enforcement core + adversarial regression test battery (the project's thesis — do this before any module)
- [ ] Missingness mechanism check: can any feature predict another's nullness? (in progress in eda.py)
- [ ] Fix lint findings in notebooks/eda.py (unsorted/unused imports; `pixi run check` fails on it)
- [ ] Baseline model + first submission (simple GBM, default params, CV score)
- [ ] Feature engineering informed by EDA
- [ ] Model tuning and ensembling (LightGBM / XGBoost)
- [ ] Portfolio polish: results section in README, exported figures, clean notebook narrative

## ✅ Done
- [x] Profiling notebook on 10% stratified sample: quality triage, distributions, target associations (notebooks/eda_profile.py) (2026-08-29)
- [x] Download competition data via `pixi run data` (2026-08-29)
- [x] Project scaffold: pixi env, src package, tests, tooling (2026-08-24)
