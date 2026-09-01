import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Profiling a 10% sample of `train.csv`

    **Purpose.** Disciplined first pass over the S6E8 competition data: structure,
    per-column profile, quality triage, and a bounded discovery loop — every finding
    labeled exploratory, key numbers independently recomputed.

    **Data.** `data/raw/train.csv` (44,855,546 bytes,
    sha256 `f4669147311c76eb…`), downloaded via `pixi run data` from
    [Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8).

    **Environment.** pixi default env — Python 3.14.7, polars 1.43.2,
    altair 6.2.2, marimo 0.24.0. Date: 2026-08-29.

    **Frame.** One row = one survey respondent (`id`). No temporal columns.
    Population: synthetic data generated from a smartphone-usage survey
    (`data/raw/original.csv` is the seed dataset). Target: `addicted_label` (0/1).
    """)
    return


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    from sklearn.model_selection import train_test_split as tts

    from smartphone_addiction.paths import TRAIN_CSV

    alt.data_transformers.enable("vegafusion")
    return TRAIN_CSV, alt, cs, mo, np, pl, tts


@app.cell
def _(pl):
    SEED = 42
    schema_override = {
        "gender": pl.Categorical,
        "stress_level": pl.Enum(["Low", "Medium", "High"]),
        "academic_work_impact": pl.Categorical,
    }
    return SEED, schema_override


@app.cell
def _(TRAIN_CSV, pl, schema_override):
    train = pl.read_csv(TRAIN_CSV, schema_overrides=schema_override)
    grain_checks = {
        "rows": train.height,
        "cols": train.width,
        "id_unique": train["id"].n_unique() == train.height,
        "id_range": (train["id"].min(), train["id"].max()),
        "exact_dup_rows_excl_id": train.drop("id").is_duplicated().sum(),
        "target_rate_full": round(train["addicted_label"].mean(), 6),
    }
    grain_checks
    return grain_checks, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Grain verified: `id` is unique (691,369 rows, 0–691,368, no exact duplicate
    rows once `id` is excluded), so one row is one respondent and per-row
    aggregation is safe.

    ## Sampling decision (forking-paths ledger #1)

    Per the analysis brief, all profiling below runs on a **10% sample**
    (n = 69,136), drawn with `train_test_split(train_size=0.1, random_state=42,
    stratified by target)` — the same seed and stratification convention as
    `eda.py`, so both notebooks describe the same subsample. Alternative not
    taken: profiling the full table (cheap enough here; the sample was chosen to
    keep plots and pairwise work fast and to leave the remaining 90% untouched
    for later confirmatory checks).
    """)
    return


@app.cell
def _(SEED, train, tts):
    eda, _rest = tts(
        train,
        train_size=0.1,
        random_state=SEED,
        stratify=train.select("addicted_label"),
    )
    eda
    return (eda,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Column classification

    | Column | Class | Notes |
    |---|---|---|
    | `id` | identifier | unique, drop for modeling |
    | `age` | metric | integer-valued, 18–35 |
    | `daily_screen_time_hours` | metric | hours/day |
    | `social_media_hours` | metric | hours/day |
    | `gaming_hours` | metric | hours/day |
    | `work_study_hours` | metric | hours/day |
    | `sleep_hours` | metric | hours/day |
    | `notifications_per_day` | metric | integer-valued |
    | `app_opens_per_day` | metric | integer-valued |
    | `weekend_screen_time` | metric | hours/day (weekend) |
    | `gender` | dimension | Male / Female / Other |
    | `stress_level` | dimension (ordinal) | Low < Medium < High |
    | `academic_work_impact` | dimension | Yes / No |
    | `addicted_label` | target | binary, 71% positive |
    """)
    return


@app.cell
def _(eda, pl):
    profile = pl.DataFrame(
        {
            "column": eda.columns,
            "nulls": [eda[c].null_count() for c in eda.columns],
            "null_rate": [
                round(eda[c].null_count() / eda.height, 4) for c in eda.columns
            ],
            "distinct": [eda[c].n_unique() for c in eda.columns],
        }
    ).sort("null_rate", descending=True)
    profile
    return


@app.cell
def _(cs, eda):
    eda.select(cs.numeric().exclude("id")).describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).with_columns(cs.numeric().round(2))
    return


@app.cell
def _(cs, eda, mo, pl):
    _feats = eda.drop("id", "addicted_label")
    rows_with_any_null = _feats.select(
        pl.any_horizontal(pl.all().is_null()).sum()
    ).item()
    _num_cols = cs.expand_selector(eda, cs.numeric().exclude("id", "addicted_label"))
    zero_neg_sweep = pl.DataFrame(
        {
            "column": list(_num_cols),
            "zeros": [eda.select(pl.col(c).eq(0).sum()).item() for c in _num_cols],
            "negatives": [eda.select(pl.col(c).lt(0).sum()).item() for c in _num_cols],
        }
    )
    mo.vstack(
        [
            mo.md(
                f"**Row-level null burden:** {rows_with_any_null:,} of "
                f"{eda.height:,} rows ({rows_with_any_null / eda.height:.1%}) "
                "have at least one null feature."
            ),
            zero_neg_sweep,
        ]
    )
    return


@app.cell
def _(eda, pl):
    _tb = ["daily_screen_time_hours", "work_study_hours", "sleep_hours"]
    _tb_complete = eda.select(_tb).drop_nulls()
    _wd = eda.select("daily_screen_time_hours", "weekend_screen_time").drop_nulls()
    _sg = eda.select(
        "social_media_hours", "gaming_hours", "daily_screen_time_hours"
    ).drop_nulls()
    consistency_checks = pl.DataFrame(
        [
            {
                "check": "screen + work + sleep > 24 h (nulls treated as 0)",
                "violations": eda.select(
                    (
                        pl.col("daily_screen_time_hours").fill_null(0)
                        + pl.col("work_study_hours").fill_null(0)
                        + pl.col("sleep_hours").fill_null(0)
                    )
                    .gt(24)
                    .sum()
                ).item(),
                "n_evaluated": eda.height,
            },
            {
                "check": "screen + work + sleep > 24 h (complete cases)",
                "violations": _tb_complete.select(
                    pl.sum_horizontal(pl.all()).gt(24).sum()
                ).item(),
                "n_evaluated": _tb_complete.height,
            },
            {
                "check": "implied weekday screen time (7·daily − 2·weekend)/5 < 0",
                "violations": _wd.select(
                    (
                        (
                            pl.col("daily_screen_time_hours") * 7
                            - pl.col("weekend_screen_time") * 2
                        )
                        / 5
                    )
                    .lt(0)
                    .sum()
                ).item(),
                "n_evaluated": _wd.height,
            },
            {
                "check": "social_media + gaming > daily screen time",
                "violations": _sg.select(
                    (pl.col("social_media_hours") + pl.col("gaming_hours"))
                    .gt(pl.col("daily_screen_time_hours"))
                    .sum()
                ).item(),
                "n_evaluated": _sg.height,
            },
        ]
    ).with_columns(
        (pl.col("violations") / pl.col("n_evaluated")).round(4).alias("rate")
    )
    consistency_checks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quality triage

    Every evidence figure below is computed in this notebook: null rates and
    distinct counts in the per-column profile, min/max bounds in the `describe`
    table, duplicate/grain counts in the `grain_checks` cell, the null burden
    and zeros/negatives sweep and the `consistency_checks` table directly
    above, and the by-target null-rate comparison in the chart further down.

    | Issue | Evidence (10% sample) | Disposition | Reason |
    |---|---|---|---|
    | Pervasive missingness | every feature 4–20% null (per-column profile); 42,217 rows = 61.1% have ≥1 null (null-burden cell) ; worst: `social_media_hours` 19.6%, `gaming_hours` 18.4%, `weekend_screen_time` 16.4% | **caveats** | too common to drop rows; imputation strategy becomes a modeling decision. Null rates are near-identical across target labels (max gap <0.5 pp — by-target chart below), so listwise deletion is not obviously biased w.r.t. the target — but the mechanism is unverified |
    | Time budget violations | 953 rows have screen + work + sleep > 24 h (`consistency_checks` rows 1–2: count is identical whether nulls are zero-filled or only complete cases are evaluated, so all violators are complete cases — 1.8% of 52,011; the rate among incomplete rows is unknowable) | **caveats** | synthetic-data artifact; flags that columns are generated quasi-independently — engineered "hours" ratios can produce impossible values |
    | Implied negative weekday screen time | 335 of 51,714 complete daily/weekend pairs (0.6%) where (7·daily − 2·weekend)/5 < 0 (`consistency_checks` row 3) | **caveats** | same artifact, already noted in `eda.py`; avoid the weekday-screen-time derivation or clip it |
    | Truncated/bounded ranges | age caps at exactly 18–35; sleep 4.5–9.0; notifications ≤250; app opens ≤180 | **negligible** for modeling | generator clipping; tree models are indifferent, but extrapolation beyond bounds is meaningless |
    | Internal consistency holds | social + gaming never exceeds daily screen time (0 of 45,160 complete triples — `consistency_checks` row 4) | **negligible** | one consistency rule the generator did respect |
    | `gaming_hours` zeros | 16 exact zeros, no negatives anywhere (zeros/negatives sweep) | **negligible** | plausible non-gamers |
    | No duplicate rows, unique key | 0 exact dups excl. `id` | **negligible** | grain is clean |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discovery loop

    **Stop condition (set before looping):** exhaustion-based — stop when the queue
    holds no question whose answer would change a modeling decision or follow-up.
    Queue seeded from profiling: target balance → feature–target association →
    missingness mechanism → hours consistency → categorical effects. All five were
    examined; the loop stopped there. Everything below is **exploratory**.
    """)
    return


@app.cell
def _(alt, eda, pl):
    target_counts = (
        eda.group_by("addicted_label")
        .len()
        .with_columns((pl.col("len") / eda.height).alias("share"))
    )
    chart_target = (
        alt.Chart(target_counts)
        .mark_bar()
        .encode(
            x=alt.X("addicted_label:N", title="addicted_label"),
            y=alt.Y("len:Q", title="rows"),
            color=alt.Color("addicted_label:N", legend=None),
            tooltip=["addicted_label", "len", alt.Tooltip("share", format=".1%")],
        )
        .properties(
            width=250,
            height=250,
            title="Target balance — 70.9% addicted (n=69,136)",
        )
    )
    chart_target
    return


@app.cell
def _(alt, cs, eda):
    feature_cols = cs.expand_selector(eda, cs.numeric().exclude("id", "addicted_label"))
    long = (
        eda.select([*feature_cols, "addicted_label"])
        .unpivot(index="addicted_label", on=list(feature_cols))
        .drop_nulls("value")
    )
    chart_hist = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=40), title=None),
            y=alt.Y("count()", title="rows"),
        )
        .properties(width=240, height=160)
        .facet(facet="variable:N", columns=3)
        .resolve_scale(x="independent", y="independent")
        .properties(
            title="Distributions, non-null values (raw first look — no transforms)"
        )
    )
    chart_hist
    return (long,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shapes are unimodal and roughly symmetric with generator-clipped tails —
    no bimodality or round-number spikes at this bin width (checked at 20/40/80
    bins; same story). The integer-valued columns (`age`,
    `notifications_per_day`, `app_opens_per_day`) show comb artifacts at fine
    bin widths, as expected for counts.
    """)
    return


@app.cell
def _(alt, long, pl):
    top_feats = [
        "daily_screen_time_hours",
        "weekend_screen_time",
        "social_media_hours",
    ]
    chart_overlay = (
        alt.Chart(long.filter(pl.col("variable").is_in(top_feats)))
        .mark_bar(opacity=0.55)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=50), title="hours"),
            y=alt.Y("count()", stack=None, title="rows"),
            color=alt.Color("addicted_label:N"),
        )
        .properties(width=260, height=180)
        .facet(facet="variable:N", columns=3)
        .resolve_scale(x="independent", y="independent")
        .properties(title="Screen-time family by target — separated but overlapping")
    )
    chart_overlay
    return


@app.cell
def _(SEED, alt, eda, pl):
    scatter_df = eda.select(
        "daily_screen_time_hours", "weekend_screen_time", "addicted_label"
    ).drop_nulls()
    chart_scatter = (
        alt.Chart(scatter_df.sample(3000, seed=SEED))
        .mark_circle(size=12, opacity=0.4)
        .encode(
            x=alt.X("daily_screen_time_hours:Q"),
            y=alt.Y("weekend_screen_time:Q"),
            color=alt.Color("addicted_label:N"),
        )
        .properties(
            width=380,
            height=380,
            title=(
                "Daily vs weekend screen time (3,000-row subsample of complete cases)"
            ),
        )
    )
    weekend_lower = scatter_df.filter(
        pl.col("weekend_screen_time") < pl.col("daily_screen_time_hours")
    ).height
    print(
        f"weekend < daily in {weekend_lower}/{scatter_df.height} complete cases "
        f"({weekend_lower / scatter_df.height:.1%})"
    )
    chart_scatter
    return


@app.cell
def _(cs, eda, pl):
    corr_target = pl.DataFrame(
        [
            {
                "feature": c,
                "r": eda.select(pl.corr(c, "addicted_label")).item(),
                "n_complete": eda.select(pl.col(c).is_not_null().sum()).item(),
            }
            for c in cs.expand_selector(
                eda, cs.numeric().exclude("id", "addicted_label")
            )
        ]
    ).sort("r", descending=True)
    corr_target
    return (corr_target,)


@app.cell
def _(alt, corr_target):
    chart_corr = (
        alt.Chart(corr_target)
        .mark_bar()
        .encode(
            x=alt.X("r:Q", title="Pearson r with addicted_label"),
            y=alt.Y("feature:N", sort="-x"),
            tooltip=["feature", alt.Tooltip("r", format=".3f"), "n_complete"],
        )
        .properties(
            width=500,
            height=250,
            title="Feature–target correlation (pairwise-complete; pattern, "
            "not mechanism)",
        )
    )
    chart_corr
    return


@app.cell
def _(alt, cs, eda):
    null_by_target = (
        eda.group_by("addicted_label")
        .agg(cs.exclude("id", "addicted_label").is_null().mean())
        .unpivot(index="addicted_label", variable_name="feature", value_name="rate")
    )
    chart_nulls = (
        alt.Chart(null_by_target)
        .mark_bar()
        .encode(
            x=alt.X("feature:N", title=None),
            xOffset="addicted_label:N",
            y=alt.Y("rate:Q", title="null rate", axis=alt.Axis(format="%")),
            color=alt.Color("addicted_label:N"),
            tooltip=["feature", "addicted_label", alt.Tooltip("rate", format=".2%")],
        )
        .properties(
            width=700,
            height=250,
            title="Null rate by target — no visible dependence "
            "(see eda.py for bootstrap CIs)",
        )
    )
    chart_nulls
    return


@app.cell
def _(eda, pl):
    cat_rates = pl.concat(
        [
            eda.group_by(c)
            .agg(pl.len().alias("n"), pl.col("addicted_label").mean().alias("rate"))
            .with_columns(
                pl.lit(c).alias("feature"),
                pl.col(c).cast(pl.String).fill_null("(null)").alias("level"),
            )
            .select("feature", "level", "n", pl.col("rate").round(4))
            for c in ["gender", "stress_level", "academic_work_impact"]
        ]
    )
    cat_rates
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Findings — all **exploratory** (observed in this 10% sample)

    Ordered by confidence:

    1. **The target is imbalanced: 70.9% addicted.** Full-train rate 0.709424;
       stratified sample matches (0.709428). Baselines and CV must respect this
       (a constant classifier already gets 71% accuracy — use ROC-AUC/log-loss).
    2. **The screen-time family carries the signal.** In this sample,
       `daily_screen_time_hours` (r = 0.61), `weekend_screen_time` (r = 0.59),
       and `social_media_hours` (r = 0.53) are strongly associated with the
       label; group means separate cleanly (e.g. daily screen time 8.70 h
       addicted vs 5.05 h not). Distributions overlap, so no single threshold
       separates the classes.
    3. **Weak/no linear association** for `age` (r = 0.002),
       `notifications_per_day` (r = −0.01), `sleep_hours` (r = 0.04);
       `app_opens_per_day` (r = 0.07), `gaming_hours` (r = 0.21) and
       `work_study_hours` (r = 0.25) sit in between. Linear r can miss
       non-monotone structure; the overlay histograms show none obvious.
    4. **Missingness is pervasive but looks target-independent.** Every feature
       is 4–20% null; 61% of rows have at least one null. Null rates differ by
       at most ~0.5 pp between classes (consistent with `eda.py`'s bootstrap
       CIs). Mechanism (MCAR vs MAR given other features) is *not* settled here.
    5. **Categorical effects are small.** Gender levels are near-uniform
       (~1/3 each); Male 72.7% addicted vs ~70.0% for Female/Other. Stress and
       academic impact shift the rate by ≤0.5 pp. As one-hot features they will
       contribute little on their own.
    6. **Synthetic-data artifacts are real and quantified**: 953 rows (1.8% of
       complete cases) exceed a 24-hour day; 335 imply negative weekday screen
       time; both computed in the `consistency_checks` cell. Engineered
       time-budget features must be built defensively (clip or avoid).

    ### Key-number recomputation (honesty rail 3)

    | Number | polars route | independent route | Match |
    |---|---|---|---|
    | full row count | 691,369 | PowerShell line count 691,370 − header | ✅ |
    | full target rate | 0.709424 | pandas `mean()` 0.709424 | ✅ |
    | sample null rate, `social_media_hours` | 0.1956 | pandas 0.1956 | ✅ |
    | r(daily screen, target) | 0.6106 | pandas `.corr()` 0.6106 | ✅ |
    | group means daily screen (0/1) | 5.053 / 8.702 | pandas groupby 5.053 / 8.702 | ✅ |
    | >24 h rows | 953 | pandas 953 | ✅ |
    | implied weekday < 0 | 335 | pandas 335 | ✅ |

    ### Forking-paths ledger

    | # | Decision | Trigger | Alternative not taken |
    |---|---|---|---|
    | 1 | Profile a 10% stratified sample (seed 42) | analysis brief; speed | full-table profiling |
    | 2 | Pairwise-complete deletion for correlations | pervasive nulls | imputing before correlating |
    | 3 | 24-hour check run under both null policies (zero-fill and complete-cases) | nulls in the summed columns | picking one policy silently — both are shown in `consistency_checks` and happen to agree (953) |
    | 4 | Nulls kept as their own level in categorical rate tables | 4–8% null dims | dropping null rows |
    | 5 | 3,000-row subsample for the scatter | overplotting at 58k points | binned heatmap |

    ## Ranked follow-ups

    1. **Missingness mechanism test** — does any feature predict another
       feature's nullness better than chance? (`eda.py` has this running with
       logistic classifiers.) *Confirm:* held-out AUC ≈ 0.5 for all columns ⇒
       treat as MCAR, simple imputation suffices. *Kill:* any AUC well above 0.5
       ⇒ add missingness-indicator features and model-based imputation.
    2. **Baseline model on the untouched 90%** — GBM with native-null handling,
       stratified CV, ROC-AUC. *Confirms/kills* whether the screen-time family's
       sample-level separation translates to held-out ranking power, and
       promotes finding 2 beyond exploratory.
    3. **Interaction/ratio features** (screen-time-to-sleep, weekend/daily
       ratio) built with clipping per finding 6. *Confirm:* CV AUC gain over
       baseline. *Kill:* no gain ⇒ GBMs already capture the interactions.
    4. **Original vs synthetic distribution comparison** (`original.csv`, which
       also has a 4-level `addiction_level`) — per-column KS distances.
       *Confirm:* close match ⇒ original rows can augment training; the ordinal
       label may enable auxiliary tasks. *Kill:* large drift ⇒ use original data
       cautiously or not at all.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## How those follow-ups resolved

        Added after the competition closed, so the profiling pass can be read
        against what actually happened. Every number here is from the README
        ledger, and each was measured as a paired arm on identical folds.

        | # | Follow-up | Outcome |
        |---|---|---|
        | 1 | Missingness mechanism | **Killed.** Missingness indicators and null counts have univariate AUC 0.5017 — MCAR and inert, exactly the "confirm" branch. |
        | 2 | Baseline on the untouched 90% | **Confirmed.** OOF AUC 0.963947, public LB 0.96541. The screen-time family's separation did translate. |
        | 3 | Interaction / ratio features | **Confirmed, but not as expected.** Generic ratios were worth little; the *generator constraint* `daily >= social + gaming + work`, which holds for 100.000% of complete rows, was worth +0.00105. |
        | 4 | Original vs synthetic | **Killed, twice.** Appending `original.csv` rows scored −0.00004 (4/5 folds worse); an ordinal transfer feature from its 4-level label scored −0.00024 (5/5 folds worse), despite that model recovering the label at 0.9893 AUC. |

        Follow-up 4 is the sharpest instance of this project's thesis: a model
        that predicts the label almost perfectly on its own data made the
        competition model *worse*. Standalone strength does not predict
        incremental value.
        """
    )
    return


if __name__ == "__main__":
    app.run()
