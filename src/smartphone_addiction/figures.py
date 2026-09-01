"""Export the experiment ledger as figures.

EVERY NUMBER HERE IS TRANSCRIBED FROM THE README's ledger, not recomputed. These are
the results of runs that happened; the competition data is not committed, so this module
deliberately does no modelling. It reads as a table of measured facts and draws them.
The one thing it does compute is the CV->LB offset, as a subtraction of two ledger
columns, so the figure cannot disagree with the numbers it is drawn from.

Run:  pixi run figures     ->  docs/figures/*.svg, *.png, and specs/*.json
"""

from __future__ import annotations

import json
from typing import Literal

import altair as alt
import polars as pl

from smartphone_addiction.paths import FIGURES_DIR

# Categorical slots 1 and 2 of the validated default palette, light mode. The pair
# clears every gate: worst adjacent CVD dE 24.7 (protan), normal-vision dE 33.6.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#c8c7c2"
FONT = "Ubuntu Sans, DejaVu Sans, Helvetica, Arial, sans-serif"

LabelAlign = Literal["left", "center", "right"]

# --- the ledger -----------------------------------------------------------------
SUBMISSIONS = [
    # (short label, module or description, OOF CV AUC, public LB AUC, rank)
    ("baseline", "raw features", 0.963947, 0.96541, 1761),
    ("pipeline", "representation", 0.968069, 0.96947, 904),
    ("stack-7", "7 members", 0.968630, 0.96982, 821),
    ("stack-14", "+ XGBoost", 0.968738, 0.96994, 777),
    ("stack-18", "+ ES-reclaim, NN", 0.968955, 0.97008, 740),
    ("stack-26", "+ feature subsets", 0.969010, 0.97013, 728),
]

GAINS = [
    # (label, delta) -- the measured decomposition, each a paired same-fold arm
    ("max_bin 255 -> 2047", 0.00215),
    ("value-level encoding", 0.00122),
    ("generator structure", 0.00105),
    ("imputation as augmentation", 0.00031),
    ("hill-climb rank stack", 0.00024),
]

REJECTED = [
    # (label, signed delta) -- everything measured on identical folds and not kept
    ("lower learning rate 0.03 -> 0.015", 0.00019),
    ("LGB + XGB blend on raw features", 0.00011),
    ("neighbourhood-shrunk target encoding", 0.00004),
    ("capacity re-tune, 127 leaves", 0.00001),
    ("init_score from TE logit, top-2 cols", 0.00001),
    ("appending original.csv rows", -0.00004),
    ("DART boosting", -0.00020),
    ("ordinal transfer from original.csv", -0.00024),
    ("init_score from TE logit, all 9 cols", -0.00027),
    ("LightGBM extra_trees", -0.00044),
    ("LightGBM linear_tree", -0.00081),
]
RESOLUTION_FLOOR = 0.00014  # measured paired sd, stack.py

DIVERSITY = [
    # (source, combined stack weight) -- what earned weight among 29 candidates.
    # Single-line labels: a newline inside an axis label is drawn as one overlapping
    # text mark, not as two lines.
    ("Feature subsets \u2014 struct_imp, no_te", 0.133),
    ("Different model class \u2014 embedding NN", 0.100),
    (
        "Hyperparameter variants \u2014 linear_tree, extra_trees, DART, ExtraTrees",
        0.000,
    ),
]


def _base() -> alt.Chart:
    """Shared chrome: hairline recessive grid, no domain rules, one sans face."""
    return alt.Chart()


def _cfg(chart: alt.LayerChart | alt.Chart | alt.VConcatChart) -> alt.TopLevelMixin:
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            grid=True,
            gridColor=GRID,
            gridWidth=1,
            domain=False,
            tickColor=GRID,
            labelColor=MUTED,
            titleColor=MUTED,
            labelFont=FONT,
            titleFont=FONT,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=MUTED,
            titleColor=MUTED,
            labelFont=FONT,
            titleFont=FONT,
            labelFontSize=11,
            titleFontSize=11,
            symbolStrokeWidth=0,
            symbolType="square",
        )
        .configure_title(
            color=INK, font=FONT, fontSize=13, fontWeight=600, anchor="start", dy=-6
        )
        .configure_text(font=FONT)
    )


def fig_cv_lb() -> tuple[alt.TopLevelMixin, pl.DataFrame]:
    """CV and LB per submission, with the offset between them as its own panel.

    Two measures on the same AUC scale, so one axis serves both -- never a second
    y-scale. The offset gets its own panel rather than an annotation, because the
    offset's *stability* is the claim being made.
    """
    rows = []
    for label, note, cv, lb, rank in SUBMISSIONS:
        rows.append(
            {
                "submission": label,
                "note": note,
                "kind": "CV (OOF)",
                "auc": cv,
                "rank": rank,
            }
        )
        rows.append(
            {
                "submission": label,
                "note": note,
                "kind": "Public LB",
                "auc": lb,
                "rank": rank,
            }
        )
    df = pl.DataFrame(rows)
    order = [s[0] for s in SUBMISSIONS]

    offs = pl.DataFrame(
        [
            {"submission": s[0], "offset": round(s[3] - s[2], 6), "rank": s[4]}
            for s in SUBMISSIONS
        ]
    )

    x = alt.X("submission:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
    top = (
        alt.Chart(df.to_pandas())
        .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("auc:Q", title="ROC-AUC", scale=alt.Scale(zero=False, nice=True)),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(domain=["CV (OOF)", "Public LB"], range=[BLUE, ORANGE]),
                # Bottom-right is the only quadrant both series leave empty: they rise
                # left-to-right, so a top-left legend sits on top of the LB line.
                legend=alt.Legend(
                    orient="bottom-right",
                    direction="horizontal",
                    fillColor="#fcfcfb",
                    padding=6,
                ),
            ),
            tooltip=[
                "submission",
                "note",
                "kind",
                alt.Tooltip("auc:Q", format=".6f"),
                "rank",
            ],
        )
        .properties(
            width=520,
            height=190,
            title="CV tracks the leaderboard, submission after submission",
        )
    )

    off_scale = alt.Scale(domain=[0.0010, 0.0016], zero=False, nice=False)
    mean_off = float(offs["offset"].to_numpy().mean())
    mean_df = pl.DataFrame(
        [{"m": mean_off, "label": f"mean {mean_off:+.5f}"}]
    ).to_pandas()
    band = (
        alt.Chart(mean_df)
        .mark_rule(color=NEUTRAL, strokeWidth=1)
        .encode(y=alt.Y("m:Q", title="LB - CV", scale=off_scale))
    )
    band_label = (
        alt.Chart(mean_df)
        .mark_text(align="left", baseline="bottom", dy=-3, color=MUTED, fontSize=10)
        .encode(y=alt.Y("m:Q", scale=off_scale), text="label:N", x=alt.value(3))
    )
    line = (
        alt.Chart(offs.to_pandas())
        .mark_line(
            point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2, color=BLUE
        )
        .encode(
            x=x,
            y=alt.Y(
                "offset:Q",
                title="LB - CV",
                scale=off_scale,
                axis=alt.Axis(format=".4f"),
            ),
            tooltip=["submission", alt.Tooltip("offset:Q", format=".6f")],
        )
    )
    bottom = (band + band_label + line).properties(
        width=520,
        height=120,
        title="The gap never inverts, but it is not constant: it narrows",
    )
    return _cfg(alt.vconcat(top, bottom).resolve_scale(x="shared")), offs


def fig_diversity() -> alt.TopLevelMixin:
    """Stack weight by diversity source.

    One series, so no legend box: the title names what is being measured.
    """
    df = pl.DataFrame([{"source": s, "weight": w} for s, w in DIVERSITY])
    order = [s for s, _ in DIVERSITY]
    base = alt.Chart(df.to_pandas()).encode(
        y=alt.Y(
            "source:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=400, labelFontSize=11),
        ),
        x=alt.X(
            "weight:Q",
            title="Combined weight in the 26-member stack",
            scale=alt.Scale(domain=[0, 0.16]),
        ),
    )
    bars = base.mark_bar(height=18, cornerRadiusEnd=4, color=BLUE).encode(
        tooltip=["source", alt.Tooltip("weight:Q", format=".3f")]
    )
    # Few enough marks that a label on each is selective, not chaos.
    labels = base.mark_text(align="left", dx=6, color=MUTED, fontSize=11).encode(
        text=alt.Text("weight:Q", format=".3f")
    )
    return _cfg(
        (bars + labels).properties(
            width=430,
            height=alt.Step(46),
            title="Diversity has to come from what the model sees, not how it fits",
        )
    )


def fig_gain() -> alt.TopLevelMixin:
    """Waterfall from the raw-feature baseline to the final stack.

    The two totals are drawn as reference rules, not bars. A bar means a magnitude
    measured from zero, and this axis cannot start at zero -- the whole story is
    0.005 wide on a 0.97 scale. Floating bars are legitimate here because each one
    encodes a RANGE (a step from one running total to the next); a truncated bar
    claiming to be a total would not be.
    """
    base_cv = SUBMISSIONS[0][2]
    final_cv = SUBMISSIONS[-1][2]
    floor = 0.9635
    rows, running = [], base_cv
    for label, d in GAINS:
        rows.append({"step": label, "start": running, "end": running + d, "delta": d})
        running += d
    resid = final_cv - running
    rows.append(
        {
            "step": "unattributed residual",
            "start": running,
            "end": final_cv,
            "delta": resid,
        }
    )
    df = pl.DataFrame(rows)
    order = [r["step"] for r in rows]
    y_scale = alt.Scale(domain=[floor, 0.9695], nice=False)

    base = alt.Chart(df.to_pandas()).encode(
        x=alt.X(
            "step:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelAngle=-30, labelLimit=150),
        )
    )
    bars = base.mark_bar(size=30, cornerRadius=3, color=BLUE).encode(
        y=alt.Y("start:Q", title="OOF AUC", scale=y_scale, axis=alt.Axis(format=".4f")),
        y2="end:Q",
        tooltip=[
            "step",
            alt.Tooltip("delta:Q", format="+.5f"),
            alt.Tooltip("end:Q", format=".6f"),
        ],
    )
    labels = base.mark_text(
        align="center", baseline="bottom", dy=-5, color=MUTED, fontSize=10
    ).encode(y=alt.Y("end:Q", scale=y_scale), text=alt.Text("delta:Q", format="+.5f"))

    marks = pl.DataFrame(
        [
            {"level": base_cv, "label": f"raw-feature baseline  {base_cv:.6f}"},
            {"level": final_cv, "label": f"26-member stack  {final_cv:.6f}"},
        ]
    ).to_pandas()
    # The first bar rises off the baseline on the left, so that label is anchored right;
    # the top of the plot is clear on the left, so that label is anchored left.
    base_mark = marks.iloc[[0]]
    final_mark = marks.iloc[[1]]
    rules = (
        alt.Chart(marks)
        .mark_rule(color=NEUTRAL, strokeWidth=1)
        .encode(y=alt.Y("level:Q", scale=y_scale))
    )

    def _rule_label(frame, align: LabelAlign, px: int) -> alt.Chart:
        return (
            alt.Chart(frame)
            .mark_text(baseline="bottom", dy=-3, align=align, color=MUTED, fontSize=10)
            .encode(y=alt.Y("level:Q", scale=y_scale), text="label:N", x=alt.value(px))
        )

    rule_labels = _rule_label(base_mark, "right", 515) + _rule_label(
        final_mark, "left", 3
    )
    return _cfg(
        (rules + rule_labels + bars + labels).properties(
            width=520,
            height=280,
            title="Where the +0.0051 came from: representation, not tuning",
        )
    )


def fig_rejected() -> alt.TopLevelMixin:
    """Signed effects of everything measured and not kept, against the resolution floor.

    Blue helped, red hurt, and the grey band is the noise floor. Most of the list sits
    inside it -- which is the finding, not a rendering accident.
    """
    df = pl.DataFrame([{"idea": i, "delta": d} for i, d in REJECTED])
    order = [i for i, _ in REJECTED]
    x_scale = alt.Scale(domain=[-0.00095, 0.00030], nice=False)
    x_axis = alt.Axis(
        format="+.5f",
        values=[-0.00075, -0.00050, -0.00025, 0.0, 0.00025],
        title="Change in OOF AUC, paired on identical folds",
    )
    base = alt.Chart(df.to_pandas()).encode(
        y=alt.Y(
            "idea:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=300, labelFontSize=11),
        )
    )
    band = (
        alt.Chart(
            pl.DataFrame(
                [{"lo": -RESOLUTION_FLOOR, "hi": RESOLUTION_FLOOR}]
            ).to_pandas()
        )
        .mark_rect(color=NEUTRAL, opacity=0.30)
        .encode(x=alt.X("lo:Q", scale=x_scale, axis=x_axis), x2="hi:Q")
    )
    bars = base.mark_bar(height=13, cornerRadiusEnd=3).encode(
        x=alt.X("delta:Q", scale=x_scale, axis=x_axis),
        color=alt.condition(alt.datum.delta >= 0, alt.value(BLUE), alt.value(RED)),
        tooltip=["idea", alt.Tooltip("delta:Q", format="+.5f")],
    )
    # Without an explicit zero rule the reader cannot locate the origin on a
    # signed axis whose ticks are mostly negative.
    zero = (
        alt.Chart(pl.DataFrame([{"z": 0.0}]).to_pandas())
        .mark_rule(color=MUTED, strokeWidth=1)
        .encode(x=alt.X("z:Q", scale=x_scale, axis=x_axis))
    )
    return _cfg(
        (band + zero + bars).properties(
            width=430,
            height=alt.Step(22),
            title=(
                "Measured and rejected. Grey band = the "
                f"+/-{RESOLUTION_FLOOR} resolution floor"
            ),
        )
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    specs = FIGURES_DIR / "specs"
    specs.mkdir(exist_ok=True)

    cv_lb, offs = fig_cv_lb()
    charts = {
        "cv-lb-offset": cv_lb,
        "stack-weight-by-diversity": fig_diversity(),
        "gain-decomposition": fig_gain(),
        "measured-and-rejected": fig_rejected(),
    }
    for name, chart in charts.items():
        chart.save(str(FIGURES_DIR / f"fig-{name}.svg"))
        chart.save(str(FIGURES_DIR / f"fig-{name}.png"), ppi=200)
        (specs / f"{name}.json").write_text(json.dumps(chart.to_dict(), indent=1))
        print(f"wrote fig-{name}.svg / .png / specs/{name}.json")

    print("\nCV -> LB offset, computed from the ledger (not the README's summary):")
    for row in offs.iter_rows(named=True):
        print(f"  {row['submission']:10s} {row['offset']:+.6f}  (rank {row['rank']})")
    o = offs["offset"]
    print(
        f"  mean {o.mean():+.6f}   sd {o.std():.6f}   "
        f"range {o.min():+.6f} .. {o.max():+.6f}"
    )


if __name__ == "__main__":
    main()
