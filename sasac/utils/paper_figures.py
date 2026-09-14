# Paper figures and tables, drawn from results/.
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from plant import DriveCycle, cycle_file, fuel_economy_mpg
from .history import TrainingHistory

# MATLAB's default axes colour order, used for the Fig. 3 lanes in panel order.
MATLAB_ORDER = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
# MATLAB yyaxis colours the left and right axes with the first two order entries.
AXIS_LEFT, AXIS_RIGHT = "#0072BD", "#D95319"

# Fig. 4 series colours are the paper's, so a lane reads the same in every panel.
SERIES = {
    "DP":  dict(color="#00C000", label="DP"),
    "DNN": dict(color="#000000", label="FFN"),
    "GRU": dict(color="#0000FF", label="GRU"),
    "DT":  dict(color="#FF0000", label="DT"),
}
VELOCITY_COLOR = "#1C7C7C"
CYCLE_TITLES = {"FHDS_10": "(a) HFET", "US06_21": "(b) US06", "HHDT_5": "(c) HHDDT Cruise"}
CYCLE_LABEL = {"FHDS_10": "HFET", "US06_21": "US06", "HHDT_5": "HHDDT"}

# Fig. 3 panels: (title, [(run name, legend label)])
STUDY_PANELS = [
    ("Standard SAC", [("study1_ffn_random_1cycle", "Random Sampling 1 cycle"),
                      ("study1_ffn_random_10cycle", "Random Sampling 10 cycle"),
                      ("study1_ffn_sequential_10cycle", "Sequence Sampling 10 cycle")]),
    # Paper Study 2 says input length 1; the DT-GRU and GRU-GRU lanes here are the k=10 runs.
    ("Varying Network Architectures", [("study1_ffn_random_10cycle", "FFN-FFN"), ("study2_gru_ffn", "GRU-FFN"),
                                       ("study3_gru_gru_k10", "GRU-GRU"), ("study3_dt_gru_k10", "DT-GRU"),
                                       ("study2_dt_dt", "DT-DT")]),
    ("Varying Input Sequence Length", [("study3_gru_gru_k10", "GRU k=10"), ("study3_gru_gru_k100", "GRU k=100"),
                                       ("study3_dt_gru_k10", "DT k=10"), ("study3_dt_gru_k100", "DT k=100")]),
    ("Varying initial SOC", [("study4_ffn_ffn_vary_soc", "FFN"), ("study4_gru_gru_vary_soc", "GRU"),
                             ("study4_dt_gru_vary_soc", "DT")]),
    ("Varying SOC and Cycle Duration", [("study5_ffn_ffn_vary_soc_duration", "FFN"),
                                        ("study5_gru_gru_vary_soc_duration", "GRU"),
                                        ("study5_dt_gru_vary_soc_duration", "DT")]),
    ("Varying SOC, Duration, and Power", [("study6_ffn_ffn_vary_all", "FFN"), ("study6_gru_gru_vary_all", "GRU"),
                                          ("study6_dt_gru_vary_all", "DT")]),
]


def moving_average(x, w=10):
    x = np.asarray(x, float)
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode="valid")


def _matlab_axes(ax, grid=True):
    # MATLAB default axes: full box, inward ticks, light dotted grid, bold tick labels.
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.9)
        s.set_color("#262626")
    ax.tick_params(direction="in", length=5, width=0.9, top=True, right=True,
                   labelsize=9, colors="#262626")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    if grid:
        ax.grid(True, linestyle=":", linewidth=0.7, color="#bfbfbf")
        ax.set_axisbelow(True)


def _matlab_legend(ax, handles=None, labels=None, **kw):
    # MATLAB legends are boxed, white, thin black edge.
    kw.setdefault("framealpha", 1.0)
    kw.setdefault("fancybox", False)
    kw.setdefault("fontsize", 8)
    leg = (ax.legend(handles, labels, **kw) if handles is not None else ax.legend(**kw))
    leg.get_frame().set_edgecolor("#262626")
    leg.get_frame().set_linewidth(0.8)
    return leg


def fig3(runs_dir, window=10, out=None):
    # Fig. 3: six ablation panels, episode against 10-episode moving-average reward.
    runs_dir = Path(runs_dir)
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.6))
    for idx, (ax, (title, runs)) in enumerate(zip(axes.ravel(), STUDY_PANELS)):
        n = 0
        for i, (name, label) in enumerate(runs):
            f = runs_dir / name / f"{name}.mat"
            if not f.exists():
                continue
            h = TrainingHistory.from_mat(f)
            y = moving_average(h["rewards"], window)
            x = np.asarray(h["episode"], float)[len(h["rewards"]) - len(y):]
            # Dashes on the first two lanes of a panel, dots after, as the paper does.
            ax.plot(x, y, color=MATLAB_ORDER[i % len(MATLAB_ORDER)],
                    linestyle="--" if i < 2 else ":", linewidth=1.3, label=label)
            n += 1
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 500)
        ax.set_ylim(50, 100)
        ax.set_yticks(np.arange(50, 101, 10))
        ax.set_xticks(np.arange(0, 501, 100))
        if idx % 3 == 0:
            ax.set_ylabel("Reward", fontsize=10, fontweight="bold")
        if idx == 4:
            ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
        _matlab_axes(ax)
        if n >= 2:
            # Two columns once a panel has over three lanes or long names, to keep the box inside
            long_labels = max(len(lbl) for _, lbl in runs) > 12
            _matlab_legend(ax, loc="best", ncol=2 if (len(runs) > 3 or long_labels) else 3)
        elif n == 0:
            ax.text(250, 75, "no runs yet", ha="center", va="center", color="#777")
    return _finish(fig, out)


def _finish(fig, out):
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def _struct(path):
    # The single named struct inside a roll-out .mat.
    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return m[[k for k in m if not k.startswith("__")][0]]


def _soc_of(path):
    s = _struct(path)
    return np.asarray(s.next_state)[:, 0] if hasattr(s, "next_state") else np.asarray(s.SOC).ravel()


def _fuel_of(path):
    return np.asarray(_struct(path).m_dot_fuel, float).ravel()


def fig4(val_dir, dp_dir=None, cycles=("FHDS_10", "US06_21", "HHDT_5"), out=None, t_max=180):
    # Fig. 4: velocity and SOC against time, one stacked panel per cycle.
    val_dir, dp_dir = Path(val_dir), Path(dp_dir) if dp_dir else None
    fig, axes = plt.subplots(len(cycles), 1, figsize=(10, 3.9 * len(cycles)))
    axes = np.atleast_1d(axes)
    for ax_v, cyc in zip(axes, cycles):
        v = DriveCycle(cycle_file(cyc)).v * 2.23694  # mph
        t = np.arange(len(v)) / 60.0
        h_v, = ax_v.plot(t, v, color=VELOCITY_COLOR, lw=0.7, label="Velocity Profile")
        ax_v.set_xlim(0, t_max)
        ax_v.set_ylim(0, 80)
        ax_v.set_yticks(np.arange(0, 81, 10))
        ax_v.set_xticks(np.arange(0, t_max + 1, 20))
        ax_v.set_xlabel("Time (min)", fontsize=11, fontweight="bold")
        ax_v.set_ylabel("Velocity (MPH)", fontsize=11, fontweight="bold", color=AXIS_LEFT)
        ax_v.tick_params(axis="y", colors=AXIS_LEFT)
        ax_v.set_title(CYCLE_TITLES.get(cyc, cyc), fontsize=11, fontweight="bold", loc="left")

        ax_s = ax_v.twinx()
        series = {}
        if dp_dir and (dp_dir / f"DP_{cyc}.mat").exists():
            series["DP"] = np.asarray(loadmat(str(dp_dir / f"DP_{cyc}.mat"), squeeze_me=True)["SOC"], float).ravel()
        for tag in ("DNN", "GRU", "DT"):
            f = val_dir / f"SAC_{tag}_{cyc}_test.mat"
            if f.exists():
                series[tag] = _soc_of(f)
        handles = [h_v]
        for tag, soc in series.items():
            st = SERIES[tag]
            h, = ax_s.plot(np.arange(len(soc)) / 60.0, 100 * soc, color=st["color"],
                           linestyle=(0, (4, 2)), lw=4.0, solid_capstyle="butt", label=st["label"])
            handles.append(h)
        ax_s.set_ylim(0, 100)
        ax_s.set_yticks(np.arange(0, 101, 20))
        ax_s.set_ylabel("SOC (%)", fontsize=11, fontweight="bold", color=AXIS_RIGHT)
        ax_s.tick_params(axis="y", colors=AXIS_RIGHT)

        _matlab_axes(ax_v)
        _matlab_axes(ax_s, grid=False)
        for lab in ax_v.get_yticklabels():
            lab.set_color(AXIS_LEFT)
        for lab in ax_s.get_yticklabels():
            lab.set_color(AXIS_RIGHT)
        _matlab_legend(ax_s, handles, [h.get_label() for h in handles],
                       loc="upper center", ncol=len(handles), fontsize=9)
    return _finish(fig, out)


def _row(tag, cyc, soc, mf, v):
    n = min(len(v), len(mf))
    return dict(agent=tag, cycle=CYCLE_LABEL.get(cyc, cyc), soc_final=100 * soc[-1],
                mpg=fuel_economy_mpg(mf[:n], v[:n]))


def validation_table(val_dir, dp_dir=None, tags=("DNN", "GRU", "DT")):
    # Final SOC [%] and MPG for every SAC_<tag>_<cycle>_test.mat (plus DP rows if exported).
    import pandas as pd

    val_dir, dp_dir = Path(val_dir), Path(dp_dir) if dp_dir else None
    rows = []
    for f in sorted(val_dir.glob("SAC_*_test.mat")):
        m = re.match(r"SAC_(.+?)_(FHDS_10|US06_21|HHDT_5|.+)_test\.mat", f.name)
        if not m:
            continue
        tag, cyc = m.group(1), m.group(2)
        try:
            v = DriveCycle(cycle_file(cyc)).v
        except FileNotFoundError:
            continue
        rows.append(_row(tag, cyc, _soc_of(f), _fuel_of(f), v))
    if dp_dir:
        for cyc in ("FHDS_10", "US06_21", "HHDT_5"):
            f = dp_dir / f"DP_{cyc}.mat"
            if f.exists():
                d = loadmat(str(f), squeeze_me=True)
                rows.append(_row("DP", cyc, np.asarray(d["SOC"], float).ravel(),
                                 np.asarray(d["m_dot_fuel"], float).ravel(),
                                 DriveCycle(cycle_file(cyc)).v))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["agent"] = df["agent"].replace({"DNN": "FFN"})
    table = df.pivot(index="agent", columns="cycle", values=["soc_final", "mpg"]).round(2)
    order = [a for a in ("DP", "FFN", "GRU", "DT") if a in table.index] + [a for a in table.index if a not in ("DP", "FFN", "GRU", "DT")]
    return table.loc[order]
