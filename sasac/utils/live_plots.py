# Live-updating training dashboard (six figures, redrawn per episode).
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from .history import TrainingHistory

PANELS = [
    # (figure index, key, ylabel, title)
    (0, "rewards", "Reward", "Episode vs Reward"),
    (0, "avg_alpha", "Average Alpha", "Episode vs Alpha"),
    (1, "avg_speeds", "Average Speed", "Episode vs Average Speed"),
    (1, "avg_torques", "Average Torque", "Episode vs Average Torque"),
    (1, "socs", "SOC", "Episode vs Final SOC"),
    (1, "initial_soc_list", "Initial SOC", "Episode vs Initial SOC"),
    (2, "avg_fuel_list", "Fuel", "Episode vs Fuel Consumption"),
    (2, "pow_demand_list", "Power Demand", "Episode vs Power Demand Multiply"),
    (3, "time_list", "Time (min)", "Episode vs Time"),
    (3, "q_action_sens", "|dQ| on a wrong action", "Episode vs Critic Action Sensitivity"),
    (4, "torq_switch_counts", "Torque Fluctuation Count", "Episode vs Torque Switches"),
    (4, "speed_switch_counts", "Speed Fluctuation Count", "Episode vs Speed Switches"),
    (5, "q1_loss_list", "Q1_loss", "Episode vs Q1_loss"),
    (5, "q2_loss_list", "Q2_loss", "Episode vs Q2_loss"),
    (5, "actor_loss_list", "Actor Loss", "Episode vs Actor Loss"),
    (5, "alpha_loss_list", "Alpha Loss", "Episode vs alpha loss"),
]
FIG_LAYOUTS = [((1, 2), (15, 4)), ((2, 2), (15, 10)), ((1, 2), (15, 4)), ((1, 2), (15, 4)), ((1, 2), (15, 4)), ((2, 2), (15, 10))]


def headline(stats: dict) -> str:
    # The six numbers worth watching, ahead of the full dump.
    q = stats.get("actor_loss_list", float("nan"))
    return (f"ep {stats['episode']:>3} | reward {stats['rewards']:6.2f} | "
            f"soc {stats['socs']:.3f} | alpha {stats['avg_alpha']:.4f} | "
            f"Q {abs(q):6.2f} | q_action_sens {stats.get('q_action_sens', float('nan')):.5f}")


class _Dashboard:
    def __init__(self):
        self.figs, self.lines, self.axes = [], {}, {}
        counters = {}
        for i, ((r, c), size) in enumerate(FIG_LAYOUTS):
            fig, axs = plt.subplots(r, c, figsize=size)
            self.figs.append(fig)
            flat = list(axs.ravel()) if hasattr(axs, "ravel") else [axs]
            counters[i] = iter(flat)
        for fig_i, key, ylabel, title in PANELS:
            ax = next(counters[fig_i])
            (line,) = ax.plot([], [])
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            self.lines[key], self.axes[key] = line, ax
        for fig in self.figs:
            fig.tight_layout()

    def redraw(self, history: TrainingHistory) -> None:
        x = history["episode"]
        for key, line in self.lines.items():
            line.set_data(x, history[key])
            self.axes[key].relim()
            self.axes[key].autoscale_view()


class LiveTrainingPlots(_Dashboard):
    # Interactive dashboard for Jupyter (calls ``clear_output`` + ``display``).

    def __init__(self, pause: float = 0.01):
        plt.ion()
        super().__init__()
        self.pause = pause
        try:
            from IPython.display import clear_output, display

            self._clear, self._display = clear_output, display
        except ImportError:  # pragma: no cover
            self._clear = self._display = None

    def update(self, history: TrainingHistory, episode_stats: Optional[dict] = None) -> None:
        self.redraw(history)
        if self._clear is not None:
            self._clear(wait=True)
            for fig in self.figs:
                self._display(fig)
        # after the figures, so the inline backend cannot swallow it
        if episode_stats:
            print(headline(episode_stats), flush=True)
            print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in episode_stats.items()), flush=True)
        plt.pause(self.pause)

    def close(self) -> None:
        plt.ioff()
        for fig in self.figs:
            plt.close(fig)

