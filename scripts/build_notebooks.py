#!/usr/bin/env python
# Write one training notebook per config in configs/ablation and configs/paper.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sasac.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "training"

# Stage prefix, derived from the load_from chain: stage 1 is from scratch, 0 is standalone.
STANDALONE_STAGE = 0

PANEL = {
    "study1": "Fig. 3, Standard SAC",
    "study2": "Fig. 3, Varying Network Architectures",
    "study3": "Fig. 3, Varying Input Sequence Length",
    "study4": "Fig. 3, Varying Initial SOC",
    "study5": "Fig. 3, Varying SOC and Cycle Duration",
    "study6": "Fig. 3, Varying SOC, Duration, and Power",
}
NAME = {"ffn": "FFN", "gru": "GRU", "dt": "DT"}
STANDALONE = {"ffn_ffn", "gru_gru_k100", "dt_gru_k100"}


def stage_prefix(cfg, stage):
    # ``<stage>_<index>``: read it as "run stage 1 before stage 2".
    return f"{stage}_{cfg._order}"


def parent_of(cfg):
    # Name of the run this one continues from, or None if it trains from scratch.
    return Path(cfg.load_from).name if cfg.load_from else None


def stages(configs):
    # Map run name -> stage by walking the load_from chain; raises on a cycle or a dangling parent.
    by_name = {c.name: c for c in configs}
    stage, visiting = {}, set()

    def walk(name):
        if name in stage:
            return stage[name]
        if name in visiting:
            raise ValueError(f"load_from cycle through {name!r}")
        visiting.add(name)
        cfg = by_name[name]
        parent = parent_of(cfg)
        if parent is None:
            stage[name] = STANDALONE_STAGE if cfg.name in STANDALONE else 1
        elif parent not in by_name:
            raise ValueError(f"{name} continues from {parent!r}, which has no config")
        else:
            stage[name] = walk(parent) + 1
        visiting.discard(name)
        return stage[name]

    for c in configs:
        walk(c.name)
    return stage


def header(cfg, stage, siblings):
    h = f"# {stage_prefix(cfg, stage)}  {cfg.name}\n\n{NAME[cfg.agent.actor]} actor, {NAME[cfg.agent.critic]} critic"
    if cfg.agent.actor != "ffn":
        h += f", k = {cfg.agent.context}"
    h += f". Env `{cfg.env}`. {PANEL.get(cfg.name.split('_')[0], 'Table II configuration')}."
    h += f"\n\nWrites `results/runs/{cfg.name}/`."
    parent = parent_of(cfg)
    if parent:
        h += (f"\n\n**Stage {stage}.** Continues from `{parent}` (stage {stage - 1}), so run that "
              f"one to completion first.")
    elif stage == STANDALONE_STAGE:
        h += "\n\n**Standalone.** Trains from scratch and depends on nothing; not part of Fig. 3."
    else:
        h += f"\n\n**Stage {stage}.** Trains from scratch."
    if siblings:
        label = "standalone" if stage == STANDALONE_STAGE else f"stage-{stage}"
        if len(siblings) <= 3:
            h += (f" Independent of the other {label} runs "
                  f"({', '.join(f'`{n}`' for n in siblings)}), so they can go in parallel.")
        else:
            h += (f" Independent of the other {len(siblings)} {label} runs, so the whole stage "
                  f"can go in parallel; see `notebooks/README.md`.")
    return h


def cells(cfg, rel, stage, siblings):
    load = (f"cfg = load_config('../../{rel}')\ncfg.out_dir = '../../results/runs'\n"
            + ("cfg.load_from = '../../' + cfg.load_from   # or None to train from scratch\n"
               if cfg.load_from else "")
            + "# cfg.num_episodes = 20\n"
            + "# cfg.warmup_episodes = 2\n"
            + "cfg")
    src = [
        ("markdown", header(cfg, stage, siblings)),
        ("code", "import sys\nsys.path.insert(0, '../..')\n%matplotlib inline\n"
                 "from IPython.display import display\n"
                 "from sasac.config import load_config\n"
                 "from sasac.train import build, run_training, training_summary\n"
                 "from sasac.utils.live_plots import LiveTrainingPlots"),
        ("code", load),
        ("code", "env, agent, buffer = build(cfg)\n"
                 "display(env.describe(), agent.describe(), training_summary(cfg, env))"),
        ("code", "plots = LiveTrainingPlots()\n"
                 "history = run_training(cfg, on_episode_end=plots.update, "
                 "env=env, agent=agent, buffer=buffer)\nplots.close()"),
        ("code", "import pandas as pd\n"
                 "# read the log file, not `history`, so this still works after an interrupt\n"
                 "pd.read_csv(f'{cfg.run_dir}/{cfg.name}.csv').tail(10)"),
        ("code", "from sasac.utils.probe import report\nreport(env, agent, cfg)"),
    ]
    out = []
    for kind, s in src:
        cell = {"cell_type": kind, "metadata": {}, "source": s}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        out.append(cell)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.ipynb"):
        old.unlink()
    paths = [f for d in ("configs/ablation", "configs/paper")
             for f in sorted((ROOT / d).glob("*.yaml"))]
    configs = [load_config(f) for f in paths]
    source = {c.name: f for c, f in zip(configs, paths)}
    stage = stages(configs)

    # Index within a stage, so a name reads <stage>_<index>; stages run in order, the runs inside do not.
    order, counter = {}, {}
    for c in sorted(configs, key=lambda c: (stage[c.name], c.name)):
        counter[stage[c.name]] = counter.get(stage[c.name], 0) + 1
        order[c.name] = counter[stage[c.name]]

    for cfg in sorted(configs, key=lambda c: (stage[c.name], c.name)):
        cfg._order = order[cfg.name]
        st = stage[cfg.name]
        siblings = [n for n in sorted(order) if stage[n] == st and n != cfg.name]
        nb = {"cells": cells(cfg, source[cfg.name].relative_to(ROOT).as_posix(), st, siblings),
              "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                          "name": "python3"},
                           "language_info": {"name": "python"}}}
        path = OUT / f"{stage_prefix(cfg, st)}_{cfg.name}.ipynb"
        path.write_text(json.dumps(nb, indent=1))
        print(f"stage {st}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
