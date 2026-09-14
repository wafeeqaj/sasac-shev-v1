# Notebooks

Filenames carry the run order: `<stage>_<index>_<name>.ipynb`. Read the stage as "finish
every stage 1 before starting stage 2". The index only keeps the listing stable; runs inside
a stage are independent of each other.

```
0_*   three standalone Table II configurations, depend on nothing
1_*   nine runs from scratch          (studies 1-3)
2_*   varying initial SOC             (study 4)
3_*   and cycle duration              (study 5)
4_*   and EM power                    (study 6)  -> the Table IV agents
5_    evaluate_agents
6_    paper_figures
```

Stages 2-4 are continued training (paper Sec. IV-A): each run loads the checkpoint its
parent wrote, so the order is not optional. A notebook whose parent has not run stops and
names the one to run first; `load_from: null` trains it from scratch instead.

## Running in parallel

Every run inside a stage is independent, so on several GPUs launch the whole stage at once
and wait for it before starting the next:

```
CUDA_VISIBLE_DEVICES=0 jupyter nbconvert --execute notebooks/training/1_1_*.ipynb &
CUDA_VISIBLE_DEVICES=1 jupyter nbconvert --execute notebooks/training/1_2_*.ipynb &
wait                       # stage 1 complete; stage 2 can now start
```

Budget roughly 5 GPU-hours per run. Sequentially the 18 Fig. 3 runs are about 90 GPU-hours;
three GPUs, one per lane, brings the critical path down to about 20 hours.

## Stage 1 (from scratch, all nine in parallel)

| notebook | env |
|---|---|
| `1_1_study1_ffn_random_10cycle` | `fixed_hfet_10` |
| `1_2_study1_ffn_random_1cycle` | `fixed_hfet_1` |
| `1_3_study1_ffn_sequential_10cycle` | `fixed_hfet_10` |
| `1_4_study2_dt_dt` | `fixed_hfet_10` |
| `1_5_study2_gru_ffn` | `fixed_hfet_10` |
| `1_6_study3_dt_gru_k10` | `fixed_hfet_10` |
| `1_7_study3_dt_gru_k100` | `fixed_hfet_10` |
| `1_8_study3_gru_gru_k10` | `fixed_hfet_10` |
| `1_9_study3_gru_gru_k100` | `fixed_hfet_10` |

## Stages 2-4 (continued training)

Three lanes run down the table independently. A lane's stage *n* waits only on its own
stage *n-1*, so FFN, GRU and DT can proceed in parallel throughout.

| stage | env | FFN lane | GRU lane | DT lane |
|---|---|---|---|---|
| 2 | `vary_soc` | `2_2_study4_ffn_ffn_vary_soc` | `2_3_study4_gru_gru_vary_soc` | `2_1_study4_dt_gru_vary_soc` |
| 3 | `vary_soc_duration` | `3_2_study5_ffn_ffn_vary_soc_duration` | `3_3_study5_gru_gru_vary_soc_duration` | `3_1_study5_dt_gru_vary_soc_duration` |
| 4 | `vary_all` | `4_2_study6_ffn_ffn_vary_all` | `4_3_study6_gru_gru_vary_all` | `4_1_study6_dt_gru_vary_all` |

Each lane's stage 2 continues from its stage-1 run: FFN from `study1_ffn_random_10cycle`,
GRU from `study3_gru_gru_k100`, DT from `study3_dt_gru_k100`.

## Standalone

`0_1_dt_gru_k100`, `0_2_ffn_ffn` and `0_3_gru_gru_k100` are the three Table II
configurations. They train from scratch, depend on nothing, and are not part of Fig. 3.

## Then

| notebook | what it does |
|---|---|
| `5_evaluate_agents.ipynb` | rolls trained agents through the plant -> `results/validation/` |
| `6_paper_figures.ipynb` | Fig. 3, Fig. 4, Tables III/IV from `results/` |
