# TELA

**Unified Temporal-Exogenous-Lag Attention for Ultra-Short-Term Wind Power Forecasting**

Standalone implementation accompanying the paper accepted by IEEE ICCBD+AI 2026.
Historical variable-time patch tokens and context-conditioned horizon queries share
a relation-biased self-attention space.

[Data Preparation](datasets/README.md) | [Data Manifest](datasets/manifest.json) | [Citation](CITATION.bib) | [MIT License](LICENSE)

## Setup

Tested with Python 3.12.9, PyTorch 2.8.0, NumPy 2.1.2, and pandas 3.0.3.
The research environment used a CUDA 12.9 PyTorch build; CPU smoke tests are supported.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

Choose a PyTorch build compatible with your GPU driver for GPU training.
The smoke test uses generated synthetic inputs solely for software verification.

## Data

Obtain the State Grid competition data from its original provider:
[Chen and Xu, Scientific Data (2022)](https://doi.org/10.1038/s41597-022-01696-6).
Prepare the following wide-table CSVs under `datasets/wind/`:

```text
grid_wind_site1_99MW.csv
grid_wind_site2_200MW.csv
grid_wind_site4_66MW.csv
```

Each CSV contains a `date` column, historical numeric covariates, and a `power`
target in MW. The time range is 2019-01-01 through 2020-12-31 at 15-minute intervals,
with 70,176 rows per site. Exact column order and prepared-input SHA-256 fingerprints
are recorded in `datasets/manifest.json`. Site2 has no humidity column.
Raw-provider file formatting and prepared-input fingerprints are distinct.

```bash
python scripts/verify_data.py --data-root datasets/wind
```

Training uses chronological 70/10/20 splits. Normalization statistics are fitted
on the training partition. Wind-direction columns are encoded as sine/cosine pairs.
Validation/test history can extend before a partition boundary; forecast targets
remain within their respective partitions.

## Reproduction

```bash
python scripts/reproduce.py --sites site1 --seeds 2026
python scripts/reproduce.py --sites site1 site2 site4 --seeds 2026 2027 2028
python scripts/reproduce.py --sites site1 --seeds 2026 --dry-run
python scripts/reproduce.py --sites site1 --seeds 2026 --evaluate
```

Use `--data-root PATH`, `--gpu INDEX`, or `--cpu` as needed. Existing experiment
outputs are protected by the reproduction wrapper; `--overwrite` enables replacement.
Evaluation writes to `result/evaluation/`, preserving training JSON metadata.
Repository-relative data/checkpoint/result paths are anchored to the repository root.

`configs/conference.json` fixes the paper protocol: 96 input steps, 16 forecast
steps, patch length/stride 16, width 128, FFN 256, 4 heads, 2 blocks, top-k 32,
relation-bias weight 1, batch size 512, maximum 50 epochs, patience 10, Adam LR 0.001,
MSE + 0.5 MAE, EMA decay 0.995, and validation-loss checkpoint selection.
RevIN and future-covariate conditioning are disabled in the main configuration.

```bash
python scripts/reproduce.py --sites site1 site4 --seeds 2026 2027 2028 --variants full no_temporal no_exogenous no_lag no_relation_bias dense raw_direction
python scripts/reproduce.py --sites site1 --seeds 2026 --variants topk16 topk64 patch8 patch32 blocks1 blocks3 heads2 heads8
```

Direct CLI tuning remains available:

```bash
python run.py --exp-name tela_site1 --turned topk16_seed2026 --top-k 16 --random-seed 2026
```

Direct `run.py` calls overwrite matching experiment/tag outputs. Untagged JSON:
`result/NAME/metres.json`; tagged JSON: `result/NAME/turnd-TAG.json`; tagged checkpoint:
`checkpoints/NAME_turnd_TAG/checkpoint.pth`. Use distinct tags for different settings.

Result JSON stores arguments, commands, seeds, metrics, selected epoch, checkpoint
path, parameters, runtime, software versions, and the input dataset fingerprint.
Metrics ending in `_norm` use training-standardized targets; unsuffixed metrics use
original-scale power. Paper main-table metrics use `_norm`; WAPE values are ratios.

## Analysis

```bash
python scripts/export_predictions.py --result-file result/tela_site1_full/turnd-seed2026.json
python -m pip install -r requirements-visualization.txt
python scripts/plot_attention_heatmaps.py --result-file result/tela_site1_full/turnd-seed2026.json
python scripts/plot_power_conditioned_attention.py --result-file result/tela_site1_full/turnd-seed2026.json
python scripts/export_relation_bias.py --result-file result/tela_site1_full/turnd-seed2026.json
```

Prediction exports are chronological `[sample, horizon, 1]` arrays at stride-one
forecast origins, accompanied by original-scale arrays and horizon-wise errors.
Prediction export supports `--data-root` and `--checkpoint` when relocating a run.
Analysis scripts expose intermediate attention/bias quantities; their default
plot styles are not the final multi-panel paper layout.

## Code Map

Start at `run.py`, then read `data_provider/data_loader.py`, `models/TELA.py`,
`layers/TELATokenizer.py`, `layers/FutureQuery.py`, `layers/TELAAttention.py`,
and `exp/exp_forecast.py`. The main prediction head is the persistence-residual
head in `layers/PredictionHead.py`.

Historical research options for direct/shape heads, quantiles, future covariates,
RevIN, and ramp losses remain compatible with their saved configurations.
The main reproduction protocol fixes the conference model. Checkpoint loading
is strict except for known inactive, all-zero legacy `query_power_bias` entries.

## Release and License

This directory is the standalone repository root. It contains model source and
reproduction utilities; comparison baselines use their respective implementations.
Data, trained weights, local results, and registration materials are separate.
The source code is released under the [MIT License](LICENSE). Dataset access and
reuse follow the original data provider's terms. See `NOTICE.md` for method
references and `CITATION.bib` for citation metadata. Publication DOI can be added
after publication.
