# Battery State-of-Health Estimation with Deep Learning

Estimating the State-of-Health (SoH) of lithium-ion cells from raw Battery Management System (BMS) signals (voltage, current, temperature) using a family of models that runs from a gradient-boosting baseline up to convolutional-recurrent hybrids with attention.

**Authors:** ELIOS Lab (Matteo Fresta, Hadi Ballout, Sina Gholami Fashkhami)
**Supervisor:** Prof. Francesco Bellotti
**Affiliation:** ELIOS Lab, University of Genoa

---

## 1. Read this first: what the project is really about

A compact hybrid model (CNN feeding a bidirectional recurrent backbone with attention) reaches a competitive error on a hard, realistic dataset, evaluated the honest way (no cell is ever in both training and test). That is the good half of the finding.

The costly half is that the accuracy was reached with a very long input window, on the order of three weeks of cell history. A BMS cannot wait three weeks to emit its first estimate, and a three-week window ties the estimate to slow drift rather than the cell's current state. So the central research problem of this repository is not "which architecture wins". The architecture question is largely settled. The open problem is: **can we keep the accuracy while making the input window short and practical?**

Everything in the frontier work (notebook `15`, plus the future directions in Section 12) exists to answer that question. Two levers are being pulled: redefining a window around charge/discharge cycles instead of clock time (cycle-aware windowing), and normalizing each cell by its own statistics instead of a global scale (per-battery normalization).

---

## 2. The dataset

The experiments use the comprehensive lithium-ion aging dataset published by Luh and Blank (also referred to in the literature as the RWTH-Aachen / KIT comprehensive aging dataset). The cells have an NMC/C-SiO chemistry, meaning a nickel-manganese-cobalt positive electrode and a graphite/silicon-oxide negative electrode, which is close to modern automotive cells. What makes the dataset valuable is that it mixes three aging regimes in one experiment: calendar aging (cells rest at a fixed state of charge and temperature), cyclic aging (repeated charge/discharge at controlled rates), and profile aging (realistic driving-cycle load profiles), across roughly 0, 10, 25, and 40 degrees Celsius. A model that generalizes across that variety has learned something closer to the physics of degradation than a model tuned to one protocol, which is exactly the property this project wants to test.

Public source (Kaggle):
- Measurement data: https://www.kaggle.com/datasets/matthiasluh/battery-aging-dataset-measurement-data-v2
- Result data: https://www.kaggle.com/datasets/matthiasluh/battery-aging-dataset-result-data-v2

This repository does not download from Kaggle directly. Notebook `0` pulls a partially preprocessed `.npy` version prepared by the team from a public Google Drive folder (the file id is hardcoded in the download cell, and the link is public, so no access request is needed).

**Working subset:** 76 cells labeled `P001` through `P076`. Each cell contributes several recording files. After preprocessing (Section 5) each file becomes a set of windows, and each window carries a SoH label from the capacity check-ups. The number of windows per cell varies a lot (from a few dozen for short-lived cells to over a hundred for long-lived ones), which is why the validation protocol partitions by cell rather than by window.

**Signal channels:** the three measured channels are voltage, current, temperature. Two derived channels, the time derivatives dV/dt and dI/dt, are computed on the fly and highlight the transitions between rest, charge, and discharge. The derivatives are treated as an option, not a requirement, because differentiation also amplifies noise. Most notebooks here run with all five channels `[V, I, T, dV/dt, dI/dt]`.

**Label range:** SoH runs roughly from 70% to 98%, with most windows clustered in the low-to-mid eighties. The distribution is skewed, which is why the training loss is Huber (Section 8) and why R² is always reported next to RMSE and MAE (a naive mean-predictor already scores a deceptively small absolute error on a narrow label range).

---

## 3. Repository layout

Every notebook lives at the repository root, and every data / output folder is created at the root as well. Run notebooks from the repository root so that `os.getcwd()` resolves correctly.

```
.
├── 0_Preprocessing_Data.ipynb        # download, downsample, drop SoC, window, split, scale
├── 1_Window_Stride_Search.ipynb      # sweep window size and stride under an equal gradient budget
├── 2_Data_Visualization.ipynb        # EDA over raw/processed/windowed/normalized data
├── 2_Data_Visualization2.ipynb       # SoH-span deep dive (within-file vs across-file span, cycles)
├── 3_XGBoost.ipynb                   # gradient-boosting baseline (final train + figures)
├── 4_XGBoost_Optuna.ipynb            # Optuna tuning for XGBoost (grouped CV)
├── 5_GRU.ipynb                       # GRU + attention (final train + figures)
├── 6_GRU_Optuna.ipynb                # Optuna tuning for GRU (composite RMSE+R2 objective)
├── 7_1dCNN.ipynb                     # 1D-CNN (final train + figures)
├── 8_1dCNN_Optuna.ipynb              # Optuna tuning for 1D-CNN
├── 9_CNN_GRU.ipynb                   # CNN + GRU + attention (no dedicated Optuna notebook)
├── 10_LSTM.ipynb                     # LSTM + attention (final train + figures)
├── 11_LSTM_Optuna.ipynb              # Optuna tuning for LSTM
├── 12_CNN_LSTM.ipynb                 # CNN + LSTM + attention (final train + figures)
├── 13_CNN_LSTM_Optuna.ipynb          # Optuna tuning for CNN + LSTM
├── 14_Spectro_Autoencoder.ipynb      # exploratory dual-branch STFT+CWT autoencoder
├── 15_Config_Search.ipynb            # SOURCE OF TRUTH: fixed vs cycle windowing, per-battery norm
├── requirements.txt
├── .gitignore
└── README.md
```

Folders created at runtime (all git-ignored, see Section 11):

```
data_raw/               # downloaded .npy cell logs
data_downsampled/       # after time-based downsampling
data_processed/         # after dropping SoC  -> columns [time, V, I, T, SoH]
data_windowed/          # GRU/XGB windows, per source file
data_normalized/        # GRU/XGB train/val/test .npy + scale params
data_cnn/               # CNN windows, per source file
data_cnn_normalized/    # CNN train/val/test .npy + scale params
data_group/             # groups_train.npy for XGBoost grouped CV
data_spectrograms_v2/   # cached STFT/CWT tensors for notebook 14
model/                  # saved weights / pickles
figures/                # all generated figures, per model
log/                    # split_batteries.json and Optuna/config-search result JSON
```

---

## 4. Environment and setup

Kernel used during development: Python 3.10.11.

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch to match your CUDA version rather than relying on the plain wheel. Follow the selector at https://pytorch.org/get-started/locally/ if the default install does not pick up your GPU. The recurrent and hybrid notebooks are slow on CPU but will run.

Then launch Jupyter from the repository root:

```bash
jupyter lab
```

---

## 5. The pipeline, end to end

Run the notebooks in this order the first time.

1. **`0_Preprocessing_Data.ipynb`** builds every dataset the models consume. It downloads the raw `.npy` files, downsamples from a 30-second interval to 30 minutes by time binning (keeping the last sample per bin), drops the SoC column so the processed layout is `[time, V, I, T, SoH]`, then windows the signal twice. It produces a GRU/XGB dataset in `data_normalized/` and a CNN dataset in `data_cnn_normalized/`. Both use the five features `[V, I, T, dV/dt, dI/dt]` and both label a window with the **SoH at its last timestep** (not the mean). The split is battery-aware 70/15/15, statistics are fit on the training split only, and the scale parameters are saved for inference. It also writes `data_group/groups_train.npy` so XGBoost can do grouped cross-validation.

2. **`1_Window_Stride_Search.ipynb`** is the experiment that first exposed the window-length problem. It sweeps window sizes and strides for the GRU and the CNN+GRU under an equalized gradient budget (every configuration gets the same number of gradient steps, so a config with more windows does not get an unfair training advantage), keeps the best-validation-epoch checkpoint for each run, and plots validation/test RMSE, R², the overfitting gap, active-window fraction, and the best prediction scatter. This is where the long best-window came from.

3. **`2_Data_Visualization.ipynb`** and **`2_Data_Visualization2.ipynb`** are EDA. The first walks the data through every stage (raw, processed, windowed, normalized, CNN) with distributions, correlations, frequency-domain views, and per-split label histograms. The second is a focused investigation of where the SoH span comes from (within a file versus across files of the same cell) and how many charge/discharge cycles each cell contains. Run these to build intuition and to catch silent data problems before training.

4. **Model notebooks (`3`, `5`, `7`, `9`, `10`, `12`) and their Optuna partners (`4`, `6`, `8`, `11`, `13`).** Each `_Optuna` notebook searches hyperparameters and writes the best configuration; the corresponding model notebook trains a final model with a fixed configuration, evaluates on the held-out test cells, and saves ten diagnostic figures plus the model. Note that `9_CNN_GRU.ipynb` has no dedicated Optuna notebook; its hyperparameters were set from the related searches.

5. **`14_Spectro_Autoencoder.ipynb`** is an exploratory side branch (see Section 7).

6. **`15_Config_Search.ipynb` is the source of truth.** When results disagree between the hardcoded windowing in notebook `0` and this notebook, trust notebook `15`. It is the standardized experiment that decides which windowing and normalization strategy to use (Section 9).

---

## 6. Data representation and conventions you must respect

These conventions are load-bearing. Breaking one silently produces numbers that look fine and mean nothing.

**Two dataset formats, same information.** Tree and recurrent models read a window as a sequence of feature vectors from `data_normalized/`. Convolutional models read the same windows with the channel axis arranged for 1D convolution from `data_cnn_normalized/`. Both come from the same segmentation, so a comparison across models measures the architecture, not a data difference.

**Label is the last SoH in the window, not the mean.** This was a deliberate correction. The window's job is to describe the cell's health at the moment the window ends, which is what a BMS would report.

**Battery-aware split, always.** Windows from one cell are highly correlated. Every window from a given cell goes into exactly one of train, validation, or test. No cell is split across partitions. A random window-level split would leak and inflate the score. Cells, not windows, are the unit of partition, so the window-level shares (roughly 62.5 / 20 / 17.5 percent) do not exactly match the cell-level 70 / 15 / 15.

**Fit scaling on train only.** Normalization statistics come from the training split and are applied unchanged to validation and test. Notebook `0` saves the scale parameters so inference can reuse them.

**Best-epoch checkpoint.** Training keeps the checkpoint with the lowest validation loss, not the final-epoch weights, so the reported model is the one that generalized best.

**Report over three seeds.** Run-to-run variation in R² is around 0.2 with a single seed, which is large enough to reverse a ranking. Notebook `15` trains seeds 42, 43, 44 and reports mean plus or minus standard deviation. Read every gap against the standard deviation, not the mean alone.

**Bidirectional hidden size.** When a recurrent layer is bidirectional, its output width is `hidden_size * 2`. The attention module and the final linear layer must be built with that doubled width (`H = hidden_size * 2`). Getting this wrong is a silent shape bug that trains without error.

---

## 7. The model family

Six regression models plus one exploratory autoencoder, all trained with the same Huber loss and the same cell-grouped protocol so that differences reflect the architecture.

- **XGBoost (gradient-boosting baseline).** Flattens each window into one row. It discards the explicit time order, which is the point: it is a floor. A deep model that cannot beat a well-tuned XGBoost is not earning its complexity.
- **1D-CNN.** Convolutions along time with a wide first kernel then narrower kernels, pooling between stages, and a global average over time so the output does not depend on exact window length. Light and deployment-friendly.
- **GRU + attention** and **LSTM + attention.** Recurrent backbones with additive (Bahdanau-style) attention over time instead of the last-hidden-state bottleneck, so no timestep is forgotten and the model learns which parts of the window matter.
- **CNN + GRU + attention** and **CNN + LSTM + attention.** The convolution compresses the raw window into a short sequence of local-feature vectors, then the bidirectional recurrent backbone models how those features evolve, with attention pooling before a small regression head. The CNN+GRU is the strongest model in the study.
- **Spectrogram autoencoder (exploratory, notebook 14).** A dual-branch encoder over STFT and CWT time-frequency representations, trained in three phases (unsupervised reconstruction, then a frozen-encoder SoH head, then end-to-end fine-tuning). It is a side idea, reported as such, and it is not part of the main contribution.

### Results

Test-set performance on held-out cells under the cell-grouped protocol.

| Model | RMSE | MAE (SoH point) | R² | Notes |
|---|---|---|---|---|
| XGBoost (flattened windows) | 0.46 | 3.12 | 0.79 | tabular baseline |
| 1D-CNN | 0.44 | 2.61 | 0.81 | learned local features |
| GRU + attention | 0.35 | 2.48 | 0.88 | bidirectional, attention pooling |
| LSTM + attention | 0.44 | 2.53 | 0.81 | bidirectional, attention pooling |
| CNN + GRU + attention | 0.62 | 2.21 | 0.93 | strongest; long window |
| CNN + LSTM + attention | 0.35 | 2.41 | 0.88 | hybrid variant |
| Spectrogram autoencoder | 0.48 | 3.37 | 0.77 | exploratory side idea |

The stable conclusions are the ones to rely on: the deep models beat the tabular baseline, the CNN+GRU hybrid is strongest by R², and it earns that strength with a long window. Note that the two columns are on different scales and some of them are done with unoptimized models, so confirm the result before publishing.

---

## 8. Loss and training protocol

All deep models minimize Huber loss, which is quadratic for small residuals and linear for large ones. This matters because the label distribution is skewed with sparse extremes: under plain squared error a few hard windows at very high or very low health would dominate the gradient and pull the model toward the extremes at the expense of the dense middle. The Huber threshold (delta) is itself tuned. Optimization uses Adam with weight decay, gradient-norm clipping (needed for the recurrent models), and a learning-rate schedule that drops on a validation plateau. Training runs up to several hundred epochs with early stopping on validation loss, keeping the best-validation checkpoint.

---

## 9. Notebook 15, search experiment

`15_Config_Search.ipynb` is a single standardized harness that compares windowing and normalization strategies on equal terms, so any difference is attributable to the configuration and not to a training accident. It:

- persists the battery split to `log/split_batteries.json` and refuses to run if the file no longer matches the cells on disk, so the test set is stable across experiments;
- implements cycle detection from the current signal (splitting active runs into charge and discharge half-cycles) and builds cycle-based windows (a fixed number of half-cycles, resampled to 128 points by linear interpolation) alongside fixed-time windows;
- implements per-battery normalization against a fresh-cell reference, with a standard-deviation floor and sigma clipping, as an alternative to global normalization;
- trains every configuration with three seeds (42, 43, 44) and reports R² and RMSE as mean plus or minus standard deviation, plus a per-battery RMSE/R² breakdown.

Configurations compared include fixed windows at three lengths (`F1000`, `F1666`, `F1166`, all global-normalized) and cycle windows (`C5`, `C8`, global; `C5` per-battery). Set `QUICK_TEST=True` first to verify the plumbing on a small subset in roughly ten minutes, then flip to `False` for the real run.

Current standing from this experiment: **`F1166_global` leads** (a 1166-length fixed window with global normalization, stride 291), and per-battery normalization was ruled out as a win in this comparison. Note that the stride here (291) differs from the stride hardcoded in notebook `0` (145 for the CNN dataset); when they disagree, notebook `15` is authoritative.

---

## 10. State of the art

The current headline in early SoH prediction is **BatteryGPT** (Hu et al., *Nature Communications*, 2025, https://doi.org/10.1038/s41467-025-66819-0). It is a two-stage method: a GPT-Small causal transformer autoregressively generates a cell's full-lifecycle charging signals from early data, and a CNN-LSTM SoH estimator then reads those generated signals. Given the first 30% of a cell's life, it reports an SoH-variation RMSE around 0.21%.

The comparison with this repository is deliberately not like-for-like, and the differences are the point. BatteryGPT is evaluated on a small number of held-out cells of a single chemistry aged under a single fast-charging protocol (the MIT dataset), which is an easier generalization test than the calendar/cyclic/profile mix across temperatures used here. It reaches its figure with a heavy generative pipeline of tens of millions of parameters, and it is given a third of the lifecycle as context, which is precisely the long-context requirement this project is trying to remove. The honest reading is that the state of the art achieves a very low error under favorable, narrow conditions with a large model, and the aim here is a competitive error under harder, more realistic conditions with a model small enough to run on a BMS.


## 11. Future work and ideas

The frontier items below are the intended next steps; the rest are open directions.

**Close the window-length gap (highest priority).** The whole point is a short, practical input window that keeps the hybrid's accuracy. Continue the cycle-aware windowing in notebook `15`: sweep the number of half-cycles per window and find the smallest count that matches the long fixed-window accuracy under the same cell-grouped protocol. The fair-reporting rule is strict here: any short-window win must be shown on the full diverse population with cells held out, never on an easier subset or a random split.

**Online, per-battery normalization.** The offline per-battery normalization tested so far did not win, but the deployment-realistic version is online per-cell normalization, where a held-out cell accumulates its own statistics over time exactly as a BMS would, without ever touching training-cell statistics. Compare three regimes on equal terms: global, offline per-cell, online per-cell.

**Uncertainty quantification.** A SoH number without a confidence is weak in a safety-critical setting. Add calibrated uncertainty so a BMS can distrust an estimate on an unfamiliar cell and defer or request a reference measurement. Natural options that fit the existing models: a deep ensemble across seeds (the three-seed runs are already halfway there), Monte-Carlo dropout at inference, or a heteroscedastic head that predicts a mean and a spread.

**Edge deployment, measured not assumed.** The models are designed to be light, but that claim should be demonstrated. Measure size, latency, and energy per inference on a representative embedded target, then apply quantization, pruning, or distillation (the road-surface work in the wider thesis reached a few-hundred-kilobyte model this way).

**Extensions worth trying.**
- Add a dedicated Optuna notebook for CNN+GRU and re-tune the strongest model directly.
- Push the spectrogram branch further, or drop it cleanly if it stays uncompetitive; right now it sits between the two.
- Test whether the three raw channels alone match the five-channel setup on the best model, since fewer channels is cheaper to run online.
- Add a fixed random seed and deterministic flags across the model notebooks (only notebook `15` seeds carefully) so single-run results are reproducible.
- Consider a small config module or shared `utils.py` so the `Attention` class, the loaders, and the split logic are defined once.

---

## 12. References

- Luh, M. and Blank, T. Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell. *Scientific Data* 11, 1004 (2024).
- Hu, J. et al. Early prediction of lithium-ion battery degradation with a generative pre-trained transformer. *Nature Communications* 17, 126 (2025). https://doi.org/10.1038/s41467-025-66819-0
- Severson, K. A. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* 4, 383-391 (2019). (The fast-charging dataset used by the BatteryGPT benchmark.)


