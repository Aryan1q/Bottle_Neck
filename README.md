# Traffic Bottleneck Predictor
### CONVOKE 8.0 — KnowledgeQuarry | Problem 01: The Bottleneck Problem

A multi-task deep learning model that predicts traffic merge delays, waiting times, and congestion status at road bottlenecks. The architecture is inspired by Neural Style Transfer (NST), mapping its encoder–decoder–loss-net pattern onto a traffic simulation problem.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Components](#model-components)
- [Training](#training)
- [Evaluation](#evaluation)
- [Signal Optimiser](#signal-optimiser)
- [Outputs](#outputs)

---

## Overview

This notebook trains a neural network to simultaneously:

1. **Regress** `merge_delay_sec` — how long a vehicle is delayed at the bottleneck.
2. **Regress** `waiting_time` — total time a vehicle spends in the system.
3. **Classify** `congestion_flag` — whether a vehicle experiences congestion (binary).

A rule-based **Signal Optimiser** then uses the model's predictions to recommend traffic signal green-phase durations per lane and driver aggressiveness scenario.

---

## Dataset

**File:** `traffic_bottleneck_dataset.csv`

| Column | Description |
|---|---|
| `vehicle_id` | Unique vehicle identifier |
| `arrival_time_sec` | When the vehicle arrives at the bottleneck (seconds) |
| `lane` | Approach lane (1, 2, or 3) |
| `speed_kmph` | Approach speed in km/h |
| `aggressiveness` | Driver behaviour class: `low`, `medium`, or `high` |
| `merge_delay_sec` | Delay incurred during the merge/bottleneck event (seconds) |
| `exit_time_sec` | When the vehicle clears the bottleneck (seconds) |

**Derived features (computed during preprocessing):**

- `waiting_time` = `exit_time_sec` − `arrival_time_sec`
- `speed_norm` = speed normalised to [0, 1]
- `lane_gap` = lane-wise inter-vehicle arrival gap (replaces global gap)
- `rolling_congestion` = rolling 5-vehicle mean of `merge_delay_sec` per lane
- `congestion_flag` = 1 if `merge_delay_sec` ≥ 60th percentile threshold

---

## Architecture

The model mirrors the NST pipeline structure:

```
vehicle features
    → Encoder           (Dense encoder ≈ VGG encoder)
    → Decoder           (merge_delay regression ≈ content decoder)
    → WaitingDecoder    (waiting_time regression ≈ AdaIN secondary synthesis)
    → CongestionClf     (binary congestion classification ≈ loss net)
```

**Multi-task loss:**
```
total_loss = delay_MSE + waiting_MSE + STYLE_WEIGHT × congestion_BCE
```

---

## Installation

This notebook is designed for **Google Colab**. No local setup is needed beyond uploading the notebook and dataset.

**Dependencies** (all pre-installed in Colab):

```
tensorflow >= 2.x
numpy
pandas
scikit-learn
matplotlib
```

**Google Drive** is used to persist model checkpoints across sessions.

---

## Usage

1. Open the notebook in Google Colab.
2. Run all cells in order.
3. When prompted, upload `traffic_bottleneck_dataset.csv`.
4. Training will begin automatically and save the best weights to your Google Drive at:
   ```
   /content/drive/MyDrive/traffic_bottleneck/model_weights.weights.h5
   ```
5. Evaluation, dashboard, signal optimiser, and inference will run automatically after training.

**To load a pre-trained checkpoint** instead of training from scratch, uncomment:
```python
model.load_weights(CHECKPOINT_PATH)
```

---

## Model Components

### Encoder
- Input → Dense(128, ReLU) → BatchNorm → Dropout(0.2) → Dense(64, ReLU) → BatchNorm
- Output: 64-dimensional latent traffic embedding

### Decoder (Merge Delay)
- Latent → Dense(64, ReLU) → Dropout(0.15) → Dense(32, ReLU) → Dense(1, linear)

### Waiting Decoder
- Latent → Dense(32, ReLU) → Dense(1, linear)

### Congestion Classifier
- Latent → Dense(32, ReLU) → Dropout(0.2) → Dense(1, sigmoid)

---

## Training

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Epochs | 50 (with early stopping) |
| Latent dim | 64 |
| Learning rate | 1e-3 |
| Style weight (classification loss) | 0.4 |
| Congestion percentile threshold | 60th |
| Train/val split | 80/20, stratified |

**Callbacks:**
- `ModelCheckpoint` — saves best weights to Drive (monitors `val_loss`)
- `EarlyStopping` — patience of 10 epochs, restores best weights
- `ReduceLROnPlateau` — halves LR after 5 stagnant epochs (min LR: 1e-6)
- `TrafficMonitor` — plots predicted vs actual merge delay every 10 epochs

---

## Evaluation

Metrics reported on the validation set:

- **Merge delay:** MAE, RMSE
- **Waiting time:** MAE, RMSE
- **Congestion classification:** Accuracy, classification report (precision/recall/F1)

A 6-panel **Model Dashboard** is generated and saved to `/content/traffic_dashboard.png`, showing:

1. Predicted vs actual merge delay (scatter)
2. Training loss curve
3. Congestion ROC curve
4. Merge delay distribution by driver aggressiveness
5. Throughput per lane over time (stacked bar)
6. Congestion classification confusion matrix

---

## Signal Optimiser

After training, the optimiser evaluates 9 hypothetical scenarios (3 lanes × 3 aggressiveness levels) and recommends a traffic signal green-phase duration for each:

```
Base green phase: 30s
Extra green      = max(0, int((predicted_delay − 10) × 0.8))
Recommended      = base + extra
```

A bar chart is saved to `/content/signal_optimiser.png` with green bars (no congestion predicted) and red bars (congestion likely).

---

## Outputs

| File | Description |
|---|---|
| `model_weights.weights.h5` | Best model checkpoint (saved to Drive) |
| `traffic_dashboard.png` | 6-panel evaluation dashboard |
| `signal_optimiser.png` | Signal timing recommendation chart |

---

## Project Context

This project is part of **CONVOKE 8.0 — KnowledgeQuarry**, ML Engineering Track, Problem 01. The NST-inspired architecture is an intentional design choice to demonstrate how encoder–decoder–loss-net patterns generalise beyond image stylisation to structured tabular/simulation problems.
