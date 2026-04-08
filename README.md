# 🚦 The Bottleneck Problem
### CONVOKE 8.0 — KnowledgeQuarry | Data Science Challenge
**CIC · University of Delhi · ML Engineering Track**

---

## 🧠 Problem Context

In urban environments, traffic congestion rarely stems from vehicle density alone — it emerges from **unstructured human behaviour in constrained spaces**. A wide road narrows into a bottleneck (lane reduction, railway crossing, intersection), vehicles abandon lane discipline, merge aggressively, and conflicting diagonal movements create cascading delays or full gridlock.

This project is a **data-driven solution** to model, analyse, and improve traffic flow at such bottlenecks. It targets the ML Engineering Track, which requires:
- Predicting congestion and flow patterns
- Simulating traffic dynamics
- Optimising control strategies (signal timing)

The solution addresses all three **required metrics** from the problem statement:

| Metric | How it's addressed |
|---|---|
| ✅ Avg Waiting Time | Regressed directly as a model output |
| ✅ Throughput (veh/unit time) | Computed and visualised per lane over time windows |
| ✅ Congestion | Binary classification + confusion matrix evaluation |

---

## 📐 Approach: Simulated Environment

The solution follows **Approach 01 (Simulated Environment)** from the problem statement. A synthetic dataset (`traffic_bottleneck_dataset.csv`) models a 3-lane bottleneck scenario with parameterised road structure, arrival rates, and driver behaviour.

> **Assumptions (C-03):**
> - Traffic rules: vehicles in all 3 lanes compete for the same merge point
> - Driver behaviour model: encoded as a 3-class aggressiveness level (`low` / `medium` / `high`) that influences merge delay
> - Data source: fully simulated

---

## 🏗️ Architecture

The model is a **multi-task neural network** mapping the encoder → decoder → loss-net pattern onto a traffic problem:

```
Vehicle Features
      │
      ▼
┌─────────────┐
│   Encoder   │  Dense(128) → BN → Dropout → Dense(64) → BN
│  (≈ VGG)    │  Learns a latent "traffic state" embedding
└──────┬──────┘
       │  64-dim latent vector
   ┌───┴────────────────┬───────────────────┐
   ▼                    ▼                   ▼
┌──────────┐   ┌──────────────────┐   ┌───────────────┐
│ Decoder  │   │ Waiting Decoder  │   │  Congestion   │
│          │   │                  │   │  Classifier   │
│          │   │                  │   │ (≈ loss net)  │
└──────────┘   └──────────────────┘   └───────────────┘
      │                 │                    │
      ▼                 ▼                    ▼
merge_delay_sec   waiting_time_sec    congestion_flag
  (regression)      (regression)      (binary class)
```

**Multi-task loss:**
```
total_loss = MSE(merge_delay) + MSE(waiting_time) + 0.4 × BCE(congestion)
```

---

## 📦 Dataset

**File:** `traffic_bottleneck_dataset.csv`

| Column | Description |
|---|---|
| `vehicle_id` | Unique vehicle identifier |
| `arrival_time_sec` | When the vehicle arrives at the bottleneck (seconds) |
| `lane` | Approach lane: 1, 2, or 3 |
| `speed_kmph` | Approach speed in km/h |
| `aggressiveness` | Driver behaviour class: `low`, `medium`, or `high` |
| `merge_delay_sec` | Delay incurred at the bottleneck merge point (seconds) |
| `exit_time_sec` | When the vehicle clears the bottleneck (seconds) |

**Engineered features (computed in preprocessing):**

| Feature | Description |
|---|---|
| `waiting_time` | `exit_time_sec − arrival_time_sec` |
| `speed_norm` | Speed normalised by fleet max |
| `lane_gap` | Lane-wise inter-vehicle arrival gap (more accurate than global gap) |
| `rolling_congestion` | Rolling 5-vehicle mean of `merge_delay_sec` per lane |
| `congestion_flag` | 1 if `merge_delay_sec` ≥ 60th percentile threshold |

---

## ⚙️ Setup

This notebook runs entirely on **Google Colab** — no local setup needed.

**Dependencies** (all pre-installed in Colab):
```
tensorflow >= 2.x  |  numpy  |  pandas  |  scikit-learn  |  matplotlib
```

Google Drive is mounted to persist checkpoints at:
```
/content/drive/MyDrive/traffic_bottleneck/model_weights.weights.h5
```

---

## 🚀 Usage

1. Open `bottleneck.ipynb` in Google Colab
2. Run all cells in order
3. When prompted, upload `traffic_bottleneck_dataset.csv`
4. Training runs automatically with live monitoring plots every 10 epochs
5. Evaluation, dashboard, signal optimiser, and inference all run automatically after training

To resume from a saved checkpoint, uncomment:
```python
model.load_weights(CHECKPOINT_PATH)
```

---

## 🔧 Model Components

### Encoder
Learns a compressed 64-dim representation of the traffic state.
```
Input(7) → Dense(128, ReLU) → BatchNorm → Dropout(0.2) → Dense(64, ReLU) → BatchNorm
```

### Merge Delay Decoder
Predicts how long a vehicle will be delayed at the merge point.
```
Latent(64) → Dense(64, ReLU) → Dropout(0.15) → Dense(32, ReLU) → Dense(1, linear)
```

### Waiting Time Decoder
Predicts total time in system from arrival to exit.
```
Latent(64) → Dense(32, ReLU) → Dense(1, linear)
```

### Congestion Classifier
Binary prediction of whether a vehicle hits congestion.
```
Latent(64) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, sigmoid)
```

---

## 🏋️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Max epochs | 50 |
| Latent dimension | 64 |
| Learning rate | 1e-3 |
| Classification loss weight | 0.4 |
| Congestion threshold | 60th percentile of `merge_delay_sec` |
| Train / val split | 80 / 20, stratified on congestion label |

**Callbacks:**

| Callback | Behaviour |
|---|---|
| `ModelCheckpoint` | Saves best weights to Drive (monitors `val_loss`) |
| `EarlyStopping` | Stops after 10 stagnant epochs; restores best weights |
| `ReduceLROnPlateau` | Halves LR after 5 stagnant epochs (floor: 1e-6) |
| `TrafficMonitor` | Plots predicted vs actual merge delay every 10 epochs |

---

## 📊 Evaluation

Metrics reported on the held-out validation set:

| Output | Metrics |
|---|---|
| Merge delay | MAE, RMSE |
| Waiting time | MAE, RMSE |
| Congestion flag | Accuracy, Precision, Recall, F1 |

A **6-panel model dashboard** is saved to `traffic_dashboard.png`:

1. Predicted vs actual merge delay (scatter)
2. Training loss curve
3. Congestion ROC curve with AUC
4. Merge delay distribution by driver aggressiveness
5. Throughput per lane over time (stacked bar)
6. Congestion confusion matrix

---

## 🚥 Signal Optimiser

The optimiser is the practical output of this project — directly addressing the problem's goal of improving flow through **signal timing control**. It evaluates 9 scenarios (3 lanes × 3 aggressiveness levels) and recommends a green-phase duration for each:

```
Base green phase : 30 seconds
Extra green      : max(0, int((predicted_delay − 10) × 0.8))
Recommended      : base + extra
```

- 🟢 Green bar — no congestion predicted, standard timing
- 🔴 Red bar — congestion likely, extended green phase recommended

Results are printed as a table and saved to `signal_optimiser.png`.

---

## 📁 Output Files

| File | Description |
|---|---|
| `model_weights.weights.h5` | Best model checkpoint (Google Drive) |
| `traffic_dashboard.png` | 6-panel evaluation dashboard |
| `signal_optimiser.png` | Signal timing recommendations per scenario |

---

## 📋 Constraints Checklist

| Constraint | Requirement | Status |
|---|---|---|
| C-01 | Single bottleneck type | ✅ 3-lane merge scenario |
| C-02 | At least one defined metric optimised | ✅ Waiting time + Throughput + Congestion |
| C-03 | Assumptions clearly stated | ✅ Simulated data, aggressiveness model, lane assumptions above |

Out of scope (per problem statement): physical infrastructure redesign, legal/enforcement systems, hardware implementation.
