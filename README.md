# AI-powered-predictive-maintenance

# AI-powered-predictive-maintenance-dashboard

## ⚙️ Predict and Maintain

> AI-powered predictive maintenance dashboard built on the **AI4I 2020 dataset**.
> Detects machine failures, predicts energy consumption, and models thermal behavior — all in a real-time Streamlit UI.


---

## 📌 Project Overview

PredictaMaintain is a three-task machine learning suite that turns raw CNC sensor readings into actionable maintenance intelligence. Given live sensor inputs — air temperature, rotational speed, torque, and tool wear — the system answers three critical operational questions:
```
|        Task            |      Type      |         Question              |      Model         |         Score           |
|------------------------|----------------|-------------------------------|--------------------|-------------------------|
| **Failure Detection**  | Classification | Will this machine fail?       | CatBoostClassifier | F1 @ 0.20 threshold 
| **Power Regression**   | Regression     | Is energy consumption normal? | CatBoostRegressor  | R² = 0.9987 
| **Thermal Regression** | Regression     | Is thermal behavior normal?   | CatBoostRegressor  | R² = 0.7990, MAE = 0.52 K 
```
---

## 🗂️ Project Structure
```
ai4i2020-predictive-maintenance/
│
├── app.py                                  # Main Streamlit application (3 tabs)
├── requirements.txt
│
├── Task 1 (classification - machine failure)/
│   ├── model.cbm                           # CatBoost classifier
│   ├── feature_cols.pkl                    # Ordered feature list
│   ├── label_encoders.pkl                  # Category encoders
│   └── config.json                         # Thresholds, feature ranges
│
├── Task 2 - regression - power/
│   ├── catboost_power_model.cbm            # CatBoost power regressor
│   ├── feature_cols.json
│   ├── feature_stats.json                  # Min/max/mean for UI sliders
│   ├── model_metrics.json
│   ├── target_stats.json
│   └── type_maps.json
│
├── Task 3 - regression - thermal/
│   ├── catboost_temp_model.cbm             # CatBoost thermal regressor
│   ├── feature_cols_temp.json
│   ├── feature_stats_temp.json
│   ├── model_metrics_temp.json
│   ├── target_stats_temp.json
│   ├── type_maps.json
│   └── raw_input_ranges_temp.json
│
└── ai4i2020.csv                            # Source dataset
```

---

## 🧠 Feature Engineering

All 24 features are derived from just **5 raw sensor inputs** and **1 categorical input**:

**Raw inputs:** `Air_temperature_K`, `Process_temperature_K`, `Rotational_speed_rpm`, `Torque_Nm`, `Tool_wear_min`, `Type (L/M/H)`

**Engineered features:**
```
| Feature           | Formula                            | Purpose                          |
|-------------------|------------------------------------|----------------------------------|
| `temp_diff`       | `process_temp − air_temp`          | HDF detection rule               |
| `power`           | `rpm × 2π/60 × torque`             | PWF detection, energy monitoring |
| `wear_torque`     | `tool_wear × torque`               | OSF detection rule               |
| `wear_speed`      | `tool_wear × rpm`                  | Degradation proxy                |
| `torque_rpm_ratio`| `torque / rpm`                     | Mechanical stress indicator      |
| `strain_limit`    | Type → {L:11000, M:12000, H:13000} | Overstrain threshold             |
| `strain_ratio`    | `wear_torque / strain_limit`       | Normalized OSF risk              |
| `mech_stress`     |  `torque × rpm`                    | Combined mechanical load         |
| `thermal_stress`  | `temp_diff × rpm`                  | Heat × friction compound         |
| `stress_acc_proxy`| `torque × rpm × tool_wear`         | Accumulated degradation          |
| `hdf_margin`      | `temp_diff − 8.6`                  | Signed HDF headroom              |
| `pwf_low_margin`  | `power − 3500`                     | Low-power boundary distance      |
| `pwf_high_margin` | `9000 − power`                     | High-power boundary distance     |
```
---

## 🚦 Task Details

### Task 1 — Machine Failure Classification

**Goal:** Predict binary machine failure (`Machine_failure = 1/0`)

- Target is highly imbalanced (~3.4% failure rate)
- Threshold tuned to **0.20** for optimal recall/F1 balance
- Features include all engineered columns above
- CatBoost handles the `Type` categorical natively via Pool

**Key design decisions:**
- Threshold slider in UI (0.05–0.95) so operators can tune sensitivity
- Risk levels: `OPERATIONAL` < 25% · `CAUTION` 25–50% · `HIGH RISK` > 50%

---

### Task 2 — Power Regression (Energy Monitoring)

**Goal:** Predict machine power consumption (Watts) from sensor readings

- R² = **0.9987**, MAE = **12.99 W**, RMSE = **37.54 W**
- Uses physics-consistent features; `power` itself excluded to avoid leakage
- `mech_stress` (= torque × rpm) also excluded as it's proportional to power
- Anomaly detection via Z-score: flags readings beyond ±Nσ of training mean

**Why R² is near 1.0:** Power = rpm × 2π/60 × torque is a deterministic physics relationship. The model is learning this mapping from raw sensor inputs without being given the formula directly — confirming physical consistency.

**Competing models:**
```
| Model            | R²         | MAE (W)   | RMSE (W) |
|------------------|------------|-----------|----------|
| **CatBoost** ✓   | **0.9987** | **12.99** | **37.54** |
| XGBoost          | 0.9985     | 17.40     | 40.90     |
| Ridge (poly+PCA) | 0.8680     | 301.36    | 384.01    |
```
---

### Task 3 — Process Temperature Regression (Thermal Modeling)

**Goal:** Predict process temperature in Kelvin and detect thermal anomalies

- R² = **0.7990**, MAE = **0.52 K**, RMSE = **0.66 K**
- All features containing `Process_temperature_K` excluded (leakage prevention):
  `temp_diff`, `thermal_stress`, `cooling_efficiency`, `speed_temp`, `temp_per_speed`, `torque_per_temp`
- HDF rule: if `process_temp − air_temp > 8.6 K` → Heat Dissipation Failure risk

**Note on R²:** The honest ceiling here. Process temperature has very low variance (std ≈ 1.47 K, range ≈ 8 K). An MAE of 0.52 K means less than half a degree of average error — operationally excellent for thermal monitoring despite moderate R².

**Competing models:**
```
| Model            | R²         | MAE (K)   | RMSE (K)  |
|------------------|------------|-----------|-----------|
| **CatBoost** ✓   | **0.7990** | **0.523** | **0.657** |
| XGBoost          | 0.7982     | 0.520     | 0.659     |
| LightGBM         | 0.7937     | 0.528     | 0.666     |
| Ridge (poly+PCA) | −0.002     | 1.223     | 1.468     |
```
---

## 🖥️ Streamlit UI — Tab Overview

### Tab 1 — ⚠ Failure Detection
- Failure probability gauge (0–100%)
- Live sensor readout cards with warning highlights
- Derived feature display with HDF/strain threshold flags
- Adjustable decision threshold slider
- Feature importance bar chart
- Radar chart: current reading vs dataset mean
- Prediction history with trend line

### Tab 2 — ⚡ Power Regression
- Predicted power gauge in kW
- "Position in Normal Range" band chart showing prediction vs ±Nσ window
- Physics formula vs ML model comparison bar chart
- Contextual anomaly insights (overconsumption / underconsumption)
- Adjustable anomaly sensitivity (σ threshold)
- Model performance metrics panel

### Tab 3 — 🌡 Thermal Regression
- Process temperature gauge with anomaly threshold
- Separate HDF status banner (fires when Δtemp > 8.6 K)
- Predicted vs Sensor vs Dataset Mean grouped bar chart
- Thermal stress decomposition (contribution % from each input)
- Insight cards covering HDF risk, Z-score anomaly, residual warning
- Adjustable HDF threshold and σ sensitivity sliders

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/predictamaintain.git
cd predictamaintain

pip install -r requirements.txt
streamlit run app.py
```

**requirements.txt:**
```
streamlit>=1.35
catboost>=1.2
xgboost>=2.0
lightgbm>=4.0
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
plotly>=5.20
joblib>=1.3
```

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- 10,000 samples, 14 original features
- Source: UCI Machine Learning Repository
- Failure rate: ~3.4% (339 failures)
- Failure types: TWF, HDF, PWF, OSF, RNF
- Dataset Link: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

---

## 🔬 Key ML Learnings

1. **Data leakage awareness** — Every engineered feature was audited per task. Features derived from the target (e.g., `temp_diff` for the thermal task, `mech_stress` for power) were excluded to prevent artificially inflated metrics.

2. **Task-appropriate framing** — Tool wear is stochastic in this dataset; modeling it as a regression is hard-capped by the data generating process. Power and temperature have real physical signals.

3. **Threshold tuning matters** — At default threshold 0.5, the classifier has poor recall on the minority failure class. Tuning to 0.20 significantly improves operational usefulness.

4. **R² context** — A low R² doesn't mean a bad model when target variance is tiny (thermal task). Reporting MAE alongside R² gives a complete picture.

---


---

## 👤 Author

Gaurav Rawat
linkedIn --> https://www.linkedin.com/in/gaurav-rawat-008086221/
