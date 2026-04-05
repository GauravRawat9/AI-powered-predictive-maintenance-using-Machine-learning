import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# ── Load all artifacts (call once at app startup) ──────────────────────────────
model = CatBoostRegressor()
model.load_model("catboost_temp_model.cbm")

with open("feature_cols_temp.json")    as f: feature_cols    = json.load(f)
with open("feature_stats_temp.json")   as f: feature_stats   = json.load(f)
with open("target_stats_temp.json")    as f: target_stats    = json.load(f)
with open("model_metrics_temp.json")   as f: model_metrics   = json.load(f)
with open("type_maps.json")            as f: type_maps        = json.load(f)
with open("raw_input_ranges_temp.json")as f: raw_input_ranges = json.load(f)


def build_features(air_temp, rpm, torque, tool_wear, machine_type):
    """
    Reconstruct all 16 engineered features from the 5 raw user inputs.
    Returns a single-row DataFrame with columns in training order.
    """
    type_num     = type_maps["type_to_num"][machine_type]
    strain_limit = type_maps["type_to_strain"][machine_type]
    wear_torque  = tool_wear * torque
    wear_speed   = tool_wear * rpm
    power        = rpm * 2 * 3.141592653589793 / 60 * torque

    row = {
        "Air_temperature_K":    air_temp,
        "Rotational_speed_rpm": rpm,
        "Torque_Nm":            torque,
        "Tool_wear_min":        tool_wear,
        "Type_num":             type_num,
        "strain_limit":         strain_limit,
        "power":                power,
        "torque_rpm_ratio":     torque / rpm,
        "mech_stress":          torque * rpm,
        "torque_type_ratio":    torque / type_num,
        "torque_sq":            torque ** 2,
        "speed_sq":             rpm ** 2,
        "wear_torque":          wear_torque,
        "wear_speed":           wear_speed,
        "strain_ratio":         wear_torque / strain_limit,
        "stress_acc_proxy":     torque * rpm * tool_wear,
    }
    return pd.DataFrame([row])[feature_cols]


def predict_temperature(air_temp, rpm, torque, tool_wear, machine_type):
    """Returns predicted Process Temperature in Kelvin."""
    X = build_features(air_temp, rpm, torque, tool_wear, machine_type)
    return float(model.predict(X)[0])


def is_thermal_anomaly(predicted_temp, threshold_std=2.0):
    """
    Returns True if predicted temp deviates beyond threshold_std
    standard deviations from the training mean.
    Useful for Task 4 framing: 'Is thermal behavior normal?'
    """
    mean = target_stats["mean"]
    std  = target_stats["std"]
    return abs(predicted_temp - mean) > threshold_std * std
