
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Load artifacts
model = CatBoostRegressor()
model.load_model("catboost_power_model.cbm")

with open("feature_cols.json")  as f: feature_cols  = json.load(f)
with open("feature_stats.json") as f: feature_stats = json.load(f)
with open("type_maps.json")     as f: type_maps      = json.load(f)
with open("target_stats.json")  as f: target_stats   = json.load(f)
with open("model_metrics.json") as f: model_metrics  = json.load(f)

def build_features(air_temp, process_temp, rpm, torque, tool_wear, machine_type):
    """Reconstruct all engineered features from raw inputs."""
    type_num     = type_maps["type_to_num"][machine_type]
    strain_limit = type_maps["type_to_strain"][machine_type]
    temp_diff    = process_temp - air_temp
    wear_torque  = tool_wear * torque
    wear_speed   = tool_wear * rpm

    row = {
        "Air_temperature_K":     air_temp,
        "Process_temperature_K": process_temp,
        "Rotational_speed_rpm":  rpm,
        "Torque_Nm":             torque,
        "Tool_wear_min":         tool_wear,
        "Type_num":              type_num,
        "strain_limit":          strain_limit,
        "temp_diff":             temp_diff,
        "torque_rpm_ratio":      torque / rpm,
        "thermal_stress":        temp_diff * rpm,
        "torque_type_ratio":     torque / type_num,
        "cooling_efficiency":    temp_diff / air_temp,
        "speed_temp":            rpm * process_temp,
        "torque_sq":             torque ** 2,
        "speed_sq":              rpm ** 2,
        "temp_per_speed":        process_temp / rpm,
        "torque_per_temp":       torque / process_temp,
        "wear_torque":           wear_torque,
        "wear_speed":            wear_speed,
        "strain_ratio":          wear_torque / strain_limit,
        "stress_acc_proxy":      torque * rpm * tool_wear,
    }
    return pd.DataFrame([row])[feature_cols]

def predict_power(air_temp, process_temp, rpm, torque, tool_wear, machine_type):
    X = build_features(air_temp, process_temp, rpm, torque, tool_wear, machine_type)
    return float(model.predict(X)[0])
