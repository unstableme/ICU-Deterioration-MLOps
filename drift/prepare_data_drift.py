import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path


DATA_PATH_A = "https://dagshub.com/unstableme/ICU-Deterioration-MLOps/raw/484a042bc261b6da049be96de0e8b767ede07fcd/data/processed/set_a_processed.pkl"
DATA_PATH_C = "https://dagshub.com/unstableme/ICU-Deterioration-MLOps/raw/484a042bc261b6da049be96de0e8b767ede07fcd/data/processed/set_c_processed.pkl"

OUTPUT_DIR = Path("data/drift")

def download_data(data_path):
    response = requests.get(data_path)
    return pickle.loads(response.content)

def convert_to_2d(X):
    """
    X shape: (n_samples, time_steps, n_features)
    Returns: DataFrame (n_samples, n_features * stats)..takes mean & 3 others stats
    across time for each patient and feature
    """
    stats = {
        "mean": np.mean(X, axis=1),
        "std": np.std(X, axis=1),
        "min": np.min(X, axis=1),
        "max": np.max(X, axis=1)
    }
    dfs = []
    for stats, values in stats.items():
        df = pd.DataFrame(values)
        df.columns = [f"f{idx}_{stats}" for idx in range(values.shape(1))]
        dfs.append(df)
    
    return pd.concat(dfs, axis=1)

def convert_to_dataframe(dataset):

    X = dataset["X"]
    y = dataset["y"]
    df = convert_to_2d(X)
    df["target"] = y
    return df

def prepare_drift_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_dataset = download_data(DATA_PATH_A)
    ref_df = convert_to_dataframe(ref_dataset)

    cur_dataset = download_data(DATA_PATH_C)
    cur_df = convert_to_dataframe(cur_dataset)

    ref_path = OUTPUT_DIR/ "reference.parquet"
    cur_path = OUTPUT_DIR/ "current.parquet"

    ref_df.to_parquet(ref_path, index=False)
    cur_df.to_parquet(cur_path, index=False)

    print("Drift data prepared successfully")
    print(f"Reference shape: {ref_df.shape}")
    print(f"Current shape:   {cur_df.shape}")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    prepare_drift_data()




