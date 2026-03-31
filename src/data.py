import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print(f"Data loaded from {path}, shape: {df.shape}")
    return df