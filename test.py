from src.preprocessing import preprocess_data
from src.visualize import create_master_figure

dfg = preprocess_data("data/preprocessed/2024pg.csv")
#print(dfg.iloc[:2,200:].head())
#print(dfg.shape)

import pandas as pd
import numpy as np
df = pd.read_csv("results/metrics.csv")
create_master_figure(df)