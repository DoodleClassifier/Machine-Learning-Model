import numpy as np

OBJECT = "HousePlant"

data = np.load(f"./raw_data/{OBJECT}.npy")[:1000]

np.save(f"./data/{OBJECT}.npy", data)

