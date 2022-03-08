from sklearn.ensemble import RandomForestClassifier as RFC;
import numpy as np
import pandas as pd
from os.path import exists

objects = {
    0: "Bowtie",
    1: "Broom",
    2: "Crown",
    3: "EiffelTower",
    4: "HotAirBalloon",
    5: "HousePlant"
}

data = pd.DataFrame()

# Load data from all npy files
for object in objects:
    
    # Load the numpy file
    object_data = None
    if exists(f"./data/{objects[object]}.npy"):
        object_data = np.load(f"./data/{objects[object]}.npy")
    else:
        object_data = np.load(f"./DoodleClassifierModel/data/{objects[object]}.npy")
    
    # Append object data to main dataframe
    data = data.append(pd.DataFrame(object_data), ignore_index=True)

print(data)





# X,     Y
# [784], eiffel tower
# [784], book

# Split into x-train/test, and y-train/test

# Fit to model

# Win
# print(data[0])