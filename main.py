from sklearn.ensemble import RandomForestClassifier as RFC;
import numpy as np
import pandas as pd

objects = {
    0: "Bowtie",
    1: "Broom",
    2: "Crown",
    3: "EiffelTower",
    4: "HotAirBalloon",
    5: "HousePlant"
}

data = pd.DataFrame(columns=["X", "Y"])

# Load data from all npy files
for object in objects:
    
    # Load the file and take first 1000 entries
    object_data = np.load(f"./data/{objects[object]}.npy")[:1000]

    # Add label tag to each entry
    object_data = [[x, objects[object]] for x in object_data]

    # Create temporary data frame to store new object data
    temp = pd.DataFrame(object_data, columns=["X", "Y"])

    # Append temporary data frame to data
    data = data.append(temp, ignore_index=True)

print(data)





# X,     Y
# [784], eiffel tower
# [784], book

# Split into x-train/test, and y-train/test

# Fit to model

# Win
# print(data[0])









model = RFC(100)

model.fit