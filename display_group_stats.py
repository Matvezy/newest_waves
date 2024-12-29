import csv
import os
import numpy as np

csv_file = "statistics.csv"

def compute_group_averages():
    if not os.path.isfile(csv_file):
        print("No data file found.")
        return

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        #next(reader)  # Skip the header
        data = np.array([list(map(float, row)) for row in reader])
    
    # Separate data into groups
    group_indices = [data[i::3] for i in range(3)]  # Split data into every 3 rows
    
    group_names = ["Normal", "0.05 Split", "0.1 Split"]
    for idx, group_data in enumerate(group_indices):
        group_average = group_data.mean(axis=0)
        print(f"\n--- Averages for {group_names[idx]} ---")
        print(f"Total images processed (average): {group_average[0]:.2f}")
        print(f"Average green token ratio (overall): {group_average[1]:.4f}")
        print(f"Standard deviation of green token ratio (overall): {group_average[2]:.4f}")
        print(f"Average red token ratio (overall): {group_average[3]:.4f}")
        print(f"Standard deviation of red token ratio (overall): {group_average[4]:.4f}")

compute_group_averages()