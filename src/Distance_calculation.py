import pandas as pd
import numpy as np
import sys
from scipy.spatial import cKDTree

# Get iteration number from the command line arguments
iteration = int(sys.argv[1])

# Step 1: Parse LAMMPS Data File
def parse_lammps_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data_start = lines.index('ITEM: ATOMS id type x y z radius\n') + 1
    data_end = len(lines)
    
    data = [line.split() for line in lines[data_start:data_end] if line.strip()]
    df = pd.DataFrame(data, columns=['ID', 'type', 'x', 'y', 'z', 'radius'])
    df[['ID', 'type']] = df[['ID', 'type']].astype(int)
    df[['x', 'y', 'z', 'radius']] = df[['x', 'y', 'z', 'radius']].astype(float)
    
    return df

# Step 2: Calculate Distances with k-d Tree
def calculate_distances_with_kdtree(df, cutoff_distance):
    # Filter out atoms of type 2
    df_type2 = df[df['type'] == 2].reset_index(drop=True)
    
    positions = df_type2[['x', 'y', 'z']].values  # Extract positions for type 2 atoms
    radii = df_type2['radius'].values  # Extract radii for type 2 atoms
    num_atoms = len(df_type2)
    
    # Construct the k-d tree with type 2 atom positions
    kdtree = cKDTree(positions)
    
    # Find all pairs of type 2 atoms within the cutoff distance
    neighbors = kdtree.query_ball_tree(kdtree, cutoff_distance)
    
    # Initialize a distance matrix with infinity values
    distances = np.full((num_atoms, num_atoms), np.inf, dtype=np.float64)
    
    # Calculate distances for each pair of neighboring type 2 atoms
    for i, neighbors_i in enumerate(neighbors):
        for j in neighbors_i:
            if i < j:  # To avoid duplicate calculations
                distance_between_centers = np.linalg.norm(positions[i] - positions[j])
                adjusted_distance = distance_between_centers - (radii[i] + radii[j])
                distances[i][j] = adjusted_distance
                distances[j][i] = adjusted_distance
    
    return distances, df_type2

# Main Execution
input_filename = f"combined_output_{iteration}.lammps"
output_distances_file = 'distances.npy'
output_data_file = 'data_df.csv'
cutoff_distance = 120.0  

data_df = parse_lammps_data(input_filename)
distances, type2_df = calculate_distances_with_kdtree(data_df, cutoff_distance)

np.save(output_distances_file, distances)
type2_df.to_csv(output_data_file, index=False)
