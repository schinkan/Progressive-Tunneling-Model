import pandas as pd
import numpy as np
import networkx as nx
import sys
import os

# Get iteration number from the command line arguments
iteration = int(sys.argv[1])

# Step 3: Create Adjacency Matrix with Distances
def create_adjacency_matrix_with_distances(distances, threshold):
    adjacency_matrix = (distances <= threshold).astype(int)
    return adjacency_matrix

# Step 4: Check for Percolation and Extract Bonds for all directions
def check_percolation_and_extract_all_bonds(adjacency_matrix, data_df, surface_threshold=10.0):
    # Filter to include only type 2 elements
    type2_df = data_df[data_df['type'] == 2].reset_index(drop=True)
    num_atoms = len(type2_df)
    print(f"Number of type 2 atoms: {num_atoms}")

    G = nx.Graph()

    # Add nodes
    for i in range(num_atoms):
        G.add_node(i)

    # Add edges
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)
    
    # Define atoms near the surfaces, considering their radii
    min_x, max_x = type2_df['x'].min(), type2_df['x'].max()
    min_y, max_y = type2_df['y'].min(), type2_df['y'].max()
    min_z, max_z = type2_df['z'].min(), type2_df['z'].max()

    surface_min_x = type2_df[type2_df['x'] - type2_df['radius'] <= min_x + surface_threshold].index.tolist()
    surface_max_x = type2_df[type2_df['x'] + type2_df['radius'] >= max_x - surface_threshold].index.tolist()
    surface_min_y = type2_df[type2_df['y'] - type2_df['radius'] <= min_y + surface_threshold].index.tolist()
    surface_max_y = type2_df[type2_df['y'] + type2_df['radius'] >= max_y - surface_threshold].index.tolist()
    surface_min_z = type2_df[type2_df['z'] - type2_df['radius'] <= min_z + surface_threshold].index.tolist()
    surface_max_z = type2_df[type2_df['z'] + type2_df['radius'] >= max_z - surface_threshold].index.tolist()
  
    percolates = [False, False, False]
    bonds_x, bonds_y, bonds_z = [], [], []

    for surface, label, axis_bonds in zip([(surface_min_x, surface_max_x), (surface_min_y, surface_max_y), (surface_min_z, surface_max_z)], 
                                         ['x', 'y', 'z'], [bonds_x, bonds_y, bonds_z]):
        print(f"Checking percolation along {label}-axis")
        if not surface[0] or not surface[1]:
            print(f"No surface atoms found along {label}-axis.")
            continue

        for node1 in surface[0]:
            for node2 in surface[1]:
                if nx.has_path(G, node1, node2):
                    percolates[['x', 'y', 'z'].index(label)] = True
                    path_edges = list(nx.shortest_path(G, source=node1, target=node2))
                    for k in range(len(path_edges) - 1):
                        axis_bonds.append((path_edges[k], path_edges[k+1]))
                    break
            if percolates[['x', 'y', 'z'].index(label)]:
                break

    return percolates, bonds_x, bonds_y, bonds_z

# Output bonds to LAMMPS
def output_to_lammps_with_bonds(data_df, bonds_x, bonds_y, bonds_z, output_filename):
    with open(output_filename, 'w') as f:
        # Write header
        f.write('LAMMPS data file via Python script\n\n')
        f.write(f'{len(data_df)} atoms\n')
        total_bonds = len(bonds_x) + len(bonds_y) + len(bonds_z)
        f.write(f'{total_bonds} bonds\n\n')
        f.write('2 atom types\n')
        f.write('3 bond types\n\n')
        f.write('0 3000.0 xlo xhi\n')
        f.write('0 3000.0 ylo yhi\n')
        f.write('0 3000.0 zlo zhi\n\n')
        
        f.write('Atoms\n\n')
        for idx, row in data_df.iterrows():
            f.write(f'{idx + 1} {row["type"]} {row["x"]} {row["y"]} {row["z"]}\n')

        f.write('\nBonds\n\n')
        bond_id = 1
        for bond in bonds_x:
            f.write(f'{bond_id} 1 {bond[0] + 1} {bond[1] + 1}\n')
            bond_id += 1
        for bond in bonds_y:
            f.write(f'{bond_id} 2 {bond[0] + 1} {bond[1] + 1}\n')
            bond_id += 1
        for bond in bonds_z:
            f.write(f'{bond_id} 3 {bond[0] + 1} {bond[1] + 1}\n')
            bond_id += 1

# Main Execution
input_distances_file = 'distances.npy'
input_data_file = 'data_df.csv'
output_filename = f'rve_percolation_bonds_{iteration}.lammps'
threshold = 20.0  # tunneling distance

distances = np.load(input_distances_file)
data_df = pd.read_csv(input_data_file)
print(f"Loaded distances with shape: {distances.shape}")
print(f"Loaded data_df with shape: {data_df.shape}")

# Filter to include only type 2 elements for percolation check
type2_df = data_df[data_df['type'] == 2].reset_index(drop=True)
if len(type2_df) == 0:
    print("No type 2 elements found.")
    sys.exit()

adjacency_matrix_with_distances = create_adjacency_matrix_with_distances(distances[:len(type2_df), :len(type2_df)], threshold)

# Check for percolation and extract bonds
percolates, bonds_x, bonds_y, bonds_z = check_percolation_and_extract_all_bonds(adjacency_matrix_with_distances, type2_df)

# Create a DataFrame to store the percolation results
percolation_results = pd.DataFrame({
    'iteration': [iteration],
    'x_to_x': [int(percolates[0])],
    'y_to_y': [int(percolates[1])],
    'z_to_z': [int(percolates[2])]
})

# Append the results to a CSV file
percolation_results.to_csv('percolation_results.csv', mode='a', header=not os.path.exists('percolation_results.csv'), index=False)

# If percolation occurs in any direction, output the bonds to the LAMMPS file
if any(percolates):
    output_to_lammps_with_bonds(data_df, bonds_x, bonds_y, bonds_z, output_filename)

print("Percolation checking: Done")
