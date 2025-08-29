import numpy as np
import sys

# Get iteration number from the command line arguments
iteration_number = int(sys.argv[1])

def calculate_volume_of_sphere(radius):
    return (4/3) * np.pi * radius**3

def generate_element_positions(fcc_atoms, box_size, num_initial_elements, min_radius, max_radius, target_volume_fraction):
    new_positions = []
    existing_positions = fcc_atoms.copy()
    
    total_box_volume = box_size[0] * box_size[1] * box_size[2]
    
    # Add initial elements
    while len(new_positions) < num_initial_elements:
        x = np.random.uniform(0, box_size[0])
        y = np.random.uniform(0, box_size[1])
        z = np.random.uniform(0, box_size[2])
        radius = np.random.uniform(min_radius, max_radius)
        
        if (x > radius) and (x < box_size[0] - radius) and \
           (y > radius) and (y < box_size[1] - radius) and \
           (z > radius) and (z < box_size[2] - radius):
            
            overlap = False
            for pos in existing_positions:
                if np.linalg.norm(np.array([x, y, z]) - np.array(pos[:3])) < (radius + pos[3]):
                    overlap = True
                    break

            if not overlap:
                new_positions.append([x, y, z, radius])
                existing_positions.append([x, y, z, radius])
    
    # Continue adding elements until target volume fraction is met
    current_volume = sum(calculate_volume_of_sphere(atom[3]) for atom in existing_positions)
    current_volume_fraction = current_volume / total_box_volume
    
    while current_volume_fraction < target_volume_fraction:
        x = np.random.uniform(0, box_size[0])
        y = np.random.uniform(0, box_size[1])
        z = np.random.uniform(0, box_size[2])
        radius = np.random.uniform(min_radius, max_radius)
        
        if (x > radius) and (x < box_size[0] - radius) and \
           (y > radius) and (y < box_size[1] - radius) and \
           (z > radius) and (z < box_size[2] - radius):
            
            overlap = False
            for pos in existing_positions:
                if np.linalg.norm(np.array([x, y, z]) - np.array(pos[:3])) < (radius + pos[3]):
                    overlap = True
                    break

            if not overlap:
                new_positions.append([x, y, z, radius])
                existing_positions.append([x, y, z, radius])
                
                # Update current volume fraction
                new_element_volume = calculate_volume_of_sphere(radius)
                current_volume += new_element_volume
                current_volume_fraction = current_volume / total_box_volume

                # Debug print
                print(f"Added element at ({x}, {y}, {z}) with radius {radius:.2f}")
                print(f"Current volume fraction: {current_volume_fraction:.6f}")

    return new_positions

def write_lammps_file(filename, fcc_atoms, new_elements, box_size):
    with open(filename, 'w') as f:
        f.write("ITEM: TIMESTEP\n0\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{len(fcc_atoms) + len(new_elements)}\n")
        f.write(f"ITEM: BOX BOUNDS pp pp pp\n0 {box_size[0]}\n0 {box_size[1]}\n0 {box_size[2]}\n")
        f.write("ITEM: ATOMS id type x y z radius\n")
        
        atom_id = 1
        for atom in fcc_atoms:
            f.write(f"{atom_id} 1 {atom[0]} {atom[1]} {atom[2]} {atom[3]}\n")
            atom_id += 1
        for atom in new_elements:
            f.write(f"{atom_id} 2 {atom[0]} {atom[1]} {atom[2]} {atom[3]}\n")
            atom_id += 1

# Define parameters
box_size = [3000.0, 3000.0, 3000.0]
num_initial_elements = 7878
target_volume_fraction = 0.033  # Example target volume fraction for the second element type
min_radius = 30.0
max_radius = 30.0

# Example FCC atoms data
fcc_atoms = [
    [1500.0, 0.0, 1500.0,     1130], #X
    [0.0, 1500.0, 1500.0,     1130], #Y
    [1500.0, 3000.0, 1500.0,  1130], #X"
    [3000.0, 1500.0, 1500.0,  1130], #Y"
    [1500.0, 1500.0, 0.0,     1130], #Z
    [1500.0, 1500.0, 3000.0,  1130], #Z"
    [0.0, 0.0, 0.0,           1130], #X corners
    [3000.0, 0.0, 0.0,        1130], #XE
    [0.0, 3000.0, 0.0,        1130], #YE
    [3000.0, 3000.0, 0.0,     1130], #XE"
    [0.0, 0.0, 3000.0,        1130], #Z top
    [3000.0, 0.0, 3000.0,     1130], #X
    [0.0, 3000.0, 3000.0,     1130], #Y
    [3000.0, 3000.0, 3000.0,  1130], #Z"
]

# Generate new element positions
new_elements = generate_element_positions(fcc_atoms, box_size, num_initial_elements, min_radius, max_radius, target_volume_fraction)

# Write to LAMMPS file
write_lammps_file(f'combined_output_{iteration_number}.lammps', fcc_atoms, new_elements, box_size)

# Calculate and print the final amount and volume fraction of the second element
total_box_volume = box_size[0] * box_size[1] * box_size[2]
second_element_volume = sum(calculate_volume_of_sphere(atom[3]) for atom in new_elements)
final_volume_fraction = second_element_volume / total_box_volume

print(f"Total number of second elements: {len(new_elements)}")
print(f"Final volume fraction of second elements: {final_volume_fraction:.6f}")
