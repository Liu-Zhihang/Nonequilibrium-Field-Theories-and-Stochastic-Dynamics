import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d

def simulate_spatial_rps_spirals(grid_size, rates, steps):
    """
    Simulates the spatial Rock-Paper-Scissors model.
    
    Args:
        grid_size (int): The width and height of the grid.
        rates (dict): A dictionary of probabilities {'repro', 'pred', 'mobil'}.
        steps (int): The number of animation frames to generate.
        
    Returns:
        list: A history of the grid states for animation.
    """
    # Grid states: 0=Empty, 1=A (Rock/Red), 2=B (Paper/Blue), 3=C (Scissors/Yellow)
    grid = np.random.randint(0, 4, size=(grid_size, grid_size))
    history = [grid.copy()]
    
    # Kernel for counting neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    for step in range(steps):
        # Create a copy to store the changes for this step
        new_grid = grid.copy()
        
        # --- 1. Reaction Phase ---
        # Count neighbors for each species
        neighbors_A = convolve2d((grid == 1), kernel, mode='same', boundary='wrap')
        neighbors_B = convolve2d((grid == 2), kernel, mode='same', boundary='wrap')
        neighbors_C = convolve2d((grid == 3), kernel, mode='same', boundary='wrap')

        # Identify predators for each species
        # Predators: C(3) for A(1), A(1) for B(2), B(2) for C(3)
        predators_for_A = neighbors_C
        predators_for_B = neighbors_A
        predators_for_C = neighbors_B
        
        # Calculate probabilities of events for each cell
        # Predation happens to non-empty cells
        prob_predation_A = rates['pred'] * predators_for_A
        prob_predation_B = rates['pred'] * predators_for_B
        prob_predation_C = rates['pred'] * predators_for_C
        
        # Reproduction happens in empty cells
        prob_repro_A = rates['repro'] * neighbors_A
        prob_repro_B = rates['repro'] * neighbors_B
        prob_repro_C = rates['repro'] * neighbors_C

        # Generate random numbers for all cells at once
        rand_field = np.random.rand(grid_size, grid_size)
        
        # Apply predation rules
        new_grid[(grid == 1) & (rand_field < prob_predation_A)] = 0
        new_grid[(grid == 2) & (rand_field < prob_predation_B)] = 0
        new_grid[(grid == 3) & (rand_field < prob_predation_C)] = 0
        
        # Apply reproduction rules (only to cells that are still empty)
        # Find empty cells and rank reproduction probabilities
        empty_mask = (new_grid == 0)
        repro_chances = np.stack([prob_repro_A, prob_repro_B, prob_repro_C], axis=-1)
        
        # The species with the most neighbors has the best chance to reproduce
        colonizer = np.argmax(repro_chances, axis=-1) + 1
        
        # Check if the max probability is > 0 and beats a random threshold
        max_prob = np.max(repro_chances, axis=-1)
        new_grid[empty_mask & (rand_field < max_prob)] = colonizer[empty_mask & (rand_field < max_prob)]

        # --- 2. Mobility Phase ---
        # A fraction of individuals try to move into empty spaces
        for _ in range(int(grid_size * grid_size * rates['mobil'])):
            r, c = np.random.randint(0, grid_size, 2)
            # Only non-empty sites can move
            if new_grid[r, c] != 0:
                # Pick a random neighbor
                dr, dc = [(0,1), (0,-1), (1,0), (-1,0)][np.random.randint(4)]
                nr, nc = (r + dr) % grid_size, (c + dc) % grid_size
                # If neighbor is empty, swap
                if new_grid[nr, nc] == 0:
                    new_grid[r, c], new_grid[nr, nc] = new_grid[nr, nc], new_grid[r, c]

        grid = new_grid
        history.append(grid.copy())
        
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{steps} completed.")
            
    return history

# --- Simulation Parameters ---
GRID_SIZE = 150
# These rates are chosen to favor spiral formation
RATES = {'repro': 0.1, 'pred': 0.3, 'mobil': 0.05}
ANIMATION_STEPS = 500

# --- Run Simulation ---
print("Starting spatial simulation for spirals...")
simulation_history = simulate_spatial_rps_spirals(GRID_SIZE, RATES, ANIMATION_STEPS)
print("Simulation finished. Creating animation...")

# --- Create and Save Animation ---
fig, ax = plt.subplots(figsize=(8, 8))
cmap = ListedColormap(['black', 'red', 'blue', 'yellow'])
im = ax.imshow(simulation_history[0], cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
ax.set_xticks([])
ax.set_yticks([])
title = ax.set_title("Spatial Rock-Paper-Scissors (Step 0)")

def update(frame):
    im.set_array(simulation_history[frame])
    title.set_text(f"Spatial Rock-Paper-Scissors (Step {frame})")
    return [im, title]

ani = animation.FuncAnimation(fig, update, frames=len(simulation_history), interval=30, blit=True)
ani.save('spatial_rps_spirals.gif', writer='pillow', fps=30)
print("Animation saved as 'spatial_rps_spirals.gif'")
plt.show()