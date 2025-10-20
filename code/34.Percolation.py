import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

# --- Create custom colormaps ---
# For absorbing phase: black for inactive (0), yellow for active (1)
abs_colors = ['black', 'yellow']
abs_cmap = ListedColormap(abs_colors)

# For active phase: black for inactive (0), red for active (1)
act_colors = ['black', 'red']
act_cmap = ListedColormap(act_colors)

# --- Use dark background for plots ---
plt.style.use('dark_background')

def run_dp_simulation(grid_size, p_spread, p_death, steps):
    """Runs a single 2D Directed Percolation (Contact Process) simulation."""
    # Initialize grid with a small active cluster in the center
    grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    grid[center-1:center+1, center-1:center+1] = 1
    
    history = [grid.copy()]
    
    for _ in range(steps):
        new_grid = grid.copy()
        active_sites = np.argwhere(grid == 1)
        
        if active_sites.size == 0: # Extinction
            for _ in range(steps - len(history) + 1):
                history.append(new_grid.copy())
            break

        for r, c in active_sites:
            # Spread to neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    # Check boundaries and if neighbor is inactive
                    if 0 <= nr < grid_size and 0 <= nc < grid_size and new_grid[nr, nc] == 0:
                        if np.random.rand() < p_spread:
                            new_grid[nr, nc] = 1
            
            # Spontaneous death
            if np.random.rand() < p_death:
                new_grid[r, c] = 0
        
        grid = new_grid
        history.append(grid.copy())
        
    return history

# --- Simulation Parameters ---
GRID_SIZE = 100
ANIMATION_STEPS = 200

# --- Case 1: Absorbing Phase (low spread probability, high death probability) ---
p_spread_abs = 0.05
p_death_abs = 0.22
history_abs = run_dp_simulation(GRID_SIZE, p_spread_abs, p_death_abs, ANIMATION_STEPS)

# --- Case 2: Active Phase (high spread probability, lower death probability) ---
p_spread_act = 0.1
p_death_act = 0.2
history_act = run_dp_simulation(GRID_SIZE, p_spread_act, p_death_act, ANIMATION_STEPS)

# --- Create Animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

def update(frame):
    ax1.clear()
    ax1.imshow(history_abs[frame], cmap=abs_cmap, vmin=0, vmax=1)
    ax1.set_title(f'Absorbing Phase (p_spread={p_spread_abs})\nSystem dies out', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.clear()
    ax2.imshow(history_act[frame], cmap=act_cmap, vmin=0, vmax=1)
    ax2.set_title(f'Active Phase (p_spread={p_spread_act})\nSystem remains active', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    fig.suptitle(f'Directed Percolation Simulation (Step {frame})\n'
                 f'Left: Absorbing phase with yellow active clusters eventually reaches all-black absorbing state\n'
                 f'Right: Active phase with red active sites sustains dynamic patterns', 
                 fontsize=14)

# --- Generate and save the animation ---
ani = FuncAnimation(fig, update, frames=ANIMATION_STEPS, interval=50)

# Try to save the animation, but handle exceptions gracefully
try:
    ani.save("dp_lattice_simulation.gif", writer=PillowWriter(fps=20))
    print("Animation saved successfully!")
except Exception as e:
    print(f"Failed to save animation: {e}")
    print("Displaying animation without saving...")

plt.show()