import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from matplotlib.animation import FuncAnimation, PillowWriter

# --- 1. Parameter Settings ---
grid_size = 128
# Reaction parameters
lambda_death = 1.0
gamma_coag = 0.1
mu_active = 1.15   # Active phase (slightly increased for clearer pattern)
mu_absorbing = 0.95 # Absorbing phase
# Simulation parameters
D_space = 0.2      # Spatial diffusion coefficient (horizontal)
v_y = 15.0         
dt = 0.01
total_time = 10
n_steps = int(total_time / dt)

def run_directed_spde_simulation(mu_rate):
    """ Numerically simulate the "directed" SPDE """
    # Initialization: Activate a thin line at the top
    n_field = np.zeros((grid_size, grid_size))
    start_row = 5
    n_field[start_row, grid_size//2 - 10 : grid_size//2 + 10] = 1.0
    
    history = [n_field.copy()]
    # Spatial step size
    dy = 1.0 

    for i in range(n_steps):
        # Reaction term
        reaction_drift = (mu_rate - lambda_death) * n_field - gamma_coag * n_field * (n_field - 1)
        
        # Noise term
        noise_strength_sq = (mu_rate + lambda_death) * n_field + gamma_coag * n_field * (n_field - 1)
        noise_strength = np.sqrt(np.maximum(0, noise_strength_sq))
        space_time_noise = np.random.normal(0, 1, (grid_size, grid_size)) * noise_strength / np.sqrt(dt)

        # Spatial diffusion term (Laplacian operator)
        laplacian = laplace(n_field, mode='wrap') # Periodic boundary conditions
        
        # !!! New: Advection term (using simple upwind scheme) !!!
        # np.roll(n_field, 1, axis=0) shifts entire rows down by one, simulating flow from above
        advection = -v_y * (n_field - np.roll(n_field, 1, axis=0)) / dy
        
        # Euler method to update field
        n_field += (reaction_drift + D_space * laplacian + advection + space_time_noise) * dt
        n_field = np.maximum(0, n_field) # Density cannot be negative
        
        # Save a frame every few steps
        if (i+1) % 2 == 0: 
            history.append(n_field.copy())
            
    return history

# --- 2. Run simulations for two scenarios ---
print("Running simulation for active phase...")
history_active = run_directed_spde_simulation(mu_active)
print("Running simulation for absorbing phase...")
history_absorbing = run_directed_spde_simulation(mu_absorbing)

# --- 3. Create animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor('black')
vmax = np.max(history_active) * 0.7 # Unified color range

def update(frame):
    ax1.clear()
    im1 = ax1.imshow(history_active[frame], cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title(f'Active Phase (μ={mu_active})', color='white')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.clear()
    im2 = ax2.imshow(history_absorbing[frame], cmap='hot', vmin=0, vmax=vmax)
    ax2.set_title(f'Absorbing Phase (μ={mu_absorbing})', color='white')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    fig.suptitle(f'Field Simulation of Directed Percolation (Time: {frame*dt*2:.2f})', color='white', fontsize=16)
    
    # Force aspect ratio to stretch image, better reflecting "directed" nature
    ax1.set_aspect(0.5)
    ax2.set_aspect(0.5)
    
    return [im1, im2]

# --- 4. Generate and save animation ---
frames_to_render = min(len(history_active), len(history_absorbing))
ani = FuncAnimation(fig, update, frames=frames_to_render, interval=40, blit=False)
ani.save("directed_percolation_field.gif", writer=PillowWriter(fps=25))
plt.show()