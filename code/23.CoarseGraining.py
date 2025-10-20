# -*- coding: utf-8 -*-
"""
Python script to simulate interacting Brownian particles (Langevin dynamics)
and visualize their microscopic motion alongside the coarse-grained macroscopic density field.
The final output is a GIF animation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

# --- 1. Simulation Parameters ---

# System parameters
N = 100          # Number of particles
L = 20.0         # Size of the simulation box (2D)
T = 1.0          # Temperature (we set Boltzmann constant k_B=1)
mu = 1.0         # Mobility
D = mu * T       # Diffusion coefficient, from Einstein relation

# Simulation parameters
dt = 0.02        # Timestep length
n_steps = 1000   # Total simulation steps
save_interval = 10 # Save trajectory every 'save_interval' steps for animation

# Interaction parameters (Weeks-Chandler-Andersen potential, a purely repulsive force)
sigma = 1.0      # Characteristic size of a particle
epsilon = 1.0    # Interaction strength
rcut = sigma * (2**(1/6)) # Cutoff distance for the force

# Visualization parameters
grid_bins = 50   # Number of bins for the density grid
blur_sigma = 1.5 # Sigma for the Gaussian filter to smooth the density field


# --- 2. Core Functions ---

def calculate_forces(positions, box_size, eps, sig, r_cut):
    """
    Calculates the total force on each particle using the WCA potential
    and applies periodic boundary conditions (minimum image convention).
    """
    forces = np.zeros_like(positions)
    r_cut2 = r_cut**2
    
    for i in range(N):
        for j in range(i + 1, N):
            # Displacement vector between particle i and j
            dr = positions[i] - positions[j]
            
            # Apply periodic boundary conditions
            dr = dr - box_size * np.round(dr / box_size)
            
            r_sq = np.sum(dr**2)
            
            # Calculate force only if particles are closer than the cutoff distance
            if r_sq < r_cut2:
                r_sq_inv = 1.0 / r_sq
                sig_r6 = (sig**2 * r_sq_inv)**3
                
                # Force magnitude from the derivative of the WCA potential
                force_mag = 48 * eps * r_sq_inv * (sig_r6**2 - 0.5 * sig_r6)
                force_vec = force_mag * dr
                
                # Apply force according to Newton's third law
                forces[i] += force_vec
                forces[j] -= force_vec
                
    return forces

# --- 3. Initialization ---

# Set a random seed for reproducibility
np.random.seed(42)

# Initialize particle positions randomly within the box [0, L] x [0, L]
pos = np.random.rand(N, 2) * L

# --- 4. Run Simulation & Store Trajectory ---

print("Running simulation to generate trajectory...")
# Store the trajectory for the animation
# We pre-calculate the trajectory to make the animation rendering smoother
trajectory = [pos.copy()]
num_frames = n_steps // save_interval

for step in range(n_steps):
    # Calculate deterministic forces
    F = calculate_forces(pos, L, epsilon, sigma, rcut)
    
    # Calculate drift and random kick terms
    drift = mu * F * dt
    random_kick = np.sqrt(2 * D * dt) * np.random.randn(N, 2)
    
    # Update particle positions
    pos += drift + random_kick
    
    # Enforce periodic boundary conditions on positions
    pos = pos % L
    
    # Store the current frame in the trajectory
    if (step + 1) % save_interval == 0:
        trajectory.append(pos.copy())

trajectory = np.array(trajectory)
print(f"Simulation finished. Trajectory shape: {trajectory.shape}")


# --- 5. Animation Setup ---

print("Setting up animation...")
# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Setup for the left subplot (Microscopic View)
ax1.set_title('Microscopic View: Particle Dynamics')
ax1.set_xlabel('X position')
ax1.set_ylabel('Y position')
ax1.set_xlim(0, L)
ax1.set_ylim(0, L)
ax1.set_aspect('equal', adjustable='box')
scatter = ax1.scatter(trajectory[0, :, 0], trajectory[0, :, 1], s=50, c='royalblue')

# Setup for the right subplot (Macroscopic View)
ax2.set_title('Macroscopic View: Coarse-Grained Density')
ax2.set_xlabel('X position')
ax2.set_ylabel('Y position')
ax2.set_aspect('equal', adjustable='box')

# Create the grid for the heatmap
grid_x = np.linspace(0, L, grid_bins)
grid_y = np.linspace(0, L, grid_bins)

# Initial density field
hist, _, _ = np.histogram2d(
    trajectory[0, :, 0], trajectory[0, :, 1],
    bins=[grid_x, grid_y]
)
density_field = gaussian_filter(hist.T, sigma=blur_sigma)
heatmap = ax2.imshow(density_field, origin='lower', extent=[0, L, 0, L], cmap='viridis')
fig.colorbar(heatmap, ax=ax2, label='Particle Density')

# Add a time display
time_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, ha='left', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))


# --- 6. Animation Update Function ---

def update(frame):
    """
    This function is called for each frame of the animation.
    """
    # Get current positions from the pre-computed trajectory
    current_pos = trajectory[frame]
    
    # --- Update Microscopic View ---
    scatter.set_offsets(current_pos)
    
    # --- Update Macroscopic View ---
    # 1. Coarse-graining: create a histogram
    hist, _, _ = np.histogram2d(
        current_pos[:, 0], current_pos[:, 1],
        bins=[grid_x, grid_y]
    )
    # 2. Smoothing: apply Gaussian filter
    density_field = gaussian_filter(hist.T, sigma=blur_sigma)
    
    # 3. Update heatmap data
    heatmap.set_data(density_field)
    
    # Update the color limits to match the new data range for better contrast
    heatmap.set_clim(vmin=density_field.min(), vmax=density_field.max())
    
    # Update time text
    time_text.set_text(f'Step: {frame * save_interval}')
    
    return scatter, heatmap, time_text


# --- 7. Create and Save Animation ---

print("Creating and saving animation... This may take a few minutes.")
# Create the animation object

anim = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50, # milliseconds between frames
    blit=True
)

# Save the animation as a GIF
try:
    anim.save('particle_dynamics.gif', writer='imagemagick', fps=15)
    print("\nAnimation successfully saved as 'particle_dynamics.gif'!")
except Exception as e:
    print(f"\nError saving GIF: {e}")



