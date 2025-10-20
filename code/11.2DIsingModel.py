import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initialize_lattice(N):
    """
    Initialize an N x N lattice with random spin states of +1 or -1.
    """
    return 2 * np.random.randint(2, size=(N, N)) - 1

def calculate_energy_change(lattice, i, j):
    """
    Calculate the energy change caused by flipping the spin at position (i, j).
    Periodic boundary conditions are used.
    """
    N = lattice.shape[0]
    s_current = lattice[i, j]
    
    # Find spin values of the four neighboring sites (top, bottom, left, right)
    s_top = lattice[(i - 1) % N, j]
    s_bottom = lattice[(i + 1) % N, j]
    s_left = lattice[i, (j - 1) % N]
    s_right = lattice[i, (j + 1) % N]
    
    # Energy change dE = E_final - E_initial
    # E_initial = -J * s_current * (s_top + s_bottom + s_left + s_right)
    # E_final = -J * (-s_current) * (s_top + s_bottom + s_left + s_right)
    # dE = E_final - E_initial = 2 * J * s_current * (sum of neighbors)
    # Set J=1
    dE = 2 * s_current * (s_top + s_bottom + s_left + s_right)
    return dE

def metropolis_sweep(lattice, beta):
    """
    Perform one complete Metropolis scan (N*N attempts).
    beta = 1 / (k_B * T)
    """
    N = lattice.shape[0]
    for _ in range(N * N):
        # Randomly select a lattice site
        i, j = np.random.randint(0, N, size=2)
        
        # Calculate energy change
        dE = calculate_energy_change(lattice, i, j)
        
        # Metropolis acceptance/rejection criterion
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1 # Accept flip
            
    return lattice

# --- Simulation parameters ---
N = 50  # Lattice size
T = 1.5 # Temperature (critical temperature Tc is approximately 2.269)
beta = 1.0 / T
n_sweeps = 3000 # Total number of sweeps for simulation
frame_interval = 50 # Record a frame every this many sweeps

# --- Initialization ---
lattice = initialize_lattice(N)

# Set up visualization
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(lattice, cmap='RdYlBu', vmin=-1, vmax=1, interpolation='nearest')
ax.set_title(f'2D Ising Model (T = {T:.2f}, Sweep = 0)', fontsize=16)
ax.set_axis_off()

# Add color bar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Spin', fontsize=14)

# --- Simulation and animation ---
def update(frame):
    global lattice
    # Perform frame_interval sweeps
    for _ in range(frame_interval):
        lattice = metropolis_sweep(lattice, beta)
    im.set_data(lattice)
    ax.set_title(f'2D Ising Model (T = {T:.2f}, Sweep = {frame * frame_interval})', fontsize=16)
    return [im]

# Create animation
ani = FuncAnimation(fig, update, frames=n_sweeps // frame_interval,
                    interval=100, blit=False, repeat=False)

plt.tight_layout()
plt.show()

# Save as GIF file

import imageio
print("Saving animation as GIF file...")
    
    # Re-run simulation to capture all frames
lattice_gif = initialize_lattice(N)
    
    # Create images for GIF
fig_gif, ax_gif = plt.subplots(figsize=(10, 8))
im_gif = ax_gif.imshow(lattice_gif, cmap='RdYlBu', vmin=-1, vmax=1, interpolation='nearest')
ax_gif.set_title(f'2D Ising Model (T = {T:.2f}, Sweep = 0)', fontsize=16)
ax_gif.set_axis_off()
cbar_gif = plt.colorbar(im_gif, ax=ax_gif, shrink=0.8)
cbar_gif.set_label('Spin', fontsize=14)
    
with imageio.get_writer('ising_model.gif', mode='I', duration=0.1, loop=0) as writer:
        # Save initial state
    fig_gif.canvas.draw()
    image_array = np.array(fig_gif.canvas.renderer._renderer)
    writer.append_data(image_array[:, :, :3])
    print(f"Saved 0/{n_sweeps // frame_interval} frames")
        
        # Step through simulation and save each frame
    for i in range(n_sweeps // frame_interval):
        for _ in range(frame_interval):
            lattice_gif = metropolis_sweep(lattice_gif, beta)
        im_gif.set_data(lattice_gif)
        ax_gif.set_title(f'2D Ising Model (T = {T:.2f}, Sweep = {(i+1) * frame_interval})', fontsize=16)
        fig_gif.canvas.draw()
        image_array = np.array(fig_gif.canvas.renderer._renderer)
        writer.append_data(image_array[:, :, :3])
            
        if (i+1) % 10 == 0:
            print(f"Saved {i+1}/{n_sweeps // frame_interval} frames")
    
plt.close(fig_gif)