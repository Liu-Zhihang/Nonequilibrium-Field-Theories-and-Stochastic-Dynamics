import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm

# --- 1. Parameter settings ---
L = 10.0      # System domain side length
Nx = 128      # Number of spatial grid points (same in x and y directions)
dx = L / Nx   # Spatial step size

# --- Simulation total time T=20 ---
T = 20.0
# --- Correspondingly reduce total steps ---
Nt = 8000
dt = T / Nt

# Model parameters
v0 = 1.0
g_star = 2.0
g0 = 1.2      # Average density
M = 1.0       # Mobility
kappa = 1e-4  # Surface tension coefficient

# --- 2. Define relevant functions ---
def mu_bulk_func(g, v0, g_star):
    term1 = g
    term2 = -(3.0 / (2.0 * g_star)) * g**2
    term3 = (2.0 / (3.0 * g_star**2)) * g**3
    return v0**2 * (term1 + term2 + term3)

# --- 3. Initialize density field ---
g = g0 + 0.05 * (np.random.rand(Nx, Nx) - 0.5)

# Frames for storing GIF and statistical plots
frames = []

# --- 4. Main simulation loop ---
for n in tqdm(range(Nt), desc=f"Simulating 2D MIPS (T={T})"):
    g_up, g_down = np.roll(g, -1, axis=0), np.roll(g, 1, axis=0)
    g_left, g_right = np.roll(g, -1, axis=1), np.roll(g, 1, axis=1)
    lap_g = (g_up + g_down + g_left + g_right - 4*g) / (dx**2)
    
    mu_bulk = mu_bulk_func(g, v0, g_star)
    mu = mu_bulk - kappa * lap_g
    
    mu_up, mu_down = np.roll(mu, -1, axis=0), np.roll(mu, 1, axis=0)
    mu_left, mu_right = np.roll(mu, -1, axis=1), np.roll(mu, 1, axis=1)
    lap_mu = (mu_up + mu_down + mu_left + mu_right - 4*mu) / (dx**2)

    g = g + M * dt * lap_mu

    if n % (Nt // 100) == 0:
        frames.append(g.copy())

# --- 5. Create GIF animation ---
print("\nCreating GIF animation...")
temp_dir = 'temp_frames_mips2d_short'
os.makedirs(temp_dir, exist_ok=True)
gif_filename = 'MIPS_2D.gif'
filenames = []

all_data = np.concatenate([frame.flatten() for frame in frames])
vmin = np.percentile(all_data, 0.1)
vmax = np.percentile(all_data, 99.9)

for i, frame_data in enumerate(frames):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(frame_data, cmap='magma', vmin=vmin, vmax=vmax, interpolation='bicubic')
    ax.set_title(f'2D MIPS: Density Field at t = {i / (len(frames)-1) * T:.2f}')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Density g(x,y,t)")
    
    filename = f'{temp_dir}/frame_{i:03d}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close(fig)

gif_duration = 0.2 
with imageio.get_writer(gif_filename, mode='I', duration=gif_duration, loop=0) as writer:
    for filename in tqdm(filenames, desc="Building GIF"):
        image = imageio.imread(filename)
        writer.append_data(image)

print(f'\nAnimation saved as {gif_filename}')

# --- 6. Plot core statistics: density histogram ---
print("Creating statistical histogram plot...")
fig_hist, ax_hist = plt.subplots(figsize=(10, 7))


# Select representative time points to plot
# Change last index from 100 to 99, as list indices range from 0 to 99
snapshot_indices = [0, 10, 30, 99] 
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))

for i, frame_idx in enumerate(snapshot_indices):
    time = frame_idx / (len(frames) - 1) * T # Corrected time calculation
    data = frames[frame_idx].flatten()
    ax_hist.hist(data, bins=100, density=True, alpha=0.7, 
                 label=f't = {time:.1f}', color=colors[i])

ax_hist.set_title('Evolution of Density Distribution during MIPS', fontsize=16)
ax_hist.set_xlabel('Density (g)', fontsize=14)
ax_hist.set_ylabel('Probability Density', fontsize=14)
ax_hist.legend()
ax_hist.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MIPS_density_histogram.png')
plt.show()

print(f'Histogram plot saved as MIPS_density_histogram.png')