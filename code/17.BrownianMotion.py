import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import shutil
import warnings


plt.style.use('dark_background')
warnings.filterwarnings("ignore", category=UserWarning, module='imageio')

# --- Parameter settings ---

m = 1.0
xi = 1.0         # Friction rate -> Relaxation time tau = 1/xi = 1.0s
kBT = 1.0        # Thermal energy, set to 1
gamma = m * xi
Gamma = gamma * kBT
total_time = 100.0
dt = 0.01

# --- General simulation parameters ---
num_particles = 500
num_steps = int(total_time / dt)
dimensions = 2

# --- 2. Initialization and simulation execution ---
positions = np.zeros((num_particles, num_steps + 1, dimensions))
velocities = np.zeros((num_particles, num_steps + 1, dimensions))

# Draw initial velocities from Gaussian distribution according to equipartition theorem
v_std = np.sqrt(kBT / m)
velocities[:, 0, :] = np.random.normal(0, v_std, (num_particles, dimensions))

# Pre-calculate noise term amplitude for efficiency
noise_amp = np.sqrt(2 * Gamma * dt) / m

print("Starting particle motion simulation...")
for i in range(num_steps):
    random_force = np.random.normal(0, 1, (num_particles, dimensions))
    velocities[:, i+1, :] = velocities[:, i, :] * (1 - xi * dt) + noise_amp * random_force
    positions[:, i+1, :] = positions[:, i, :] + velocities[:, i, :] * dt
print("Simulation complete.")

# --- 3. Generate GIF animation of Brownian motion for many particles ---
print("Generating Brownian motion GIF animation...")
gif_filename = 'brownian_motion.gif'
temp_dir = 'temp_frames'
if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

frame_step = max(1, int(num_steps / 150))
filenames = []
max_pos_dynamic = np.max(np.abs(positions)) * 1.1

for i in range(0, num_steps + 1, frame_step):
    fig, ax = plt.subplots(figsize=(8, 8))
    dist_from_origin = np.linalg.norm(positions[:, i, :], axis=1)
    ax.scatter(positions[:, i, 0], positions[:, i, 1], s=5, alpha=0.6, cmap='viridis', c=dist_from_origin)
    ax.set_title(f'Brownian Motion of {num_particles} Particles\nTime: {i*dt:.2e} s', color='white', fontsize=14)
    ax.set_xlabel('x position (m)', color='white')
    ax.set_ylabel('y position (m)', color='white')
    ax.set_xlim(-max_pos_dynamic, max_pos_dynamic)
    ax.set_ylim(-max_pos_dynamic, max_pos_dynamic)
    ax.set_aspect('equal', adjustable='box')
    filename = f'{temp_dir}/frame_{i:05d}.png'
    filenames.append(filename)
    plt.savefig(filename, dpi=100)
    plt.close()

with imageio.get_writer(gif_filename, mode='I', duration=0.08, loop=0) as writer:
    for filename in filenames:
        writer.append_data(imageio.imread(filename))
shutil.rmtree(temp_dir)
print(f"Animation saved to '{gif_filename}'")

# --- 4. Calculate and plot key MSD statistics ---
print("Calculating and plotting MSD statistics...")
sq_displacements = np.sum(positions**2, axis=2)
msd = np.mean(sq_displacements, axis=0)
time_axis = np.arange(num_steps + 1) * dt
fig_msd, ax_msd = plt.subplots(figsize=(10, 7))
ax_msd.loglog(time_axis[1:], msd[1:], 'o', color='#00A0FF', markersize=3, alpha=0.8, label='Simulated MSD')

tau_relax = 1/xi
ballistic_indices = (time_axis > 0) & (time_axis < 0.1 * tau_relax)
ballistic_line = (dimensions * kBT / m) * time_axis[ballistic_indices]**2
ax_msd.loglog(time_axis[ballistic_indices], ballistic_line, 'r--', lw=2.5, label=r'Ballistic Regime ($\propto t^2$)')

diffusive_indices = time_axis > 10 * tau_relax
D_eff = kBT / gamma
diffusive_line = 2 * dimensions * D_eff * time_axis[diffusive_indices]
ax_msd.loglog(time_axis[diffusive_indices], diffusive_line, 'g--', lw=2.5, label=r'Diffusive Regime ($\propto t$)')

ax_msd.axvline(x=tau_relax, color='purple', linestyle=':', lw=2, label=f'Relaxation Time $\\tau = 1/\\xi = {tau_relax:.2e}$ s')
ax_msd.set_title("Mean Squared Displacement (MSD) vs. Time", color='white', fontsize=16)
ax_msd.set_xlabel("Time (s)", color='white', fontsize=12)
ax_msd.set_ylabel("MSD (m$^2$)", color='white', fontsize=12)
ax_msd.legend(facecolor='gray', framealpha=0.2, labelcolor='white')
ax_msd.grid(True, which="both", ls="--", color='gray', alpha=0.5)
plt.tight_layout()
plt.savefig("msd_plot.png", dpi=150)
plt.show()
print(f"MSD image saved to 'msd_plot.png'")