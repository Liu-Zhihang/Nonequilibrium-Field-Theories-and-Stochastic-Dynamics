import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

# --- 0. Set plotting style ---
plt.style.use('dark_background')

# --- 1. Simulation parameter definition ---
num_particles = 500      # Increase number of particles for better statistics
k = 1.0                  # Harmonic oscillator potential strength (U = 0.5*k*r^2)
kBT = 0.5                # Thermal energy, k_B * T
mu = 1.0                 # Mobility
# Core relationship: Diffusion coefficient determined by fluctuation-dissipation theorem (Einstein relation)
D = mu * kBT             

dt = 0.01                # Time step
num_steps = 1000         # Total simulation steps
simulation_time = dt * num_steps

# --- 2. Initialize particle positions ---
# Initially distribute particles randomly within a disk
initial_radius = 4.0
theta = np.random.uniform(0, 2 * np.pi, num_particles)
r = np.sqrt(np.random.uniform(0, initial_radius**2, num_particles))
positions = np.zeros((num_particles, 2))
positions[:, 0] = r * np.cos(theta)
positions[:, 1] = r * np.sin(theta)

# --- 3. Simulation main loop and GIF generation ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

# Prepare folder to store image frames
output_folder = 'brownian_motion_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Pre-calculate coefficient for random term
noise_strength = np.sqrt(2 * D * dt)

# Store filenames for each frame
frame_files = []

print("Starting to generate GIF animation frames...")
# Simulation loop
for step in range(num_steps):
    # This is the numerical solution of the overdamped Langevin equation (Euler-Maruyama method).
    # This equation describes the trajectory of individual particles, and is the microscopic basis of the Fokker-Planck (Smoluchowski) equation.
    
    # Calculate drift term (from harmonic oscillator potential F = -k*r)
    drift = -mu * k * positions * dt
    
    # Calculate random term (diffusion)
    noise = noise_strength * np.random.randn(num_particles, 2)
    
    # Update positions of all particles
    positions += drift + noise
    
    # Save a frame image every few steps to create GIF
    if step % 10 == 0:
        ax.clear()
        # Use brighter colors to adapt to black background
        ax.scatter(positions[:, 0], positions[:, 1], alpha=0.7, c='cyan', edgecolors='w', s=30, linewidths=0.5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f"Overdamped Brownian Motion (Time = {step*dt:.2f})", fontsize=14)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        
        # Plot contour lines of theoretical equilibrium distribution
        # Theoretical standard deviation sigma^2 = kBT/k
        sigma_theory = np.sqrt(kBT / k)
        circle1 = plt.Circle((0, 0), sigma_theory, color='r', fill=False, linestyle='--', label=r'$1\sigma_{theory}$')
        circle2 = plt.Circle((0, 0), 2*sigma_theory, color='r', fill=False, linestyle=':', label=r'$2\sigma_{theory}$')
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.legend()
        
        filename = os.path.join(output_folder, f'frame_{step//10:04d}.png')
        plt.savefig(filename)
        frame_files.append(filename)

plt.close(fig)
print(f"Generated {len(frame_files)} image frames.")

# --- 4. Use imageio to combine image frames into GIF ---
print("Combining image frames into GIF...")
gif_path = 'overdamped_brownian_motion_dark.gif'
with imageio.get_writer(gif_path, mode='I', fps=20, loop=0) as writer:
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved as '{gif_path}'")

# Clean up temporary image files
for filename in frame_files:
    os.remove(filename)
os.rmdir(output_folder)
print("Cleaned up temporary image files.")


# --- 5. Final distribution analysis: Comparison of simulation results with steady-state solution of Smoluchowski equation ---
print("Plotting final probability distribution...")
fig_dist, ax_dist = plt.subplots(figsize=(10, 8))

# Plot 2D histogram (heatmap) of simulation results
# This represents the particle probability distribution P(x, y) obtained from Langevin dynamics simulation
counts, xedges, yedges, im = ax_dist.hist2d(
    positions[:, 0], positions[:, 1], 
    bins=50, density=True, cmap='inferno'
)
fig_dist.colorbar(im, ax=ax_dist, label='Simulated Probability Density')

# Calculate and plot theoretical steady-state solution
# The steady-state solution of the Smoluchowski equation is the Boltzmann distribution: P(x, y) ~ exp(-U(x,y)/kBT)
# For harmonic oscillator potential U = 0.5*k*(x^2+y^2), this is a 2D Gaussian distribution
sigma_theory = np.sqrt(kBT / k)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = (1.0 / (2 * np.pi * sigma_theory**2)) * np.exp(-(X**2 + Y**2) / (2 * sigma_theory**2))

# Overlay theoretical solution as contour lines on heatmap
ax_dist.contour(X, Y, Z, levels=5, colors='lime', linewidths=1.5, linestyles='--')

# Set chart title and labels
ax_dist.set_title("Final Distribution vs. Smoluchowski Equation Steady-State", fontsize=16)
ax_dist.set_xlabel("x position")
ax.set_aspect('equal')
ax_dist.set_ylabel("y position")

# Create a fake line object for legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='lime', lw=2, linestyle='--', label='Theory (Boltzmann Distribution)')]
ax_dist.legend(handles=legend_elements)
ax_dist.set_aspect('equal', 'box')

final_dist_path = 'final_distribution_comparison.png'
plt.savefig(final_dist_path, dpi=150)
print(f"Final distribution comparison plot saved as '{final_dist_path}'")
plt.show()