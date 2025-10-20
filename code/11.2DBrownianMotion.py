import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Simulation parameters ---
n_walkers = 10      # Number of particles to simulate
n_steps = 1000      # Number of steps for each particle
dt = 0.01           # Time step size
D = 1.0             # Diffusion coefficient

# --- Simulation process ---
# Generate random displacements for all steps
# Shape is (n_walkers, n_steps, 2), representing (particle, step, x/y coordinates)
# np.random.randn generates standard normal distribution random numbers
random_displacements = np.random.randn(n_walkers, n_steps, 2) * np.sqrt(2 * D * dt)

# Calculate trajectories for each particle
# np.cumsum(..., axis=1) performs cumulative sum along the time step dimension
# Initial position is (0,0)
trajectories = np.cumsum(random_displacements, axis=1)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(8, 8))

# Plot trajectory of each particle
for i in range(n_walkers):
    # trajectories[i, :, 0] is all x coordinates of particle i
    # trajectories[i, :, 1] is all y coordinates of particle i
    ax.plot(trajectories[i, :, 0], trajectories[i, :, 1], alpha=0.7)

# Mark start and end points
start_points = np.zeros((n_walkers, 2))
end_points = trajectories[:, -1, :]  # Take the last point of each trajectory
ax.scatter(start_points[:, 0], start_points[:, 1], color='red', s=100, zorder=3, label='Start')
ax.scatter(end_points[:, 0], end_points[:, 1], color='green', s=50, zorder=3, label='End')

ax.set_title(f'2D Brownian Motion ({n_walkers} walkers, {n_steps} steps)')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
plt.show()

# --- Plot distribution of end points ---
fig_hist, ax_hist = plt.subplots(figsize=(10, 8))

# Increase number of particles to better show Gaussian distribution
n_walkers_hist = 1000  # Increased number of particles
random_displacements_hist = np.random.randn(n_walkers_hist, n_steps, 2) * np.sqrt(2 * D * dt)
trajectories_hist = np.cumsum(random_displacements_hist, axis=1)
end_points_hist = trajectories_hist[:, -1, :]

# Use more bins and better color mapping to show distribution
hist_plot = ax_hist.hist2d(end_points_hist[:, 0], end_points_hist[:, 1], bins=50, cmap='hot')
ax_hist.set_title('Distribution of End Points\n(2D Gaussian Distribution)', fontsize=16)
ax_hist.set_xlabel('Final X Position', fontsize=12)
ax_hist.set_ylabel('Final Y Position', fontsize=12)
ax_hist.set_aspect('equal', adjustable='box')

# Add theoretical contour lines
x_min, x_max = np.min(end_points_hist[:, 0]), np.max(end_points_hist[:, 0])
y_min, y_max = np.min(end_points_hist[:, 1]), np.max(end_points_hist[:, 1])
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
# Theoretical variance
sigma_squared = 2 * D * n_steps * dt
Z = np.exp(-(X**2 + Y**2) / (2 * sigma_squared))
ax_hist.contour(X, Y, Z, levels=5, colors='blue', alpha=0.6, linewidths=1)

# Add color bar
cbar = plt.colorbar(hist_plot[3], ax=ax_hist, label='Number of walkers')
cbar.set_label('Number of walkers', fontsize=12)

plt.tight_layout()
plt.show()

# --- Plot 1D projection distributions ---
fig_proj, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 5))

# X-direction projection
ax_x.hist(end_points_hist[:, 0], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
# Theoretical X-direction distribution
sigma = np.sqrt(2 * D * n_steps * dt)
x_range = np.linspace(np.min(end_points_hist[:, 0]), np.max(end_points_hist[:, 0]), 100)
gaussian_x = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-x_range**2/(2*sigma**2))
ax_x.plot(x_range, gaussian_x, 'r-', linewidth=2, label=f'Theoretical Gaussian\nσ² = {sigma**2:.2f}')
ax_x.set_xlabel('X Position')
ax_x.set_ylabel('Probability Density')
ax_x.set_title('X Distribution')
ax_x.legend()
ax_x.grid(True, alpha=0.3)

# Y-direction projection
ax_y.hist(end_points_hist[:, 1], bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
ax_y.plot(x_range, gaussian_x, 'r-', linewidth=2, label=f'Theoretical Gaussian\nσ² = {sigma**2:.2f}')
ax_y.set_xlabel('Y Position')
ax_y.set_ylabel('Probability Density')
ax_y.set_title('Y Distribution')
ax_y.legend()
ax_y.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Create Brownian motion animation ---
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

# Set animation parameters
n_frames = 200  # Number of animation frames
step_interval = n_steps // n_frames  # Steps between each frame

# Initialize animation elements
lines = [ax_anim.plot([], [], alpha=0.7)[0] for _ in range(n_walkers)]
points = ax_anim.scatter([], [], color='red', s=50, zorder=3)

# Set axes
ax_anim.set_xlim(np.min(trajectories[:, :, 0]), np.max(trajectories[:, :, 0]))
ax_anim.set_ylim(np.min(trajectories[:, :, 1]), np.max(trajectories[:, :, 1]))
ax_anim.set_title('2D Brownian Motion Animation')
ax_anim.set_xlabel('X Position')
ax_anim.set_ylabel('Y Position')
ax_anim.grid(True)
ax_anim.set_aspect('equal', adjustable='box')

def animate(frame):
    # Calculate which step to display for current frame
    step = min(frame * step_interval, n_steps - 1)
    
    # Update each trajectory
    for i in range(n_walkers):
        lines[i].set_data(trajectories[i, :step, 0], trajectories[i, :step, 1])
    
    # Update current point positions
    current_points = trajectories[:, step, :]
    points.set_offsets(current_points)
    
    ax_anim.set_title(f'2D Brownian Motion (Step {step}/{n_steps})')
    return lines + [points]

# Create animation
anim = FuncAnimation(fig_anim, animate, frames=n_frames, interval=50, blit=False, repeat=True)
plt.show()

# Save animation as GIF
try:
    import imageio
    import io
    from PIL import Image
    
    print("Saving Brownian motion animation as GIF file...")
    
    # Create temporary image list
    images = []
    
    # Generate animation frames
    for frame in range(0, n_frames, 2):  # Take every other frame to reduce file size
        animate(frame)
        fig_anim.canvas.draw()
        
        # Convert figure to image
        buf = io.BytesIO()
        fig_anim.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        images.append(img)
        
        if frame % 20 == 0:
            print(f"Processed {frame}/{n_frames} frames")
    
    # Save as GIF
    images[0].save('brownian_motion.gif', save_all=True, append_images=images[1:], 
                   duration=100, loop=0)
    print("Animation saved as brownian_motion.gif")
    
except ImportError:
    print("Missing libraries required to save GIF (imageio or PIL), please install: pip install imageio pillow")

plt.show()