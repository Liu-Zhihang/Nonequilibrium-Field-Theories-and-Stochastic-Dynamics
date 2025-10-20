import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- 1. Set simulation parameters ---
n_particles = 50     # Number of particles to simulate (increased)
n_steps = 5000       # Number of time steps
dt = 0.01            # Length of each time step
D = 0.1              # Diffusion coefficient (controls Brownian motion strength)
gamma = 1.0          # Shear rate (controls flow field strength)

# --- 2. Initialize particle positions ---
# Randomly place particles near the center of the region
np.random.seed(42) # For reproducible results
positions = np.random.randn(n_particles, 2) * 0.5
# Record trajectories of each particle
trajectories = np.zeros((n_particles, n_steps + 1, 2))
trajectories[:, 0, :] = positions

# --- 3. Simulate evolution (Euler-Maruyama method) ---
for i in range(n_steps):
    # Current positions
    r = trajectories[:, i, :]
    # Flow field velocity v(r) = (gamma * y, 0)
    v_flow = np.zeros_like(r)
    v_flow[:, 0] = gamma * r[:, 1]
    
    # Random noise term
    noise = np.random.randn(n_particles, 2)
    
    # Update positions
    # r(t+dt) = r(t) + v_flow(r) * dt + sqrt(2*D*dt) * noise
    trajectories[:, i + 1, :] = r + v_flow * dt + np.sqrt(2 * D * dt) * noise
    
    # Keep particles within bounds to prevent them from leaving the view
    # X boundary
    outbound_x = np.where(trajectories[:, i + 1, 0] > 5)
    trajectories[outbound_x, i + 1, 0] = 5
    outbound_x = np.where(trajectories[:, i + 1, 0] < -5)
    trajectories[outbound_x, i + 1, 0] = -5
    
    # Y boundary
    outbound_y = np.where(trajectories[:, i + 1, 1] > 5)
    trajectories[outbound_y, i + 1, 1] = 5
    outbound_y = np.where(trajectories[:, i + 1, 1] < -5)
    trajectories[outbound_y, i + 1, 1] = -5

# --- 4. Create dynamic visualization ---
fig, ax = plt.subplots(figsize=(10, 8))

# Set black background
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# a) Plot background flow field (Quiver Plot)
x_grid = np.linspace(-5, 5, 20)
y_grid = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_grid, y_grid)
U = gamma * Y  # x-direction velocity
V = np.zeros_like(X) # y-direction velocity

# Create alternating color pattern for flow field arrows
# Alternate between two colors to create moving effect
colors_pattern = np.zeros((X.shape[0], X.shape[1], 4))
colors_pattern[::2, ::2] = [1, 1, 1, 0.7]  # White
colors_pattern[1::2, 1::2] = [1, 1, 1, 0.7]  # White
colors_pattern[::2, 1::2] = [0.5, 0.5, 1, 0.7]  # Blueish
colors_pattern[1::2, ::2] = [0.5, 0.5, 1, 0.7]  # Blueish

# Flatten the color array for use with quiver
C = np.zeros((X.shape[0] * X.shape[1], 4))
C[::2] = [1, 1, 1, 0.7]  # White
C[1::2] = [0.5, 0.5, 1, 0.7]  # Blueish

# Set flow field arrows with alternating colors
quiver = ax.quiver(X, Y, U, V, color=C, alpha=0.7)

# b) Initialize trajectory lines and particle points with comet effect
lines = []
points = []
# Create multiple points for each particle to achieve glowing effect
glow_points = []
colors = plt.cm.plasma(np.linspace(0, 1, n_particles))

# Tail length for comet effect (increased)
tail_length = 100

for i in range(n_particles):
    # Main particle point
    point, = ax.plot([], [], 'o', color=colors[i], markersize=8)
    # Comet tail (longer trajectory)
    line, = ax.plot([], [], color=colors[i], alpha=0.6, linewidth=2)
    # Glowing effect - multiple points with decreasing size and increasing transparency
    glow_set = []
    for j in range(5):
        glow_point, = ax.plot([], [], 'o', color=colors[i], 
                             markersize=8-j*1.2, alpha=0.8-j*0.15)
        glow_set.append(glow_point)
    
    points.append(point)
    lines.append(line)
    glow_points.append(glow_set)

# c) Set figure format
ax.set_title('Brownian Particles in a Shear Flow', fontsize=16, color='white')
ax.set_xlabel('x position', fontsize=12, color='white')
ax.set_ylabel('y position', fontsize=12, color='white')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.grid(True, linestyle='--', alpha=0.3, color='gray')

# Set axis tick colors
ax.tick_params(colors='white')

# d) Animation update function
def update(frame):
    # Animate the flow field by shifting the color pattern
    t = frame % 4  # Cycle through 4 phases
    
    # Create new color pattern based on time
    if t == 0:
        C_new = np.zeros((X.shape[0] * X.shape[1], 4))
        C_new[::2] = [1, 1, 1, 0.7]  # White
        C_new[1::2] = [0.5, 0.5, 1, 0.7]  # Blueish
    elif t == 1:
        C_new = np.zeros((X.shape[0] * X.shape[1], 4))
        C_new[1::4] = [1, 1, 1, 0.7]  # White
        C_new[3::4] = [1, 1, 1, 0.7]  # White
        C_new[::4] = [0.5, 0.5, 1, 0.7]  # Blueish
        C_new[2::4] = [0.5, 0.5, 1, 0.7]  # Blueish
    elif t == 2:
        C_new = np.zeros((X.shape[0] * X.shape[1], 4))
        C_new[::2] = [0.5, 0.5, 1, 0.7]  # Blueish
        C_new[1::2] = [1, 1, 1, 0.7]  # White
    else:  # t == 3
        C_new = np.zeros((X.shape[0] * X.shape[1], 4))
        C_new[1::4] = [0.5, 0.5, 1, 0.7]  # Blueish
        C_new[3::4] = [0.5, 0.5, 1, 0.7]  # Blueish
        C_new[::4] = [1, 1, 1, 0.7]  # White
        C_new[2::4] = [1, 1, 1, 0.7]  # White
    
    # Update quiver colors to create moving effect
    quiver.set_color(C_new)
    
    # Display more trajectory points per frame for smoother animation
    step = max(1, frame * 5)  # Add 5 time steps of data points per frame
    if step > n_steps:
        step = n_steps
    
    for i in range(n_particles):
        # Determine the range for the comet tail
        start_idx = max(0, step - tail_length)
        end_idx = step
        
        # Update comet tail (longer trajectory)
        lines[i].set_data(trajectories[i, start_idx:end_idx, 0], 
                         trajectories[i, start_idx:end_idx, 1])
        
        # Update main particle point position
        points[i].set_data(trajectories[i, step-1, 0], trajectories[i, step-1, 1])
        
        # Update glowing effect points (multiple points trailing behind)
        for j, glow_point in enumerate(glow_points[i]):
            offset = j + 1
            if step > offset:
                glow_point.set_data(trajectories[i, step-1-offset, 0], 
                                   trajectories[i, step-1-offset, 1])
            else:
                glow_point.set_data([], [])  # Hide if not enough history
    
    # Return all animated elements
    flat_glow_points = [item for sublist in glow_points for item in sublist]
    return [quiver] + lines + points + flat_glow_points

# e) Create animation
# To control animation length, we calculate total frames
n_frames = n_steps // 5
ani = FuncAnimation(fig, update, frames=range(0, n_frames, 5), blit=True, interval=30)

# f) Save as GIF
ani.save('brownian_particles_comet.gif', writer=PillowWriter(fps=25))

plt.show()