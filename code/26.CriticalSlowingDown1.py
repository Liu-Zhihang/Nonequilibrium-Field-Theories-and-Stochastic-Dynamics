import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Simulation Parameters ---
N = 100  # Lattice size (N x N)
# Critical temperature for 2D Ising model
Tc = 2 / np.log(1 + np.sqrt(2))  # ~2.269

# We will compare two scenarios
T_far = 1.5  # Temperature far below Tc (fast relaxation)
T_near = 2.3 # Temperature very close to Tc (critical slowing down)

n_frames = 200  # Number of frames in the animation
mc_steps_per_frame = 10  # Monte Carlo steps between each frame

# --- 2. Core Ising Model Functions ---
def initial_state(N):
    """Generates a random spin configuration."""
    return np.random.choice([-1, 1], size=(N, N))

def metropolis_step(config, T):
    """Performs one Monte Carlo step using the Metropolis algorithm."""
    for _ in range(N * N):
        # 1. Pick a random spin
        x, y = np.random.randint(0, N, size=2)
        spin = config[x, y]
        
        # 2. Calculate energy change if flipped
        # Periodic boundary conditions are used here
        neighbors = config[(x+1)%N, y] + config[x, (y+1)%N] + \
                    config[(x-1+N)%N, y] + config[x, (y-1+N)%N]
        delta_E = 2 * spin * neighbors  # J=1, k_B=1
        
        # 3. Flip spin based on Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            config[x, y] = -spin
    return config

# --- 3. Setup the Animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Initialize two separate spin configurations
config_far = initial_state(N)
config_near = config_far.copy()  # Start from the exact same random state

# Setup for the "Far from Tc" plot
ax1.set_title(f'Far from Critical Point: T = {T_far:.2f} < Tc')
im1 = ax1.imshow(config_far, cmap='binary', animated=True)
ax1.set_xticks([])
ax1.set_yticks([])

# Setup for the "Near Tc" plot
ax2.set_title(f'Near Critical Point: T = {T_near:.2f} ≈ Tc')
im2 = ax2.imshow(config_near, cmap='binary', animated=True)
ax2.set_xticks([])
ax2.set_yticks([])

fig.suptitle('Ising Model Quench: Visualizing Critical Slowing Down', fontsize=16)

# --- 4. Animation Update Function ---
def update(frame):
    """This function is called for each frame of the animation."""
    global config_far, config_near
    
    # Perform Monte Carlo steps for both systems
    for _ in range(mc_steps_per_frame):
        config_far = metropolis_step(config_far, T_far)
        config_near = metropolis_step(config_near, T_near)
    
    # Update the plots
    im1.set_data(config_far)
    im2.set_data(config_near)
    
    # Print progress
    if (frame + 1) % 20 == 0:
        print(f'Animating frame {frame + 1}/{n_frames}...')
        
    return im1, im2

# --- 5. Create and Save the Animation ---
print("Generating animation... this may take a minute.")
ani = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=50, blit=True, repeat=True
)

# Save the animation to a file instead of showing it
ani.save('critical_slowing_down.gif', writer='pillow', fps=10)
print("Animation saved as 'critical_slowing_down.gif'")