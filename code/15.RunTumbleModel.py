import numpy as np
import matplotlib.pyplot as plt

def run_tumble_1d_simulation(num_particles=5000, num_steps=4000, v0=20.0, dt=0.01, L=200.0):
    """
    Simulate biased run-and-tumble motion in one-dimensional space.

    Parameters:
    num_particles (int): Number of particles to simulate.
    num_steps (int): Total number of simulation steps.
    v0 (float): Run speed (um/s).
    dt (float): Time step (s).
    L (float): Length of spatial domain (-L/2 to L/2).
    """
    # Initialize particle positions and directions
    # Positions start from a uniform distribution between -1 and 1 to observe aggregation toward center
    positions = np.random.uniform(-1, 1, num_particles)
    # Initial directions randomly +1 or -1
    directions = np.random.choice([1, -1], num_particles)
    
    # Store history of positions for calculating mean and variance
    position_history = np.zeros((num_steps, num_particles))
    
    # Define position-dependent switching rates
    def alpha_plus(x):
        # When x>0, particles moving right are more likely to tumble (alpha_+ increases)
        # When x<0, particles moving right are less likely to tumble (alpha_+ decreases)
        # This creates a drift toward x=0
        return 1.0 * (1 + 2.0 * np.tanh(x / (L/4))) # Base rate 1.0/s
    
    def alpha_minus(x):
        # When x<0, particles moving left are more likely to tumble (alpha_- increases)
        # When x>0, particles moving left are less likely to tumble (alpha_- decreases)
        # This also creates a drift toward x=0
        return 1.0 * (1 - 2.0 * np.tanh(x / (L/4))) # Base rate 1.0/s
    
    for step in range(num_steps):
        position_history[step] = positions
        
        # Calculate tumbling probability at current positions
        # P(tumble) = alpha * dt
        prob_tumble_plus = alpha_plus(positions) * dt
        prob_tumble_minus = alpha_minus(positions) * dt
        
        # Generate random numbers to determine if tumbling occurs
        rand_nums = np.random.rand(num_particles)
        
        # Update directions based on current direction and tumbling probability
        # For particles moving right (directions == 1)
        tumble_indices_plus = (directions == 1) & (rand_nums < prob_tumble_plus)
        directions[tumble_indices_plus] = -1
        # For particles moving left (directions == -1)
        tumble_indices_minus = (directions == -1) & (rand_nums < prob_tumble_minus)
        directions[tumble_indices_minus] = 1
        
        # Update positions
        positions += directions * v0 * dt
        
        # Apply reflecting boundary conditions
        positions[positions > L/2] = L - positions[positions > L/2]
        directions[positions > L/2] *= -1
        positions[positions < -L/2] = -L - positions[positions < -L/2]
        directions[positions < -L/2] *= -1
    
    return position_history

# --- Run simulation and visualize ---
history = run_tumble_1d_simulation()
num_steps, num_particles = history.shape
time_points = np.arange(num_steps) * 0.01

# Calculate evolution of mean and variance over time
mean_pos = np.mean(history, axis=1)
var_pos = np.var(history, axis=1)

# --- Plot three separate figures and save to local files ---

# 1. Plot particle trajectories
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
# Plot more particle trajectories to make the figure more visually appealing
for i in range(20):
    ax.plot(time_points, history[:, i], lw=0.8, alpha=0.7)
ax.set_title('Single Particle Trajectories')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (μm)')
plt.tight_layout()
plt.savefig('particle_trajectories.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Plot particle distribution histograms at different times
fig, ax = plt.subplots(figsize=(10, 6))
time_indices_to_plot = [int(num_steps/10), int(num_steps/3), int(2*num_steps/3), num_steps-1]
for t_idx in time_indices_to_plot:
    time = t_idx * 0.01
    ax.hist(history[t_idx, :], bins=50, density=True, alpha=0.6, label=f't = {time:.1f} s')
ax.set_title('Distribution of Particles')
ax.set_xlabel('Position (μm)')
ax.set_ylabel('Probability Density')
ax.legend()
plt.tight_layout()
plt.savefig('particle_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Plot evolution of mean and variance over time
fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()
p1, = ax.plot(time_points, mean_pos, 'r-', label='Mean Position')
p2, = ax2.plot(time_points, var_pos, 'b-', label='Variance')
ax.set_title('Evolution of Mean and Variance')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean Position (μm)', color='r')
ax2.set_ylabel('Variance (μm²)', color='b')
ax.tick_params(axis='y', labelcolor='r')
ax2.tick_params(axis='y', labelcolor='b')
ax.legend([p1, p2], ['Mean Position', 'Variance'], loc='center right')
plt.tight_layout()
plt.savefig('mean_variance_evolution.png', dpi=300, bbox_inches='tight')
plt.close()