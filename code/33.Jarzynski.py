import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm # Import tqdm to show progress bar

# --- 1. Physical Parameters ---
k = 2.0
gamma = 1.0
T = 1.0
beta = 1.0 / T

# --- 2. Simulation Parameters ---
dt = 0.01
t_f = 5.0
num_steps = int(t_f / dt)
num_trajectories = 2000

# --- Optimization: Animation Parameters ---
FRAMES = 100 # only generate 100 animation frames
TRAJ_PER_FRAME = num_trajectories // FRAMES # How many trajectories per frame

# --- 3. Protocol Definition ---
lambda_0 = 0.0
lambda_f = 5.0
v_lambda = (lambda_f - lambda_0) / t_f

# --- 4. Initialize Data Storage ---
work_values = np.zeros(num_trajectories)
trajectories = np.zeros((num_trajectories, num_steps + 1))
time_array = np.linspace(0, t_f, num_steps + 1)

# --- 5. Main Simulation Loop (Generate All Data) ---
print("Generating all trajectory data...")
for i in tqdm(range(num_trajectories)): # Use tqdm to show progress
    x = np.random.normal(loc=lambda_0, scale=np.sqrt(T / k))
    trajectories[i, 0] = x
    total_work = 0.0
    
    for step in range(num_steps):
        t = step * dt
        lambda_t = lambda_0 + v_lambda * t
        
        force_on_trap = k * (x - lambda_t)
        dW = -force_on_trap * v_lambda * dt # Corrected work calculation
        total_work += dW
        
        force_on_particle = -k * (x - lambda_t)
        noise_term = np.sqrt(2 * gamma * T * dt) * np.random.randn()
        x += (force_on_particle / gamma) * dt + noise_term / gamma
        trajectories[i, step + 1] = x
        
    work_values[i] = total_work

# Pre-calculate convergence process of Jarzynski average
jarzynski_averages = [np.mean(np.exp(-beta * work_values[:(i+1)*TRAJ_PER_FRAME])) for i in range(FRAMES)]

print("Data generation completed, creating animation...")

# --- 6. Create Visualization ---
plt.style.use('dark_background')  # Set black background
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 2, figure=fig)
ax_physics = fig.add_subplot(gs[:, 0])
ax_hist = fig.add_subplot(gs[0, 1])
ax_conv = fig.add_subplot(gs[1, 1])
fig.suptitle('Jarzynski Equality Dynamic Verification (Optimized Version)', fontsize=20, y=0.95)

# Initialize plot elements for efficient updates
line_traj, = ax_physics.plot([], [], 'r-', lw=1.5, alpha=0.8)
point_traj, = ax_physics.plot([], [], 'ro', markersize=8, label='Particle Trajectory')
line_conv, = ax_conv.plot([], [], 'm-', lw=2)

# --- 7. Animation Update Function (Optimized Version) ---
def update(frame):
    # Calculate total number of trajectories for current frame
    num_traj_so_far = (frame + 1) * TRAJ_PER_FRAME
    
    # --- Update Panel 1: Physical Process (Show only the last trajectory of current batch) ---
    ax_physics.clear() # Physics panel structure is complex, easier to clear
    current_traj_index = num_traj_so_far - 1
    current_traj = trajectories[current_traj_index, :]
    ax_physics.plot(current_traj, time_array, 'r-', lw=1.5, alpha=0.8)
    ax_physics.plot(current_traj[-1], time_array[-1], 'ro', markersize=8, label='Particle Trajectory')
    
    x_range = np.linspace(-5, 10, 200)
    ax_physics.plot(x_range, 0.5 * k * (x_range - lambda_0)**2, 'b--', alpha=0.5, label='Initial Potential Well')
    ax_physics.plot(x_range, 0.5 * k * (x_range - lambda_f)**2, 'b-', alpha=0.8, label='Final Potential Well')
    lambda_t_array = lambda_0 + v_lambda * time_array
    ax_physics.plot(lambda_t_array, time_array, 'k--', lw=1, label='Trap Center Trajectory')
    
    ax_physics.set_xlabel('Position x', fontsize=12)
    ax_physics.set_ylabel('Time t', fontsize=12)
    ax_physics.set_title(f'Physical Process: Trajectory #{num_traj_so_far}', fontsize=14)
    ax_physics.set_xlim(-4, 10)
    ax_physics.set_ylim(0, t_f)
    ax_physics.legend(loc='upper left')
    ax_physics.grid(True, linestyle=':')

    # --- Update Panel 2: Work Distribution Histogram ---
    ax_hist.clear()
    current_works = work_values[:num_traj_so_far]
    ax_hist.hist(current_works, bins=50, density=True, alpha=0.7, color='skyblue', range=(0, np.max(work_values)*1.1))
    mean_work = np.mean(current_works)
    ax_hist.axvline(mean_work, color='r', linestyle='--', lw=2, label=f'<W> = {mean_work:.2f}')
    ax_hist.axvline(0, color='g', linestyle='-', lw=2, label='ΔF = 0')
    ax_hist.set_xlabel('Work W', fontsize=12)
    ax_hist.set_ylabel('Probability Density P(W)', fontsize=12)
    ax_hist.set_title('Non-equilibrium Work Distribution', fontsize=14)
    ax_hist.legend()
    ax_hist.grid(True, linestyle=':')
    
    # --- Update Panel 3: Convergence of Jarzynski Average ---
    ax_conv.clear()
    traj_counts = np.arange(1, frame + 2) * TRAJ_PER_FRAME
    ax_conv.plot(traj_counts, jarzynski_averages[:frame+1], 'm-', lw=2)
    ax_conv.axhline(1.0, color='k', linestyle='--', lw=2, label='Theoretical Value exp(-βΔF) = 1')
    ax_conv.set_xlabel('Number of Trajectories', fontsize=12)
    ax_conv.set_ylabel(r'$\langle e^{-\beta W} \rangle$', fontsize=14)
    ax_conv.set_title('Convergence of Jarzynski Average', fontsize=14)
    ax_conv.set_xscale('log')
    ax_conv.legend(loc='upper right')
    ax_conv.grid(True, which="both", linestyle=':')
    current_avg_val = jarzynski_averages[frame]
    ax_conv.text(0.95, 0.05, f'Current Value: {current_avg_val:.4f}', 
                 transform=ax_conv.transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

# --- 8. Create and Save Animation ---
# Create a progress bar to show animation saving progress
ani_pbar = tqdm(total=FRAMES, desc="Rendering animation")
ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=50, repeat=False)
ani.save('jarzynski_verification_optimized.gif', writer='pillow', fps=20, 
         progress_callback=lambda i, n: ani_pbar.update(1))
ani_pbar.close()
plt.show()
print("Animation 'jarzynski_verification_optimized.gif' saved successfully!")