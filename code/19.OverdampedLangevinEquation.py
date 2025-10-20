import numpy as np
import matplotlib.pyplot as plt

# Use the same physical parameters and potential function as simulation 1
gamma = 1.0
D = 0.25     # Moderate diffusion coefficient, balancing transition rate and rarity
T_total = 40.0  # Further increase total time to ensure complete equilibrium
N_points = 801  # Further increase time resolution
dt = T_total / (N_points - 1)
time = np.linspace(0, T_total, N_points)

def potential(x):
    return (x**2 - 1)**2

def drift(x):
    return -4 * x * (x**2 - 1) / gamma

def drift_prime(x):
    """Derivative of drift term A'(x)"""
    return (-12 * x**2 + 4) / gamma

# Simple calculation of OM instanton path (for plotting)
def get_simple_om_instanton():
    """Calculate simplified OM instanton path for visualization"""
    from scipy.optimize import minimize
    
    def get_action(path, dt):
        x = path
        x_dot = np.diff(x) / dt
        x_mid = (x[:-1] + x[1:]) / 2
        A = drift(x_mid)
        A_prime = drift_prime(x_mid)
        lagrangian_fw = (x_dot - A)**2 / (4 * D)
        lagrangian_om_corr = 0.5 * A_prime
        action = np.sum((lagrangian_fw + lagrangian_om_corr) * dt)
        return action
    
    # Simplified optimization
    initial_path = np.linspace(-1, 1, N_points)
    cons = ({'type': 'eq', 'fun': lambda x: x[0] + 1},
            {'type': 'eq', 'fun': lambda x: x[-1] - 1})
    
    try:
        result = minimize(lambda p: get_action(p, dt), initial_path, 
                         constraints=cons, method='SLSQP', 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        if result.success:
            return result.x
    except:
        pass
    
    # If optimization fails, return a simple S-shaped curve
    t_norm = np.linspace(0, 1, N_points)
    return np.tanh(4 * (t_norm - 0.5))

# 1. Langevin simulator
def langevin_trajectory(x0, T, dt):
    """Generate a Langevin trajectory"""
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps + 1)
    x[0] = x0
    
    # Pre-generate all random numbers
    noise = np.random.normal(0, 1, n_steps)
    
    for i in range(n_steps):
        A = drift(x[i])
        stochastic_term = np.sqrt(2 * D * dt) * noise[i]
        x[i+1] = x[i] + A * dt + stochastic_term
        
    return t, x

# 2. Generate large number of trajectories
num_trajectories = 2000  # Increase total trajectories to get sufficient transition samples
trajectories = []
transition_paths = []
print(f"Simulating {num_trajectories} trajectories...")
for i in range(num_trajectories):
    t, x = langevin_trajectory(x0=-1.0, T=T_total, dt=dt)
    trajectories.append(x)
    # Record successful transition paths (increase threshold for stricter definition of transition)
    if x[-1] > 0.3:  # Increase threshold to ensure truly reaching right potential well
        transition_paths.append(x)
print(f"Found {len(transition_paths)} transition paths.")
print(f"Transition rate: {len(transition_paths)/num_trajectories*100:.1f}%")

trajectories = np.array(trajectories)
transition_paths = np.array(transition_paths)

# 3. Plotting
plt.figure(figsize=(12, 8))

# Plot potential background
x_plot = np.linspace(-2, 2, 200)
U_plot = potential(x_plot)
plt.plot(x_plot, U_plot, 'k-', alpha=0.3, label='Potential $U(x)$')

# First plot transition trajectories (at bottom layer)
if len(transition_paths) > 0:
    # Only plot some transition trajectories to avoid excessive density
    num_to_plot = min(len(transition_paths), 25)  # Display at most 25 green trajectories
    for i in range(num_to_plot):
        plt.plot(transition_paths[i, :], potential(transition_paths[i, :]), color='green', alpha=0.5, linewidth=1)

# Then plot sample trajectories (at upper layer, increase transparency)
for i in range(min(150, num_trajectories)):  # Display more gray trajectories but with higher transparency
    plt.plot(trajectories[i, :], potential(trajectories[i, :]), color='gray', alpha=0.08, linewidth=0.8)

# Finally plot important paths (top layer)
if len(transition_paths) > 0:
    # Plot average transition path
    avg_transition_path = np.mean(transition_paths, axis=0)
    plt.plot(avg_transition_path, potential(avg_transition_path), 'g-', linewidth=3, label='Average Transition Path')

# Overlay theoretical instanton path
print("Computing OM instanton path...")
path_om = get_simple_om_instanton()
plt.plot(path_om, potential(path_om), 'b--', linewidth=3, label='OM Instanton (Theory)')


plt.title('Langevin Trajectories and the Most Probable Path')
plt.xlabel('Position $x$')
plt.ylabel('Potential Energy $U(x)$')
plt.ylim(-0.2, 2.5)
# Create a custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='k', alpha=0.3, lw=2, label='Potential U(x)'),
                   Line2D([0], [0], color='gray', alpha=0.4, lw=2, label='Sample Trajectories'),
                   Line2D([0], [0], color='green', alpha=0.6, lw=2, label='Transition Trajectories'),
                   Line2D([0], [0], color='b', linestyle='--', lw=3, label='OM Instanton (Theory)')]
plt.legend(handles=legend_elements)
plt.show()

# Plot evolution of probability distribution over time
fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
# Adjust time points, covering longer time range to observe complete equilibrium process
time_points = [int(0.01 * N_points), int(0.1 * N_points), int(0.3 * N_points), N_points - 1]

for i, t_idx in enumerate(time_points):
    ax = axes[i]
    if i < 3:
        time_label = f't = {time[t_idx]:.1f}'
    else:
        time_label = f't = {T_total:.1f} (Final Equilibrium)'
    
    # Plot histogram of particle positions
    ax.hist(trajectories[:, t_idx], bins=35, density=True, alpha=0.7, 
            color='skyblue', edgecolor='black', linewidth=0.5, label='Simulation')
    
    # Plot theoretical equilibrium distribution (Boltzmann distribution)
    x_theory = np.linspace(-2, 2, 200)
    boltzmann_dist = np.exp(-potential(x_theory) / D)
    boltzmann_dist /= np.trapz(boltzmann_dist, x_theory)
    ax.plot(x_theory, boltzmann_dist, 'r-', linewidth=2, label='Boltzmann Distribution')
    
    ax.set_title(time_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Position $x$')
    if i == 0:
        ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    
    # Add text showing percentage of total time
    if i < 3:
        percentage = time[t_idx] / T_total * 100
        ax.text(0.02, 0.98, f'{percentage:.1f}% of total time', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Evolution of Probability Distribution Towards Equilibrium (Extended Time)', 
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()