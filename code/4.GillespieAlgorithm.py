import numpy as np
import matplotlib.pyplot as plt

def gillespie_poisson(nu, t_max):
    """
    Simulate a Poisson process using the Gillespie algorithm.

    Parameters:
    nu (float): Event rate.
    t_max (float): Total simulation time.

    Returns:
    tuple: Two lists containing event times and corresponding event counts.
    """
    # Initialization
    t = 0.0  # Current time
    n = 0    # Current event count

    # Lists to store trajectory
    times = [t]
    counts = [n]
    
    # Simulation loop until maximum time is reached
    while t < t_max:
        # 1. Draw a uniform random number from (0, 1]
        #    Note: xi cannot be 0 because log(0) is negative infinity.
        #    np.random.random() generates random numbers in (0, 1), which meets the requirement.
        xi = np.random.random()
        
        # 2. Calculate waiting time according to the formula
        dt = -1.0/nu * np.log(xi)
        
        # 3. Update time and count
        t += dt
        n += 1
        
        # 4. Record trajectory point
        times.append(t)
        counts.append(n)
        
    return times, counts

# --- Simulation and plotting ---
# Set parameters
nu = 1.0      # Event rate
t_max = 100   # Total simulation time
num_traj = 10 # Number of simulated trajectories

# Create figure with two subplots
plt.figure(figsize=(15, 6))

# Left subplot: Single trajectory
plt.subplot(1, 2, 1)
times, counts = gillespie_poisson(nu, t_max)
plt.step(times, counts, where='post', linewidth=1.5, color='blue')
# Add theoretical expectation line (straight line y = νt)
t_theory = np.linspace(0, t_max, 100)
n_theory = nu * t_theory
plt.plot(t_theory, n_theory, 'k--', linewidth=2)
plt.xlabel('Time ($t$)')
plt.ylabel('Event Count ($x$)')
plt.title('Single Trajectory')
plt.grid(True, alpha=0.3)

# Right subplot: Multiple trajectories
plt.subplot(1, 2, 2)
for i in range(num_traj):
    times, counts = gillespie_poisson(nu, t_max)
    plt.step(times, counts, where='post', linewidth=1.5, alpha=0.7)

# Add theoretical expectation line (straight line y = νt)
t_theory = np.linspace(0, t_max, 100)
n_theory = nu * t_theory
plt.plot(t_theory, n_theory, 'k--', linewidth=2, label='Expected $\\nu t$')

# Set chart properties
plt.xlabel('Time ($t$)')
plt.ylabel('Event Count ($x$)')
plt.title('Multiple Trajectories')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()