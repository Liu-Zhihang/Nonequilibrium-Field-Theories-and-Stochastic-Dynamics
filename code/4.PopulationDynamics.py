import numpy as np
import matplotlib.pyplot as plt

def gillespie_death_process(lam, n0, t_max):
    """
    Simulate a simple death process using the Gillespie algorithm.

    Parameters:
    lam (float): Per capita death rate.
    n0 (int): Initial population size.
    t_max (float): Total simulation time.

    Returns:
    tuple: Two lists containing event times and corresponding population sizes.
    """
    # Initialization
    t = 0.0
    n = n0
    
    times = [t]
    counts = [n]
    
    while t < t_max and n > 0:
        # Key difference: Rate depends on state
        rate = lam * n
        
        # Draw waiting time
        xi = np.random.random()
        dt = -1.0 / rate * np.log(xi)
        
        # Update time
        t += dt
        
        # Check if exceeding maximum time
        if t >= t_max:
            # If exceeding maximum time, record final state and exit
            times.append(t_max)
            counts.append(n)
            break
            
        # Record state before event occurs
        times.append(t)
        counts.append(n)
        
        # Update population size (one individual dies)
        n -= 1
        
        # Record state after event occurs
        times.append(t)
        counts.append(n)
            
    # If extinction occurs before t_max, extend trajectory to t_max
    if t < t_max and n == 0:
        times.append(t_max)
        counts.append(0)

    return times, counts

# --- Simulation and plotting ---
# Set parameters
lam = 0.1     # Per capita death rate
n0 = 50       # Initial population size
t_max = 50    # Total simulation time
num_traj = 5  # Simulate 5 trajectories

# Create figure
plt.figure(figsize=(12, 7))

# Plot multiple trajectories to show randomness
for i in range(num_traj):
    times, counts = gillespie_death_process(lam, n0, t_max)
    plt.step(times, counts, where='post', label=f'Trajectory {i+1}')

# Plot deterministic model solution for comparison
t_deterministic = np.linspace(0, t_max, 200)
n_deterministic = n0 * np.exp(-lam * t_deterministic)
plt.plot(t_deterministic, n_deterministic, 'k--', linewidth=2, label='Deterministic Model ($N_0 e^{-\lambda t}$)')

plt.xlabel('Time (t)')
plt.ylabel('Population Size (n)')
plt.title('Simple Death Process: Stochastic vs. Deterministic')
plt.grid(True)
plt.legend()
plt.show()