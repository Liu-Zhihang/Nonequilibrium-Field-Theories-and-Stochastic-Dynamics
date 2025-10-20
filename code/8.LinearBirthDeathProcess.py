import numpy as np
import matplotlib.pyplot as plt

def linear_birth_death_gillespie(n0, lamb, mu, t_max):
    """
    Simulate linear birth-death process using Gillespie algorithm.
    
    Parameters:
    n0 (int): Initial population size
    lamb (float): Birth rate per individual (lambda)
    mu (float): Death rate per individual (mu)
    t_max (float): Maximum simulation time
    
    Returns:
    tuple: (list of time points, list of population sizes)
    """
    t = 0.0
    n = n0
    
    times = [t]
    populations = [n]
    
    while t < t_max:
        if n == 0:
            # Population extinct, process stops
            break
            
        # Calculate total rate
        birth_rate = lamb * n
        death_rate = mu * n
        total_rate = birth_rate + death_rate
        
        # Calculate time to next event
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        
        # Decide whether it's a birth or death
        if np.random.rand() < birth_rate / total_rate:
            n += 1  # Birth
        else:
            n -= 1  # Death
            
        times.append(t)
        populations.append(n)
        
    return np.array(times), np.array(populations)

# --- Simulation parameters ---
initial_population = 10
lambda_rate = 1.0  # Birth rate
mu_rate = 1.1      # Death rate (slightly greater than birth rate to accelerate extinction)
simulation_time = 50.0
num_simulations = 500

# --- Run multiple simulations ---
final_populations = []
plt.figure(figsize=(12, 8))

# Plot some example trajectories
plt.subplot(2, 1, 1)
for i in range(5):
    times, populations = linear_birth_death_gillespie(initial_population, lambda_rate, mu_rate, simulation_time)
    plt.step(times, populations, where='post', alpha=0.7)

plt.title(f'Example Trajectories of Linear Birth-Death Process ($n_0={initial_population}, \\lambda={lambda_rate}, \\mu={mu_rate}$)')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.grid(True)

# Collect final population sizes
for _ in range(num_simulations):
    _, populations = linear_birth_death_gillespie(initial_population, lambda_rate, mu_rate, simulation_time)
    final_populations.append(populations[-1])

# Plot histogram of final population sizes
plt.subplot(2, 1, 2)
plt.hist(final_populations, bins=np.arange(-0.5, max(final_populations) + 1.5, 1), density=True, rwidth=0.8)
plt.title(f'Distribution of Final Population Size after {num_simulations} Simulations (Steady State)')
plt.xlabel('Final Population Size')
plt.ylabel('Probability Density')
plt.xticks(np.arange(0, max(final_populations) + 1, 1))
plt.grid(True)

plt.tight_layout()
plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to properly display Chinese labels
plt.rcParams['axes.unicode_minus'] = False  # Used to properly display minus sign
plt.show()