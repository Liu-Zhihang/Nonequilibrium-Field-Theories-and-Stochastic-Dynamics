import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Parameter Setup ---
lambda_rate = 1.0  # Predator death rate
mu_rate = 1.0      # Prey birth rate
gamma_p = 0.01     # Predation rate

# --- Deterministic ODE Definition ---
def lotka_volterra_ode(y, t, lambda_r, mu_r, gamma_p_r, gamma_c_r):
    """
    Defines the system of ordinary differential equations for the Lotka-Volterra model.
    """
    a, b = y
    dadt = (-lambda_r + gamma_p_r * b) * a
    dbdt = (mu_r - gamma_p_r * a - gamma_c_r * b) * b
    return [dadt, dbdt]

# --- Gillespie Algorithm for Lotka-Volterra Process ---
def lotka_volterra_gillespie(n_a, n_b, rates, t_end):
    """
    Simulates the Lotka-Volterra process using the Gillespie algorithm.
    
    Args:
        n_a (int): Initial number of predators.
        n_b (int): Initial number of prey.
        rates (tuple): A tuple of rate constants (lambda_r, mu_r, gamma_p_r, gamma_c_r).
        t_end (float): The end time for the simulation.
        
    Returns:
        A tuple containing arrays for times, predator populations, and prey populations.
    """
    lambda_r, mu_r, gamma_p_r, gamma_c_r = rates
    
    t = 0.0
    # Ensure populations are floats to avoid potential integer division issues
    n_a = float(n_a)
    n_b = float(n_b)
    
    times = [t]
    populations_a = [n_a]
    populations_b = [n_b]
    
    # Stoichiometry matrix: rows are reactions, columns are species (A, B)
    nu = np.array([
        [-1,  0],  # Reaction 1: A -> 0 (Predator death)
        [ 0, +1],  # Reaction 2: B -> 2B (Prey birth)
        [+1, -1],  # Reaction 3: A+B -> 2A (Predation)
        [ 0, -1]   # Reaction 4: 2B -> B (Prey competition)
    ])
    
    while t < t_end:
        if n_a == 0 or n_b == 0:
            break  # Stop if one species goes extinct
            
        # Calculate propensities for each reaction
        propensities = np.array([
            lambda_r * n_a,
            mu_r * n_b,
            gamma_p_r * n_a * n_b,
            gamma_c_r * n_b * (n_b - 1)
        ])
        
        a_total = np.sum(propensities)
        
        if a_total == 0:
            break # No more reactions can occur
            
        # Generate waiting time (tau) until the next reaction
        xi1 = np.random.uniform(0, 1)
        tau = -np.log(xi1) / a_total
        
        # Choose which reaction occurs
        xi2 = np.random.uniform(0, 1)
        p_normalized = propensities / a_total
        reaction_idx = np.random.choice(len(propensities), p=p_normalized)
        
        # Update time and populations
        t += tau
        n_a += nu[reaction_idx, 0]
        n_b += nu[reaction_idx, 1]
        
        times.append(t)
        populations_a.append(n_a)
        populations_b.append(n_b)
        
    return np.array(times), np.array(populations_a), np.array(populations_b)

# --- Initial Conditions ---
# The deterministic fixed point is (100, 100) for gamma_c=0.
# We start away from the fixed point to observe oscillations.
n_a0 = 50
n_b0 = 100

# --- Plot 1: Stochastic Oscillations Time Series (Reproducing PPT_1.jpg) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(10, 6))

gamma_c_1 = 0.0
rates_1 = (lambda_rate, mu_rate, gamma_p, gamma_c_1)
# For a better single trajectory view, let's start this one away from the fixed point too
t_sim_1, n_a_sim_1, n_b_sim_1 = lotka_volterra_gillespie(50, 100, rates_1, 100)

ax1.plot(t_sim_1, n_a_sim_1, label='Predator', lw=1.5, color='royalblue')
ax1.plot(t_sim_1, n_b_sim_1, label='Prey', lw=1.5, color='darkorange')
ax1.set_xlabel('Time', fontsize=14)
ax1.set_ylabel('Population', fontsize=14)
ax1.set_title(f'Stochastic Simulation of Predator-Prey Dynamics\n'
              f'$\lambda={lambda_rate}, \mu={mu_rate}, \gamma_p={gamma_p}, \gamma_c={gamma_c_1}$', 
              fontsize=16)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, max(np.max(n_a_sim_1), np.max(n_b_sim_1)) * 1.1 if len(n_a_sim_1) > 0 else 100)
plt.tight_layout()
plt.show()

# --- Plot 2: Stochastic vs. Deterministic (Reproducing PPT2.jpg) ---
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left Subplot: gamma_c = 0 ---
ax2_left = axes[0]
gamma_c_left = 0.0
rates_left = (lambda_rate, mu_rate, gamma_p, gamma_c_left)
num_simulations = 50  # Number of stochastic trajectories to plot

# Run and plot multiple stochastic simulations
for i in range(num_simulations):
    t_sim, n_a_sim, n_b_sim = lotka_volterra_gillespie(n_a0, n_b0, rates_left, 70)
    # Use a low alpha to create a "cloud" effect
    ax2_left.plot(t_sim, n_a_sim, color='gray', lw=0.5, alpha=0.4, 
                  label='Stochastic Simulation' if i == 0 else "")
    ax2_left.plot(t_sim, n_b_sim, color='gray', lw=0.5, alpha=0.4)


# Deterministic solution
t_ode = np.linspace(0, 70, 1000)
y0 = [n_a0, n_b0]
sol_left = odeint(lotka_volterra_ode, y0, t_ode, args=rates_left)
# Plot deterministic solution with a thick black line for contrast
ax2_left.plot(t_ode, sol_left[:, 0], color='black', lw=2.0, label='Deterministic Solution')
ax2_left.plot(t_ode, sol_left[:, 1], color='black', lw=2.0)


ax2_left.set_xlabel('Time', fontsize=12)
ax2_left.set_ylabel('Population', fontsize=12)
ax2_left.set_title(f'$\gamma_p = {gamma_p}, \gamma_c = {gamma_c_left}$', fontsize=14)
ax2_left.legend()
ax2_left.set_xlim(0, 70)
ax2_left.set_ylim(0, 850)


# --- Right Subplot: gamma_c > 0 ---
ax2_right = axes[1]
gamma_c_right = 0.001
rates_right = (lambda_rate, mu_rate, gamma_p, gamma_c_right)

# Run and plot multiple stochastic simulations
for i in range(num_simulations):
    t_sim, n_a_sim, n_b_sim = lotka_volterra_gillespie(n_a0, n_b0, rates_right, 70)
    ax2_right.plot(t_sim, n_a_sim, color='gray', lw=0.5, alpha=0.4,
                   label='Stochastic Simulation' if i == 0 else "")
    ax2_right.plot(t_sim, n_b_sim, color='gray', lw=0.5, alpha=0.4)


# Deterministic solution
sol_right = odeint(lotka_volterra_ode, y0, t_ode, args=rates_right)
ax2_right.plot(t_ode, sol_right[:, 0], color='black', lw=2.0, label='Deterministic Solution')
ax2_right.plot(t_ode, sol_right[:, 1], color='black', lw=2.0)

ax2_right.set_xlabel('Time', fontsize=12)
ax2_right.set_ylabel('Population', fontsize=12)
ax2_right.set_title(f'$\gamma_p = {gamma_p}, \gamma_c = {gamma_c_right}$', fontsize=14)
ax2_right.legend()
ax2_right.set_xlim(0, 70)
ax2_right.set_ylim(0, 650)


fig2.suptitle('Comparison of Stochastic Trajectories and Deterministic Solutions', fontsize=20, y=1.02)
plt.tight_layout()
plt.show()
