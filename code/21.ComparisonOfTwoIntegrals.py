import numpy as np
import matplotlib.pyplot as plt

# --- Parameter definition ---
mu = 0.1      # True drift rate
sigma = 0.3   # Volatility
x0 = 1.0      # Initial value
dt = 0.001    # Time step
T = 2.0       # Total time
n = int(T / dt) # Total steps
num_paths = 5000 # Simulate large number of trajectories to calculate expected values

# --- Simulation process ---
x_ito = np.zeros((num_paths, n + 1))
x_strat = np.zeros((num_paths, n + 1))
x_ito[:, 0] = x0
x_strat[:, 0] = x0

# Generate all random numbers
Z = np.random.randn(num_paths, n)
dW = np.sqrt(dt) * Z

# Drift coefficient
A_ito = mu
A_strat = mu - 0.5 * sigma**2

for i in range(n):
    # 1. Ito Simulation (Euler-Maruyama)
    x_ito[:, i+1] = x_ito[:, i] + A_ito * x_ito[:, i] * dt + sigma * x_ito[:, i] * dW[:, i]

    # 2. Stratonovich Simulation (Heun's method)
    # Predictor step
    x_pred = x_strat[:, i] + A_strat * x_strat[:, i] * dt + sigma * x_strat[:, i] * dW[:, i]
    # Corrector step using midpoint approximation
    C_mid = sigma * (x_strat[:, i] + x_pred) / 2.0
    x_strat[:, i+1] = x_strat[:, i] + A_strat * x_strat[:, i] * dt + C_mid * dW[:, i]


# --- Calculate expected values and plot ---
t = np.linspace(0, T, n + 1)
expected_x_ito = np.mean(x_ito, axis=0)
expected_x_strat = np.mean(x_strat, axis=0)
analytical_expected_x = x0 * np.exp(mu * t)

plt.figure(figsize=(12, 7))
plt.plot(t, analytical_expected_x, 'r-', lw=3, label='Analytical Expectation $E[x_t]=x_0 e^{\mu t}$')
plt.plot(t, expected_x_ito, 'b--', lw=2, label='Mean of Ito Simulations')
plt.plot(t, expected_x_strat, 'g:', lw=2, label='Mean of Stratonovich Simulations')

plt.title("Ito vs. Stratonovich Simulation for Multiplicative Noise", fontsize=16)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("E[x(t)]", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()