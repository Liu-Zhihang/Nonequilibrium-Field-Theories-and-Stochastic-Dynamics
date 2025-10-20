import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# --- Target distribution (correlated 2D Gaussian distribution) --- 
# This defines our "potential energy field"
mean = np.array([0.0, 0.0])
cov = np.array([[1.0, 0.95], 
                [0.95, 1.0]])
inv_cov = np.linalg.inv(cov)
target_dist = multivariate_normal(mean, cov)

def potential_energy(theta):
    """ Potential energy U(theta) = -log p(theta) """
    return -target_dist.logpdf(theta)

def grad_potential_energy(theta):
    """ Gradient of potential energy, i.e., -d/d(theta) log p(theta) """
    # For Gaussian distribution N(mu, Sigma), the gradient of log p is -inv(Sigma) * (theta - mu)
    # So the gradient of potential energy (-log p) is inv(Sigma) * (theta - mu)
    return inv_cov @ (theta - mean)

# --- Leapfrog integrator ---
def leapfrog(theta, v, grad_U, epsilon, L):
    """
    Perform L steps of leapfrog integration.
    theta: current position
    v: current momentum
    grad_U: function to compute gradient of potential energy
    epsilon: step size
    L: number of steps
    """
    # Initial half-step momentum update
    v_half = v - 0.5 * epsilon * grad_U(theta)
    # L-1 full-step updates
    for _ in range(L - 1):
        theta = theta + epsilon * v_half
        v_half = v_half - epsilon * grad_U(theta)
    # Final full-step position update
    theta = theta + epsilon * v_half
    # Final half-step momentum update
    v = v_half - 0.5 * epsilon * grad_U(theta)
    return theta, v

# --- HMC sampler ---
def hmc_sampler(start_theta, n_samples, epsilon, L):
    samples = [start_theta]
    theta = start_theta
    
    for _ in range(n_samples - 1):
        # 1. Sample momentum
        v_current = np.random.randn(2)
        
        # 2. Simulate trajectory using leapfrog
        theta_prop, v_prop = leapfrog(np.copy(theta), np.copy(v_current), grad_potential_energy, epsilon, L)
        
        # 3. Metropolis-Hastings correction
        # H(theta, v) = U(theta) + K(v), where K(v) = 0.5 * v.T @ v
        U_current = potential_energy(theta)
        K_current = 0.5 * (v_current @ v_current)
        H_current = U_current + K_current
        
        U_prop = potential_energy(theta_prop)
        K_prop = 0.5 * (v_prop @ v_prop)
        H_prop = U_prop + K_prop
        
        # Acceptance probability
        alpha = min(1.0, np.exp(H_current - H_prop))
        
        if np.random.rand() < alpha:
            theta = theta_prop
            
        samples.append(theta)
        
    return np.array(samples)

# --- Random Walk Metropolis sampler ---
def rwm_sampler(start_theta, n_samples, step_size):
    samples = [start_theta]
    theta = start_theta
    
    for _ in range(n_samples - 1):
        # Propose a new state
        theta_prop = theta + step_size * np.random.randn(2)
        
        # Acceptance probability
        alpha = min(1.0, np.exp(potential_energy(theta) - potential_energy(theta_prop)))
        
        if np.random.rand() < alpha:
            theta = theta_prop
            
        samples.append(theta)
        
    return np.array(samples)

# --- Run simulation and plot ---
n_samples = 500
start_theta = np.array([-2.5, 2.5])

# Adjust parameters
hmc_samples = hmc_sampler(start_theta, n_samples, epsilon=0.1, L=20)
rwm_samples = rwm_sampler(start_theta, n_samples, step_size=0.15)

# Plotting
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = target_dist.pdf(pos)

plt.figure(figsize=(12, 6))

# RWM plot
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=10, cmap='viridis')
plt.plot(rwm_samples[:, 0], rwm_samples[:, 1], 'r-o', markersize=3, alpha=0.5)
plt.title(f"Random Walk Metropolis (N={n_samples})")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")

# HMC plot
plt.subplot(1, 2, 2)
plt.contour(X, Y, Z, levels=10, cmap='viridis')
plt.plot(hmc_samples[:, 0], hmc_samples[:, 1], 'b-o', markersize=3, alpha=0.5)
plt.title(f"Hamiltonian Monte Carlo (N={n_samples})")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")

plt.tight_layout()
plt.show()