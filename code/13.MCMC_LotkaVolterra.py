import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm

# 1. Load data
# Data source: https://github.com/stan-dev/example-models
# Year, Lynx (x1000), Hare (x1000)
data = np.array([
    [1900, 4.0, 30.0], [1901, 6.1, 47.2], [1902, 9.8, 70.2],
    [1903, 35.2, 77.4], [1904, 59.4, 36.3], [1905, 41.7, 20.6],
    [1906, 19.0, 18.1], [1907, 13.0, 21.4], [1908, 8.3, 22.0],
    [1909, 9.1, 25.4], [1910, 7.4, 27.1], [1911, 8.0, 40.3],
    [1912, 12.3, 57.0], [1913, 19.5, 76.6], [1914, 45.7, 52.3],
    [1915, 51.1, 19.5], [1916, 29.7, 11.2], [1917, 15.8, 7.6],
    [1918, 9.7, 14.6], [1919, 10.1, 16.2], [1920, 8.6, 24.7]
])
years = data[:, 0]
lynx_data = data[:, 1]
hare_data = data[:, 2]
y_obs = np.vstack((hare_data, lynx_data)).T # Observed data [H, L]

# 2. Define Lotka-Volterra model
def lotka_volterra(y, t, alpha, beta, gamma, delta):
    """
    Lotka-Volterra differential equations
    y: [H, L] population array
    t: time
    alpha, beta, gamma, delta: model parameters
    """
    H, L = y
    dH_dt = alpha * H - beta * H * L
    dL_dt = delta * H * L - gamma * L
    return [dH_dt, dL_dt]

# 3. Define log posterior probability function
def log_posterior(theta, y_obs, t_obs):
    alpha, beta, gamma, delta = theta
    
    # a. Log-Prior
    # Assume parameters follow wide normal distributions, and must be positive
    if any(p <= 0 for p in theta):
        return -np.inf
    log_prior_alpha = norm.logpdf(alpha, loc=1, scale=1)
    log_prior_beta = norm.logpdf(beta, loc=0.05, scale=0.05)
    log_prior_gamma = norm.logpdf(gamma, loc=1, scale=1)
    log_prior_delta = norm.logpdf(delta, loc=0.02, scale=0.02)
    log_p = log_prior_alpha + log_prior_beta + log_prior_gamma + log_prior_delta

    # b. Log-Likelihood
    # Initial conditions use the first point of data
    y0 = y_obs[0, :]
    # Numerically solve differential equations using odeint
    y_pred = odeint(lotka_volterra, y0, t_obs, args=(alpha, beta, gamma, delta))
    
    # Assume errors follow a log-normal distribution, equivalent to log-transformed data following a normal distribution
    # We also need to estimate a standard deviation sigma for the error
    # For simplicity, we fix a reasonable sigma value here
    sigma = 0.5 
    log_likelihood = np.sum(norm.logpdf(np.log(y_obs), loc=np.log(y_pred), scale=sigma))
    
    return log_p + log_likelihood

# 4. Implement Metropolis-Hastings MCMC
def metropolis_hastings(log_posterior_func, n_iter, initial_theta, proposal_std, y_obs, t_obs):
    # Initialization
    n_params = len(initial_theta)
    chain = np.zeros((n_iter, n_params))
    chain[0, :] = initial_theta
    
    current_log_post = log_posterior_func(initial_theta, y_obs, t_obs)
    
    accepted_count = 0
    
    for i in range(1, n_iter):
        if i % 1000 == 0:
            print(f"Iteration {i}/{n_iter}...")
            
        # a. Propose new point
        proposal_theta = np.random.normal(loc=chain[i-1, :], scale=proposal_std)
        
        # b. Calculate acceptance probability
        proposal_log_post = log_posterior_func(proposal_theta, y_obs, t_obs)
        
        log_alpha = proposal_log_post - current_log_post
        
        # c. Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            chain[i, :] = proposal_theta
            current_log_post = proposal_log_post
            accepted_count += 1
        else:
            chain[i, :] = chain[i-1, :]
            
    print(f"Acceptance rate: {accepted_count / n_iter:.2f}")
    return chain

# 5. Run MCMC
n_iterations = 50000
burn_in = 10000 # Discard early unstable samples
initial_params = [1.0, 0.05, 1.0, 0.02] # Initial guess
proposal_widths = [0.05, 0.001, 0.05, 0.001] # Proposal distribution standard deviation, needs tuning
t_span = np.arange(len(years)) # Time points (0, 1, 2, ...)

chain = metropolis_hastings(log_posterior, n_iterations, initial_params, proposal_widths, y_obs, t_span)

# Discard burn-in phase
posterior_samples = chain[burn_in:, :]

# 6. Results visualization
param_names = ['alpha', 'beta', 'gamma', 'delta']

# a. Plot trace plots of parameters
plt.figure(figsize=(15, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(posterior_samples[:, i])
    plt.title(f'Trace of {param_names[i]}')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
plt.tight_layout()
plt.show()

# b. Plot posterior distributions
plt.figure(figsize=(15, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.hist(posterior_samples[:, i], bins=50, density=True, alpha=0.6)
    plt.title(f'Posterior of {param_names[i]}')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
plt.tight_layout()
plt.show()

# c. Plot model predictions vs. real data
plt.figure(figsize=(12, 6))
# Randomly sample some parameter combinations from the posterior distribution for simulation
n_samples_plot = 100
sample_indices = np.random.randint(0, len(posterior_samples), n_samples_plot)

for idx in sample_indices:
    params = posterior_samples[idx, :]
    y_pred = odeint(lotka_volterra, y_obs[0,:], t_span, args=tuple(params))
    plt.plot(years, y_pred[:, 0], color='skyblue', alpha=0.1) # Predicted hares
    plt.plot(years, y_pred[:, 1], color='lightcoral', alpha=0.1) # Predicted lynx

# Plot original data points
plt.plot(years, hare_data, 'o-', color='blue', label='Observed Hares')
plt.plot(years, lynx_data, 'o-', color='red', label='Observed Lynx')
plt.xlabel('Year')
plt.ylabel('Population (x1000)')
plt.title('Lotka-Volterra Model Fit to Hudson Bay Data')
plt.legend()
plt.grid(True)
plt.show()