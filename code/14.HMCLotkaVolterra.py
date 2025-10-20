"""
Using modern PyMC (v5+) and sunode library to reproduce the Lotka-Volterra model.
This version replaces the old PyMC3 + Theano + manual gradient implementation.
"""
import pymc as pm
import numpy as np
import arviz as az
import sunode
import sunode.wrappers.as_pytensor
import matplotlib.pyplot as plt
import pandas as pd


# 1. Data preparation (Hudson Bay Company dataset)
# Keeping consistent with the data used in the original tutorial
times = np.arange(1900, 1921, 1)
hare_data = np.array([
    30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
    27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7
])
lynx_data = np.array([
    4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
    8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6
])
data = np.stack([hare_data, lynx_data]).T
species_names = ['hares', 'lynx']

# 2. Define ODE equations
# sunode uses sympy for symbolic definition and automatic differentiation, which is very efficient
def lotka_volterra(t, y, p):
    """
    Right-hand side of the Lotka-Volterra (predator-prey) equations.
    
    Args:
        t: Time (sympy symbol)
        y: Dataclass of state variables (species populations), containing y.hares and y.lynx
        p: Dataclass of parameters, containing p.alpha, p.beta, p.gamma, p.delta
        
    Returns:
        A dictionary describing the rate of change of each state variable.
    """
    return {
        'hares': p.alpha * y.hares - p.beta * y.lynx * y.hares,
        'lynx': p.delta * y.hares * y.lynx - p.gamma * y.lynx,
    }

# 3. Build PyMC probabilistic model
# Define coordinates to use 'time' and 'species' dimensions in the model
coords = {"time": times, "species": species_names}

with pm.Model(coords=coords) as model:
    # --- Prior distributions ---
    # Use Lognormal priors to ensure parameters are positive, preventing ODE solutions from becoming negative
    alpha = pm.Lognormal("alpha", mu=np.log(1), sigma=0.5)
    beta = pm.Lognormal("beta", mu=np.log(0.05), sigma=0.5)
    gamma = pm.Lognormal("gamma", mu=np.log(1), sigma=0.5)
    delta = pm.Lognormal("delta", mu=np.log(0.05), sigma=0.5)

    # Priors for initial states
    initial_hares = pm.Lognormal("initial_hares", mu=np.log(30), sigma=1)
    initial_lynx = pm.Lognormal("initial_lynx", mu=np.log(4), sigma=1)

    # Prior for observation noise
    sigma = pm.HalfNormal("sigma", sigma=1)

    # --- Solve ODE using sunode ---
    # This is the key step that replaces the manual Theano Op
    y_hat, _, _, _, _, _ = sunode.wrappers.as_pytensor.solve_ivp(
        y0={
            # Define initial conditions for the ODE
            # Format: {'variable_name': (PyTensor variable, shape)}
            'hares': (initial_hares, ()),
            'lynx': (initial_lynx, ()),
        },
        params={
            # Define parameters for the ODE
            # sunode will automatically compute gradients with respect to these PyTensor variables
            'alpha': (alpha, ()),
            'beta': (beta, ()),
            'gamma': (gamma, ()),
            'delta': (delta, ()),
            # sunode requires all parameters to be PyTensor variables or numpy arrays
            '_dummy': np.array(0.), 
        },
        # Pass in the ODE function we defined earlier
        rhs=lotka_volterra,
        # Define time points for solving
        tvals=times,
        t0=times[0],
    )

    # --- Likelihood function ---
    # sunode returns a dictionary, we need to stack the solution into a matrix to match the shape of observed data
    ode_solution = pm.math.stack([y_hat['hares'], y_hat['lynx']], axis=1)
    
    # Assume observations follow a log-normal distribution
    Y_obs = pm.Lognormal(
        "Y_obs", 
        mu=pm.math.log(ode_solution), 
        sigma=sigma, 
        observed=data, 
        dims=("time", "species")
    )
    
# 4. Run sampler
# The NUTS sampler will utilize the exact gradient information provided by sunode
with model:
    # sunode's C backend does not support multiprocessing "pickling", so we must use a single core
    idata = pm.sample(2000, tune=2000, chains=4, cores=1, target_accept=0.9)

# 5. Results visualization
# Plot posterior distributions of parameters
az.plot_posterior(idata, var_names=["alpha", "beta", "gamma", "delta", "initial_hares", "initial_lynx", "sigma"])
plt.suptitle("Posterior Distributions of Parameters", y=1.02)
plt.tight_layout()
plt.savefig("lv_posterior_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# Extract posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(idata)

# Plot posterior predictions vs. real data
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for i, (spec, color) in enumerate(zip(species_names, ['C0', 'C1'])):
    # Plot real data
    axes[i].plot(times, data[:, i], 'o', color=color, label=f"Observed {spec.capitalize()}")
    
    # Extract posterior predictive mean and 94% credible interval
    pred_mean = posterior_predictive.posterior_predictive['Y_obs'].mean(dim=('chain', 'draw')).sel(species=spec)
    hdi_94 = az.hdi(posterior_predictive.posterior_predictive['Y_obs'].sel(species=spec), hdi_prob=0.94)
    
    # Plot posterior predictions
    axes[i].plot(times, pred_mean, color=color, lw=2, label=f"Posterior Mean {spec.capitalize()}")
    # Use isel (integer selection) instead of sel (label selection) to get HDI upper and lower bounds
    axes[i].fill_between(times, hdi_94.Y_obs.isel(hdi=0), hdi_94.Y_obs.isel(hdi=1), color=color, alpha=0.3, label="94% HDI")
    
    axes[i].set_ylabel("Population")
    axes[i].set_title(f"{spec.capitalize()} Population Dynamics")
    axes[i].legend()

axes[1].set_xlabel("Year")
plt.tight_layout()
plt.savefig("lv_posterior_predictions.png", dpi=300, bbox_inches='tight')
plt.show()

# 6. Phase plot
plt.figure(figsize=(8, 8))
pred_mean_hares = posterior_predictive.posterior_predictive['Y_obs'].mean(dim=('chain', 'draw')).sel(species='hares')
pred_mean_lynx = posterior_predictive.posterior_predictive['Y_obs'].mean(dim=('chain', 'draw')).sel(species='lynx')

plt.plot(pred_mean_hares, pred_mean_lynx, lw=2, label="Posterior Mean Trajectory")
plt.plot(hare_data, lynx_data, 'o', alpha=0.8, label="Observed Data")
plt.xlabel("Hares Population")
plt.ylabel("Lynx Population")
plt.title("Lotka-Volterra Phase Portrait")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("lv_phase_portrait.png", dpi=300, bbox_inches='tight')
plt.show()


# 7. Trace plot
# This plot is used to diagnose the health of the MCMC sampling process
az.plot_trace(idata, var_names=["alpha", "beta", "gamma", "delta", "initial_hares", "initial_lynx", "sigma"])
plt.tight_layout()
plt.savefig("lv_trace_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# 8. Results visualization
# Plot posterior distributions of parameters
az.plot_posterior(
    idata, 
    var_names=["alpha", "beta", "gamma", "delta", "initial_hares", "initial_lynx", "sigma"],
    kind="hist"  # Set chart type to histogram
)
plt.suptitle("Posterior Distributions of Parameters (Histogram)", y=1.02)
plt.tight_layout()
plt.savefig("lv_posterior_histograms.png", dpi=300, bbox_inches='tight')
plt.show()