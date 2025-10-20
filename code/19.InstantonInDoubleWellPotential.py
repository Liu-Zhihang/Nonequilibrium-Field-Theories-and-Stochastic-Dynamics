import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define physical parameters
gamma = 1.0  # Friction coefficient
D = 0.05     # Reduce diffusion coefficient to enhance differences

# 1. Define double-well potential and its derivatives
def potential(x):
    """Double-well potential U(x) = (x^2 - 1)^2"""
    return (x**2 - 1)**2

def force(x):
    """Force F(x) = -U'(x)"""
    return -4 * x * (x**2 - 1)

def drift(x):
    """Drift term A(x) = F(x) / gamma"""
    return force(x) / gamma

def drift_prime(x):
    """Derivative of drift term A'(x)"""
    return (-12 * x**2 + 4) / gamma

# 2. Discretized action
def get_action(path, dt, action_type='OM'):
    """Calculate discrete action of path"""
    x = path
    x_dot = np.diff(x) / dt
    x_mid = (x[:-1] + x[1:]) / 2  # Midpoints for evaluating A(x) and A'(x)
    
    A = drift(x_mid)
    A_prime = drift_prime(x_mid)
    
    # FW Lagrangian
    lagrangian_fw = (x_dot - A)**2 / (4 * D)
    
    if action_type == 'FW':
        action = np.sum(lagrangian_fw * dt)
    elif action_type == 'OM':
        # OM correction term
        lagrangian_om_corr = 0.5 * A_prime
        action = np.sum((lagrangian_fw + lagrangian_om_corr) * dt)
    else:
        raise ValueError("action_type must be 'FW' or 'OM'")
        
    return action

# 3. Numerical optimization to find instanton
# Time and path settings
T_total = 8.0  # Increase total time
N_points = 201  # Increase number of path points
dt = T_total / (N_points - 1)
time = np.linspace(0, T_total, N_points)

# Initial path guess (straight line from -1 to +1)
initial_path = np.linspace(-1, 1, N_points)

# Optimization functions
def objective_fw(p):
    return get_action(p, dt, 'FW')

def objective_om(p):
    return get_action(p, dt, 'OM')

# Run optimization
# Fix start and end points of path
cons = ({'type': 'eq', 'fun': lambda x: x[0] + 1},
        {'type': 'eq', 'fun': lambda x: x[-1] - 1})

print("Minimizing FW action...")
result_fw = minimize(objective_fw, initial_path, constraints=cons, method='SLSQP', 
                     options={'maxiter': 1000, 'ftol': 1e-9})
path_fw = result_fw.x
print("FW minimization successful:", result_fw.success)
print("FW action value:", result_fw.fun)

print("Minimizing OM action...")
# Use FW result as initial guess for OM
result_om = minimize(objective_om, path_fw, constraints=cons, method='SLSQP',
                     options={'maxiter': 1000, 'ftol': 1e-9})
path_om = result_om.x
print("OM minimization successful:", result_om.success)
print("OM action value:", result_om.fun)


# 4. Plotting
plt.figure(figsize=(10, 6))
x_plot = np.linspace(-1.5, 1.5, 200)
U_plot = potential(x_plot)

plt.plot(x_plot, U_plot, 'k-', label='Potential $U(x)=(x^2-1)^2$')
plt.plot(path_fw, potential(path_fw), 'r.--', label='FW Instanton')
plt.plot(path_om, potential(path_om), 'b.--', label='OM Instanton')

plt.title('Most Probable Paths (Instantons) in a Double-Well Potential')
plt.xlabel('Position $x$')
plt.ylabel('Potential Energy $U(x)$')
plt.ylim(-0.2, 2)
plt.legend()
plt.grid(True)
plt.show()

# Plot path vs. time
plt.figure(figsize=(10, 6))
plt.plot(time, path_fw, 'r-', label='FW Instanton')
plt.plot(time, path_om, 'b-', label='OM Instanton')
plt.axhline(-1, color='k', linestyle='--', alpha=0.5)
plt.axhline(1, color='k', linestyle='--', alpha=0.5)
plt.title('Path vs. Time')
plt.xlabel('Time $t$')
plt.ylabel('Position $x(t)$')
plt.legend()
plt.grid(True)
plt.show()