import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Use dark background for plots ---
plt.style.use('dark_background')

# --- Parameter setup ---
beta1 = 1.0
n0 = 1.0
t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 200)

def rate_equation(t, n, delta, beta1):
    """The mean-field rate equation for DP."""
    return delta * n - beta1 * n**2

# --- Solve for the three cases ---
delta_active = 0.5
sol_active = solve_ivp(rate_equation, t_span, [n0], args=(delta_active, beta1), t_eval=t_eval)
n_act_theory = delta_active / beta1

delta_absorbing = -0.5
sol_absorbing = solve_ivp(rate_equation, t_span, [n0], args=(delta_absorbing, beta1), t_eval=t_eval)

delta_critical = 0.0
sol_critical = solve_ivp(rate_equation, t_span, [n0], args=(delta_critical, beta1), t_eval=t_eval)

# --- Create Animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

def update_mean_field(frame):
    # --- Clear canvases for new frame ---
    ax1.clear()
    ax2.clear()
    
    # --- Panel 1: Density vs. Time ---
    ax1.plot(sol_active.t, sol_active.y[0], label=f'Active Phase ($\\Delta={delta_active}$)', color='lime')
    ax1.axhline(y=n_act_theory, linestyle='--', color='lime', alpha=0.7)
    ax1.plot(sol_absorbing.t, sol_absorbing.y[0], label=f'Absorbing Phase ($\\Delta={delta_absorbing}$)', color='tomato')
    ax1.plot(sol_critical.t, sol_critical.y[0], label=f'Critical Point ($\\Delta=0$)', color='cyan')
    
    # Dynamic timeline
    current_time = t_eval[frame]
    ax1.axvline(current_time, color='yellow', linestyle='--', lw=1)
    
    ax1.set_xlabel('Time $t$', fontsize=12)
    ax1.set_ylabel('Density $n(t)$', fontsize=12)
    ax1.set_title('Density Evolution over Time', fontsize=14)
    ax1.legend()
    ax1.set_xlim(t_span)
    ax1.set_ylim(-0.1, 1.1)

    # --- Panel 2: Phase Space Flow ---
    n_range = np.linspace(-0.2, 1.2, 200)
    ax2.axhline(0, color='gray', lw=0.5)
    
    # Plot phase flow curves
    ax2.plot(n_range, rate_equation(0, n_range, delta_active, beta1), color='lime')
    ax2.plot(n_range, rate_equation(0, n_range, delta_absorbing, beta1), color='tomato')
    ax2.plot(n_range, rate_equation(0, n_range, delta_critical, beta1), color='cyan')

    # Plot fixed points
    ax2.plot(0, 0, 'wo', markersize=8, label='Fixed Point')
    ax2.plot(n_act_theory, 0, 'wo', markersize=8)

    # Plot dynamic evolution points
    ax2.plot(sol_active.y[0, frame], rate_equation(0, sol_active.y[0, frame], delta_active, beta1), 'o', color='lime', markersize=10)
    ax2.plot(sol_absorbing.y[0, frame], rate_equation(0, sol_absorbing.y[0, frame], delta_absorbing, beta1), 'o', color='tomato', markersize=10)
    ax2.plot(sol_critical.y[0, frame], rate_equation(0, sol_critical.y[0, frame], delta_critical, beta1), 'o', color='cyan', markersize=10)

    ax2.set_xlabel('Density $n$', fontsize=12)
    ax2.set_ylabel('Rate of Change $\\partial_t n$', fontsize=12)
    ax2.set_title('Dynamical Flow in Phase Space', fontsize=14)
    ax2.legend()
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.3, 0.3)
    
    fig.suptitle(f'Mean-Field Dynamics of Directed Percolation (Time t={current_time:.2f})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- Generate and save the animation ---
ani_mf = FuncAnimation(fig, update_mean_field, frames=len(t_eval), interval=50)
ani_mf.save("dp_mean_field_dynamics.gif", writer=PillowWriter(fps=20))
plt.show()