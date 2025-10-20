import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LightSource

# --- Use dark theme ---
plt.style.use('dark_background')

def gillespie_schlogl_single(n0, k, t_max):
    """Single Gillespie simulation of Schlogl model"""
    t, n = 0.0, float(n0) # Ensure n is a float
    times, populations = [t], [n]
    k1, k2, k3, k4 = k
    
    while t < t_max:
        if n < 0: n = 0 # *** Added robustness: prevent negative particle count ***
            
        # Propensity calculations (for simplicity, constant concentrations are absorbed into rate constants)
        a1 = k1 * n * (n - 1) / 2
        a2 = k2 * n * (n - 1) * (n - 2) / 6
        a3 = k3
        a4 = k4 * n
        
        a_total = a1 + a2 + a3 + a4
        if a_total <= 1e-9:
            # If total rate is extremely small, jump forward a small step to avoid infinite loop
            t += 0.1 
            if t > t_max: break
            times.append(t)
            populations.append(n)
            continue

        dt = -np.log(np.random.rand()) / a_total
        t += dt
        if t > t_max: break
            
        # Determine which reaction occurs
        rand_val = np.random.rand() * a_total
        if rand_val < a1:
            n += 1  # A + 2X -> 3X
        elif rand_val < a1 + a2:
            n -= 1  # 3X -> A + 2X
        elif rand_val < a1 + a2 + a3:
            n += 1  # B -> X
        else:
            n -= 1  # X -> B
            
        times.append(t)
        populations.append(n)
        
    return times, populations

# --- Part 2: 3D evolution of probability landscape ---
k_params = [3e-7, 1e-4, 1e-3 * 105, 1.0] 
N_TRAJECTORIES = 10000
T_MAX_SHORT = 50.0
TIME_SLICES = 100
N_BINS = 500
N0_short = 100

print("Running large-scale simulations for probability landscape...")
final_trajectories = np.zeros((N_TRAJECTORIES, TIME_SLICES))
time_points = np.linspace(0, T_MAX_SHORT, TIME_SLICES)

for i in range(N_TRAJECTORIES):
    if i % 1000 == 0:
        print(f"Simulating trajectory {i}/{N_TRAJECTORIES}")
    t, p = gillespie_schlogl_single(N0_short, k_params, T_MAX_SHORT)
    interp_p = np.interp(time_points, t, p, right=p[-1])
    final_trajectories[i, :] = interp_p

print("Building probability distribution surface P(n,t)...")
prob_surface = np.zeros((N_BINS, TIME_SLICES))
n_values = np.arange(N_BINS)
for i in range(TIME_SLICES):
    counts, _ = np.histogram(final_trajectories[:, i], bins=np.arange(N_BINS + 1) - 0.5)
    prob_surface[:, i] = counts / N_TRAJECTORIES

# --- 3D Visualization (corrected) ---
fig2 = plt.figure(figsize=(16, 12))
ax2 = fig2.add_subplot(111, projection='3d')

# *** Fix: Adjust meshgrid input order to match prob_surface dimensions ***
# prob_surface dimensions are (N_BINS, TIME_SLICES) -> (n, t)
# We need T_mesh and N_mesh to have the same dimensions
T_mesh, N_mesh = np.meshgrid(time_points, n_values)

ls = LightSource(azdeg=20, altdeg=45)

def update_surface(frame):
    ax2.clear()
    
    # Slice to get current frame and previous data
    current_T = T_mesh[:, :frame+1]
    current_N = N_mesh[:, :frame+1]
    current_P = prob_surface[:, :frame+1]
    
    # Apply lighting effects only when there is sufficient data
    if current_P.shape[1] > 1:
        rgb = ls.shade(current_P, cmap=plt.cm.magma, vert_exag=0.1, blend_mode='soft')
        surf = ax2.plot_surface(current_T, current_N, current_P, 
                               rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=True, alpha=1.0)
    else: # For the first frame, use colormap without lighting
        surf = ax2.plot_surface(current_T, current_N, current_P, 
                               rstride=1, cstride=1, cmap=plt.cm.magma,
                               linewidth=0, antialiased=True, alpha=1.0)

    ax2.set_title(f'Emergence of Bistability: $P(n,t)$ at t={time_points[frame]:.2f}', fontsize=20, pad=20)
    # Swap X and Y axis labels to match the new grid
    ax2.set_xlabel('Time (t)', fontsize=15, labelpad=20)
    ax2.set_ylabel('Particle Number (n)', fontsize=15, labelpad=20)
    ax2.set_zlabel('Probability P(n,t)', fontsize=15, labelpad=20)
    
    ax2.set_xlim(0, T_MAX_SHORT)
    ax2.set_ylim(0, N_BINS)
    ax2.set_zlim(0, np.max(prob_surface) * 1.1 if np.max(prob_surface) > 0 else 0.1)
    ax2.view_init(elev=25, azim=-150 + frame * 0.5) 

    ax2.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax2.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax2.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))

ani = animation.FuncAnimation(fig2, update_surface, frames=TIME_SLICES, interval=80)

print("Saving animation... (This may take a moment)")
ani.save('schlogl_bistability_evolution.gif', writer='pillow', fps=15)

