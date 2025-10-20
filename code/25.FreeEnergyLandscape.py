import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib to support Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei']  # Specify default font
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem that the minus sign '-' is displayed as a square when saving the image

def gld_potential(phi, r, u):
    """
    Calculate the Ginzburg-Landau local free energy density f(φ).
    
    Parameters:
    phi (np.ndarray): Order parameter values
    r (float): Temperature parameter
    u (float): Stability parameter
    
    Returns:
    np.ndarray: Free energy density f(φ)
    """
    return 0.5 * r * phi**2 + 0.25 * u * phi**4

# Define parameters
u = 1.0  # Stability parameter, kept positive
phi_range_3d = np.linspace(-2.5, 2.5, 100)

# Create a 3D figure with dark background
fig = plt.figure(figsize=(14, 10), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True
ax.xaxis.pane.set_color('black')
ax.yaxis.pane.set_color('black')
ax.zaxis.pane.set_color('black')
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.tick_params(axis='z', colors='black')

# Set title and axis labels with detailed information
ax.set_title('3D Ginzburg-Landau Free Energy Landscape $f(\\phi, r)$\nHigh Temperature (r > 0) → Low Temperature (r < 0)', 
             color='black', fontsize=14, pad=20)
ax.set_xlabel('Order Parameter $\\phi$\n(-2.5 to 2.5)', color='black', labelpad=10)
ax.set_ylabel('Parameter $r$\n(Positive: T > T$_c$ | Negative: T < T$_c$)', color='black', labelpad=10)
ax.set_zlabel('Free Energy Density $f(\\phi, r)$', color='black', labelpad=10)

# Create two separate surfaces for r > 0 and r < 0 cases
# r > 0 (High temperature, disordered phase)
r_range_positive = np.linspace(0.1, 2.0, 50)
Phi_3d_pos, R_3d_pos = np.meshgrid(phi_range_3d[::2], r_range_positive)
F_3d_positive = 0.5 * R_3d_pos * Phi_3d_pos**2 + 0.25 * u * Phi_3d_pos**4

# r < 0 (Low temperature, ordered phase)
r_range_negative = np.linspace(-2.0, -0.1, 50)
Phi_3d_neg, R_3d_neg = np.meshgrid(phi_range_3d[::2], r_range_negative)
F_3d_negative = 0.5 * R_3d_neg * Phi_3d_neg**2 + 0.25 * u * Phi_3d_neg**4

# Plot both surfaces
surf1 = ax.plot_surface(Phi_3d_pos, R_3d_pos, F_3d_positive, cmap='viridis', alpha=0.8, label='r > 0 (T > T_c) - Disordered Phase')
surf2 = ax.plot_surface(Phi_3d_neg, R_3d_neg, F_3d_negative, cmap='plasma', alpha=0.8, label='r < 0 (T < T_c) - Ordered Phase')

# Add annotations for key features
# For r > 0 case: single minimum at phi = 0
ax.plot([0], [1.0], [0], 'ro', markersize=10)
ax.text(0.8, 1.5, 0.4, 'Unique minimum\n$\\phi = 0$\n(Disordered Phase)', color='white', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', edgecolor='white'))

# Add critical point indicator
ax.plot([0], [0], [0], 'wo', markersize=8)
ax.text(0.9, 0.6, 0.8, 'Critical Point\nr = 0\nPhase Transition', 
        color='white', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', edgecolor='yellow'))

# For r < 0 case: degenerate minima at phi = ±√(-r/u)
phi_min_pos = np.sqrt(-(-1.0) / u)  # For r = -1.0
phi_min_neg = -phi_min_pos
f_min = 0.5 * (-1.0) * phi_min_pos**2 + 0.25 * u * phi_min_pos**4
ax.plot([phi_min_pos, phi_min_neg], [-1.0, -1.0], [f_min, f_min], 'co', markersize=10)
ax.text(phi_min_pos+0.5, -1.5, f_min+0.5, 'Degenerate minima\n$\\phi = \\pm\\sqrt{-r/u}$\n(Ordered Phase)', 
        color='white', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', edgecolor='white'))

plt.tight_layout()
plt.show()