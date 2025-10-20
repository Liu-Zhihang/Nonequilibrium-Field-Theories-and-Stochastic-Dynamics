import matplotlib.pyplot as plt
import numpy as np

def plot_feynman_rules(figsize=(12, 5)):
    """
    Plot the basic Feynman rules for phi^4 theory in the J-D formalism.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Feynman Rules for Stochastic $\phi^4$ Theory (MSRJD)', fontsize=16)

    # --- 1. Response Propagator G_0 ---
    ax1.set_title(r'Response Propagator $G_0 = \langle \phi \tilde{\phi} \rangle_0$')
    ax1.plot([0.1, 0.9], [0.5, 0.5], 'k-', lw=2)
    ax1.arrow(0.5, 0.5, 0.01, 0, head_width=0.08, head_length=0.08, fc='k', ec='k')
    ax1.text(0, 0.55, r'$\phi(x,t)$', fontsize=12, ha='center')
    ax1.text(1, 0.55, r'$\tilde{\phi}(x^\prime,t^\prime)$', fontsize=12, ha='center')
    ax1.text(0.5, 0.3, r'$G_0(x,t; x^\prime,t^\prime)$', fontsize=14, ha='center', color='blue')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # --- 2. Correlation Propagator C_0 ---
    ax2.set_title(r'Correlation Propagator $C_0 = \langle \phi \phi \rangle_0$')
    x = np.linspace(0.1, 0.9, 100)
    y = 0.5 + 0.05 * np.sin(x * 5 * np.pi)
    ax2.plot(x, y, 'k-', lw=2)
    ax2.text(0, 0.55, r'$\phi(x,t)$', fontsize=12, ha='center')
    ax2.text(1, 0.55, r'$\phi(x^\prime,t^\prime)$', fontsize=12, ha='center')
    ax2.text(0.5, 0.3, r'$C_0(x,t; x^\prime,t^\prime)$', fontsize=14, ha='center', color='green')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # --- 3. Interaction Vertex ---
    ax3.set_title(r'Interaction Vertex from $S_{int} \propto u \tilde{\phi} \phi^3$')
    center = (0.5, 0.5)
    # Vertex
    ax3.plot(center[0], center[1], 'ko', markersize=10)
    # Response field line (incoming)
    ax3.arrow(0.9, 0.5, -0.38, 0, head_width=0.08, head_length=0.08, fc='k', ec='k', lw=2)
    # Three correlation field lines (outgoing)
    angles = [np.pi * 5/6, np.pi, np.pi * 7/6]
    for angle in angles:
        x_end = center[0] + 0.4 * np.cos(angle)
        y_end = center[1] + 0.4 * np.sin(angle)
        x_wave = np.linspace(center[0], x_end, 50)
        y_wave = np.linspace(center[1], y_end, 50)
        offset = 0.03 * np.sin(np.linspace(0, 3*np.pi, 50))
        perp_vec = np.array([-(y_end-center[1]), x_end-center[0]])
        perp_vec /= np.linalg.norm(perp_vec)
        ax3.plot(x_wave + offset*perp_vec[0], y_wave + offset*perp_vec[1], 'k-', lw=2)
    ax3.text(0.5, 0.2, 'Coupling Strength $-Lu$', fontsize=14, ha='center', color='red')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Run the plotting function
plot_feynman_rules()