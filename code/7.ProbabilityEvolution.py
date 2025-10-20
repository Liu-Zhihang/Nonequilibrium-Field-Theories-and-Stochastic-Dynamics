"""
Probability Evolution and Phase Space Animation Demonstration

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint

# Set English font and mathematical formula display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  

class BirthDeathAnimator:
    """Birth-Death Process Animation Demonstration Class"""
    
    def __init__(self, N=15, mu=1.0, lam=1.2):
        """
        Initialize parameters
        N: State space size
        mu: Birth rate
        lam: Death rate
        """
        self.N = N
        self.mu = mu
        self.lam = lam
        self.Q = self._create_q_matrix()
        
        # Animation time settings
        self.t_max = 10
        self.n_frames = 150
        self.t_points = np.linspace(0, self.t_max, self.n_frames)
        
    def _create_q_matrix(self):
        """Create Q matrix for birth-death process"""
        Q = np.zeros((self.N, self.N))
        
        # Fill transition rates
        for n in range(1, self.N):
            if n < self.N - 1:
                Q[n+1, n] = self.mu * n  # Birth: n -> n+1
            Q[n-1, n] = self.lam * n     # Death: n -> n-1
            
        # Fill diagonal elements (negative exit rates)
        for n in range(self.N):
            Q[n, n] = -np.sum(Q[:, n])
            
        return Q
    
    def solve_master_equation(self, initial_state):
        """Solve master equation dP/dt = Q·P"""
        def master_eq(P, t):
            return self.Q @ P
        
        # Initial condition: all probability concentrated at initial_state
        P0 = np.zeros(self.N)
        P0[initial_state] = 1.0
        
        # Numerically solve ODE
        P_t = odeint(master_eq, P0, self.t_points)
        return P_t
    
    def create_probability_evolution_animation(self, initial_state=5, save_path=None):
        """
        Create probability evolution animation
        Display: 1) Probability distribution bar chart 2) Time series of key states
        """
        P_t = self.solve_master_equation(initial_state)
        
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'Probability Distribution Evolution Animation\n' + 
                    f'Birth-Death Process: μ={self.mu}, λ={self.lam}, Initial State={initial_state}', 
                    fontsize=14, fontweight='bold')
        
        # === Subplot 1: Probability distribution bar chart ===
        states = np.arange(self.N)
        bars = ax1.bar(states, P_t[0], alpha=0.8, color='skyblue', edgecolor='navy')
        ax1.set_xlim(-0.5, self.N-0.5)
        ax1.set_ylim(0, 1.0)
        ax1.set_xlabel('State n')
        ax1.set_ylabel('Probability P(n,t)')
        ax1.set_title('Instantaneous Probability Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Statistics text (avoid overlap)
        stats_text = ax1.text(0.7, 0.95, '', transform=ax1.transAxes, 
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # === Subplot 2: Time evolution of key states ===
        key_states = [0, max(1, initial_state-2), initial_state, 
                     min(initial_state+2, self.N-1), min(initial_state+4, self.N-1)]
        colors = ['red', 'orange', 'blue', 'green', 'purple']
        lines = []
        
        for state, color in zip(key_states, colors):
            line, = ax2.plot([], [], color=color, linewidth=2, 
                           label=f'State {state}', marker='o', markersize=2)
            lines.append((line, state))
        
        ax2.set_xlim(0, self.t_max)
        ax2.set_ylim(0, 1.0)
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('Probability P(n,t)')
        ax2.set_title('Key State Probabilities Over Time')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Time indicator line
        time_line = ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        def animate(frame):
            """Animation update function"""
            current_time = self.t_points[frame]
            current_prob = P_t[frame]
            
            # Update bar chart
            for bar, prob in zip(bars, current_prob):
                bar.set_height(prob)
                # Adjust color intensity based on probability
                intensity = prob / np.max(current_prob) if np.max(current_prob) > 0 else 0
                bar.set_color(plt.cm.viridis(intensity))
            
            # Calculate and display statistics
            mean_state = np.sum(states * current_prob)
            max_prob_state = states[np.argmax(current_prob)]
            
            stats_text.set_text(
                f'Time: {current_time:.2f}\n'
                f'Mean State: {mean_state:.2f}\n'
                f'Most Probable State: {max_prob_state}'
            )
            
            # Update time series
            for line, state in lines:
                if frame > 0:
                    x_data = self.t_points[:frame+1]
                    y_data = P_t[:frame+1, state]
                    line.set_data(x_data, y_data)
            
            # Update time indicator line
            time_line.set_xdata([current_time, current_time])
            
            return [*bars, stats_text, *[line for line, _ in lines], time_line]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames, 
                                     interval=80, blit=False, repeat=True)
        
        # Save animation
        if save_path:
            print(f"Saving probability evolution animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=12, dpi=100)
            print("Save complete!")
        
        plt.tight_layout()
        plt.show()
        return anim
    
    def create_phase_space_animation(self, initial_state=5, save_path=None):
        """
        Create phase space evolution animation
        Display comparison of statistical moment evolution trajectories for multiple initial states
        """
        # Calculate evolution for multiple different initial states
        initial_states = [1, max(2, initial_state-2), initial_state, 
                         min(initial_state+2, self.N-2), min(initial_state+4, self.N-2)]
        colors = ['red', 'orange', 'blue', 'green', 'purple']
        all_moments = []
        
        for init_state in initial_states:
            P_t = self.solve_master_equation(init_state)
            
            # Calculate statistical moments
            states = np.arange(self.N)
            moments = np.zeros((len(self.t_points), 3))
            
            for i, prob in enumerate(P_t):
                # First moment: mean
                moments[i, 0] = np.sum(states * prob)
                # Second central moment: variance
                moments[i, 1] = np.sum((states - moments[i, 0])**2 * prob)
                # Third normalized moment: skewness (simplified calculation)
                if moments[i, 1] > 1e-6:  # Avoid division by zero
                    moments[i, 2] = np.sum((states - moments[i, 0])**3 * prob) / (moments[i, 1]**(1.5))
                else:
                    moments[i, 2] = 0
            
            all_moments.append(moments)
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize multiple trajectory lines and current points
        lines = []
        points = []
        for i, (color, init_state) in enumerate(zip(colors, initial_states)):
            line, = ax.plot([], [], [], color=color, linewidth=2, alpha=0.8, 
                           label=f'Initial State {init_state}')
            point, = ax.plot([], [], [], 'o', color=color, markersize=6)
            lines.append(line)
            points.append(point)
        
        # Set axes
        ax.set_xlabel('Mean <n>')
        ax.set_ylabel('Variance Var(n)')
        ax.set_zlabel('Skewness Skew(n)')
        ax.set_title(f'Statistical Moment Evolution Comparison for Multiple Initial States\nμ={self.mu}, λ={self.lam}')
        ax.legend()
        
        # Dynamically set axis ranges (based on all trajectories)
        all_moments_combined = np.concatenate(all_moments, axis=0)
        margin = 0.1
        ax.set_xlim(np.min(all_moments_combined[:, 0])*(1-margin), 
                   np.max(all_moments_combined[:, 0])*(1+margin))
        ax.set_ylim(np.min(all_moments_combined[:, 1])*(1-margin), 
                   np.max(all_moments_combined[:, 1])*(1+margin))
        z_range = np.max(all_moments_combined[:, 2]) - np.min(all_moments_combined[:, 2])
        ax.set_zlim(np.min(all_moments_combined[:, 2])-z_range*margin, 
                   np.max(all_moments_combined[:, 2])+z_range*margin)
        
        # Information text (avoid obstruction)
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                             verticalalignment='top',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        def animate_phase(frame):
            """Phase space animation update function"""
            current_time = self.t_points[frame]
            
            # Update all trajectory lines and current points
            for i, (line, point, moments) in enumerate(zip(lines, points, all_moments)):
                if frame > 0:
                    line.set_data_3d(moments[:frame+1, 0], moments[:frame+1, 1], moments[:frame+1, 2])
                    point.set_data_3d([moments[frame, 0]], [moments[frame, 1]], [moments[frame, 2]])
            
            # Update time information (display information from main trajectory)
            main_moments = all_moments[2]  # Use middle initial state (index 2) as main display
            time_text.set_text(
                f'Time: {current_time:.2f}\n'
                f'Main Trajectory (Initial={initial_state}):\n'
                f'  Mean: {main_moments[frame, 0]:.2f}\n'
                f'  Variance: {main_moments[frame, 1]:.2f}\n'
                f'  Skewness: {main_moments[frame, 2]:.2f}'
            )
            
            return [*lines, *points, time_text]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_phase, frames=self.n_frames, 
                                     interval=80, blit=False, repeat=True)
        
        # Save animation
        if save_path:
            print(f"Saving phase space animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=12, dpi=100)
            print("Save complete!")
        
        plt.show()
        return anim

# Demonstration usage
if __name__ == "__main__":
    print("=== Section 4.3 Probability Evolution and Phase Space Animation Demonstration ===\n")
    
    # Create animation demonstrator
    animator = BirthDeathAnimator(N=15, mu=1.0, lam=1.2)
    
    print("1. Generating probability evolution animation...")
    anim1 = animator.create_probability_evolution_animation(
        initial_state=5, 
        save_path="probability_evolution.gif"
    )
    
    print("\n2. Generating phase space evolution animation...")
    anim2 = animator.create_phase_space_animation(
        initial_state=5, 
        save_path="phase_space_evolution.gif"
    )
    
    print("\nAnimation generation complete!")