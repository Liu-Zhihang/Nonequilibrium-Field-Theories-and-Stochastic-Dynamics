import numpy as np
import matplotlib.pyplot as plt

def simulate_quadratic_variation(T=1.0, N_steps_list=[100, 1000, 10000, 100000], num_trials=1000):
    """
    Simulate and demonstrate convergence of quadratic variation
    """
    results_mean = []
    results_std = []
    
    print("N_steps\t\tMean(QV)\tStd(QV)")
    print("-" * 40)

    for N in N_steps_list:
        dt = T / N
        qvs = []
        for _ in range(num_trials):
            # Generate increments of a Brownian motion path
            # 2D=1, so variance = dt
            dW = np.random.normal(0, np.sqrt(dt), N)
            
            # Calculate quadratic variation
            qv = np.sum(dW**2)
            qvs.append(qv)
        
        mean_qv = np.mean(qvs)
        std_qv = np.std(qvs)
        results_mean.append(mean_qv)
        results_std.append(std_qv)
        print(f"{N:<10}\t{mean_qv:.6f}\t{std_qv:.6f}")
        
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Mean convergence plot
    plt.subplot(1, 2, 1)
    plt.plot(N_steps_list, results_mean, 'o-', label='Mean of Simulated QV')
    plt.axhline(T, color='r', linestyle='--', label=f'Theoretical Value T={T}')
    plt.xscale('log')
    plt.xlabel('Number of Steps (N)')
    plt.ylabel('Mean of Quadratic Variation')
    plt.title('Convergence of Mean QV')
    plt.legend()
    plt.grid(True)

    # Standard deviation convergence plot
    plt.subplot(1, 2, 2)
    plt.plot(N_steps_list, results_std, 'o-', label='Std Dev of Simulated QV')
    # Theoretically standard deviation is proportional to 1/sqrt(N)
    # Var = 2T^2/N => Std = T*sqrt(2/N)
    theoretical_std = T * np.sqrt(2 / np.array(N_steps_list))
    plt.plot(N_steps_list, theoretical_std, 'r--', label=r'Theoretical scaling $\propto 1/\sqrt{N}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Steps (N)')
    plt.ylabel('Standard Deviation of QV')
    plt.title('Convergence of Std Dev of QV')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run simulation
simulate_quadratic_variation(T=1.0)