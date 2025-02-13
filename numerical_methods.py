#%% Import Functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as stats

#%% Numerical Methods 
############################## NUMERICAL METHODS #############################################

def euler_maruyama_ou(kappa, theta, sigma, X0, T, dt, num_samples=1):
    """
    Implements the Euler-Maruyama method for the Ornstein-Uhlenbeck process 

    Parameters:
    kappa : float  -> Mean reversion rate
    theta : float  -> Equilibrium value
    sigma : float  -> Noise strength
    X0 : float     -> Initial value
    T : float      -> Total simulation time
    dt : float     -> Time step size
    num_samples : int -> Number of trajectories to simulate (default is 1)

    Returns:
    t : numpy array -> Time array
    X : numpy array -> Simulated OU positions (shape: (num_samples, N))

    """

    N = int(T / dt)
    t = np.linspace(0, T, N)  
    X = np.full((num_samples, N), X0)  # Initial value is set as a delta function at posiition X0

    # Euler-Maruyama method
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt), size=num_samples)  # Brownian motion
        X[:, i] = X[:, i-1] + (-kappa * (X[:, i-1] - theta)) * dt + sigma * dW

    return t, X

################################
# Useful for Generalising code #
################################

def run_simulation(kappa, theta, sigma, X0, T, dt, num_samples=1, method="euler_maruyama"):
    """
    This is for a future case in which we have more methods/processes to simulate

    Returns:
    t : numpy array -> Time array
    X : numpy array -> Simulated OU positions (shape: (num_samples, N))

    """
    if method == "euler_maruyama":
        return euler_maruyama_ou(kappa, theta, sigma, X0, T, dt, num_samples)
    # We can add more methos here 
    else:
        raise Exception("You have not specified a method for simulation")

#%% Single Trajectory Functions
############################## SINGLE TRAJECTORY #################################

def simulate_ou_trajectory(kappa, theta, sigma, X0, T, dt, method="euler_maruyama"):
    """
    Simulates a SINGLE Ornstein-Uhlenbeck trajectory using the specified method
    Returns an array of time and the trajectory of the particle
    """
    t, X = run_simulation(kappa, theta, sigma, X0, T, dt, num_samples=1, method=method)
    return t, X.flatten() 

def plot_ou_trajectory(kappa, theta, sigma, X0, T, dt, method="euler_maruyama"):
    """
    Plots the trajectory of a SINGLE particle undergoing the Ornsetin Uhlenbeck process 
    Uses method specified (default is Euler-Maruyama)
    Plots the analytical solution for the mean and variance of the process
    """
    t, X = simulate_ou_trajectory(kappa, theta, sigma, X0, T, dt, method)

    # Analytical mean and variance
    X_mean = theta + (X0 - theta) * np.exp(-kappa * t)  # E[X_t]
    X_std = np.sqrt((sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * t))) 

    # For the plotting of variance, aesthetic value only 
    upper_bound = X_mean + 2 * X_std
    lower_bound = X_mean - 2 * X_std

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(t, X_mean, "r--", label=r"$E[X_t] = \theta + (X_0 - \theta) e^{-\kappa t}$", color="red")
    plt.fill_between(t, lower_bound, upper_bound, color="red", alpha=0.2,
                     label=r"$\pm 2\sqrt{Var[X_t]}$")
    plt.plot(t, X, label="Ornstein-Uhlenbeck Trajectory of a Single Particle")
    plt.title(f"OU Process: $\\kappa={kappa}$, $\\sigma={sigma}$, $X_0={X0}$")
    plt.xlabel("Time")
    plt.ylabel(r"X_t")
    plt.grid(True)
    plt.legend()
    plt.show()

#%% Multiple Trajectory Functions
############################## MULTIPLE TRAJECTORIES ###############################################
def simulate_ou_distribution(kappa, theta, sigma, X0, T, dt, num_samples, method="euler_maruyama"):
    """
    Simulates multiple Ornstein-Uhlenbeck trajectories for probability estimation.
    """
    return run_simulation(kappa, theta, sigma, X0, T, dt, num_samples, method=method)


def plot_multiple_ou_trajectories(kappa, theta, sigma, X0, T, dt, num_samples, method="euler_maruyama"):
    """
    Plots multiple Ornstein-Uhlenbeck trajectories on the same graph.

    Parameters:
    kappa : float  -> Mean reversion rate
    theta : float  -> Equilibrium value
    sigma : float  -> Noise strength
    X0 : float     -> Initial value
    T : float      -> Total simulation time
    dt : float     -> Time step size
    num_samples : int -> Number of trajectories to simulate
    method : str   -> Numerical method for integration (default: Euler-Maruyama)
    """
    t, X = simulate_ou_distribution(kappa, theta, sigma, X0, T, dt, num_samples, method=method)


    plt.figure(figsize=(10, 6))
    for i in range(num_samples):
        plt.plot(t, X[i, :], alpha=0.6)  

    X_mean = theta + (X0 - theta) * np.exp(-kappa * t)  
    plt.plot(t, X_mean, "r--", linewidth=2, label=r"$E[X_t]$ (Analytical Mean)")

    plt.title(f"Multiple OU Trajectories: $\\kappa={kappa}$, $\\sigma={sigma}$, $X_0={X0}$")
    plt.xlabel("Time")
    plt.ylabel(r"$X_t$")
    plt.grid(True)
    plt.legend()
    plt.show()

#%% Fokker-Planck 
####################################### FOKKER-PLANCK ##############################################

def fokker_planck_ou_anal(x_vals, t, kappa, theta, sigma, X0):
    """
    Computes the analytical solution of the Fokker-Planck equation for 
    the Ornstein-Uhlenbeck process at a given time t.

    Parameters:
    x_vals : numpy array -> Grid of x values for evaluating P(x, t)
    t : float  -> Time at which to compute P(x, t)
    kappa : float  -> Mean reversion rate
    theta : float  -> Equilibrium value
    sigma : float  -> Noise strength
    X0 : float     -> Initial value X(0)

    Returns:
    P_xt : numpy array -> Probability density P(x, t)
    """
    
    # its just a gaussian 
    mean_xt = theta + (X0 - theta) * np.exp(-kappa * t)
    var_xt = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * t))
    std_xt = np.sqrt(var_xt)
    
    #scipy stats takes care of normalisation 
    P_xt = stats.norm.pdf(x_vals, loc=mean_xt, scale=std_xt)
   
    func_form = str(r"$\rho(x,t) = \frac{1}{\sqrt{2 \pi \sigma_t^2}} \exp( -\frac{(x - \mu_t)^2}{2 \sigma_t^2})$")
    
    return P_xt, func_form


def fokker_planck_chang_cooper(kappa, theta, sigma, X0, T, dt, dx, x_range=(-4, 4)):
    """
    Solves the Fokker-Planck equation numerically using the Chang-Cooper finite difference method.
    """
    x_min, x_max = x_range
    x_vals = np.arange(x_min, x_max + dx, dx)
    Nx = len(x_vals)
    time_steps = int(T / dt)

    # Initialize probability distribution
    P_xt = np.zeros((time_steps, Nx))
    P_xt[0, np.argmin(np.abs(x_vals - X0))] = 1.0 / dx  


    D = sigma**2 / 2  
    a_vals = -kappa * (x_vals - theta)  
    b_vals = np.full_like(x_vals, D)

    for t in range(time_steps - 1):
        P_new = np.zeros(Nx)
        for i in range(1, Nx - 1):
            F_p = (a_vals[i] * dt) / dx  
            epsilon = F_p / (np.exp(F_p) - 1) if np.abs(F_p) > 1e-8 else 1.0  

            flux_right = b_vals[i] / dx**2 - a_vals[i] / (2 * dx) * (1 - epsilon)
            flux_left = b_vals[i] / dx**2 + a_vals[i] / (2 * dx) * (1 + epsilon)

            P_new[i] = P_xt[t, i] + dt * (flux_right * P_xt[t, i+1] - (flux_right + flux_left) * P_xt[t, i] + flux_left * P_xt[t, i-1])

        P_new[0] = P_new[1]
        P_new[-1] = P_new[-2]
        P_new /= np.sum(P_new * dx)

        P_xt[t + 1, :] = P_new

    return x_vals, P_xt


#%% Animation
############################## ANIMATION ########################################################
def animate_ou_distribution(kappa, theta, sigma, X0, T, dt, num_samples, method="euler_maruyama"):
    """
    Animates the evolution of the probability distribution using a histogram and numerical FP solution.
    """
    
    # Simulate multiple trajectories
    t, X = simulate_ou_distribution(kappa, theta, sigma, X0, T, dt, num_samples, method)

    time_indices = [0, int(len(t) / 2), len(t) - 1]
    time_labels = [0, T / 2, T]

    fig, ax = plt.subplots(figsize=(10, 6))
    hist_bins = 70
    
    def update(frame):

        ax.clear()

        x_vals, P_xt = fokker_planck_chang_cooper(kappa, theta, sigma, X0, T, dt, dx=0.1)
        P_analytical, func_form = fokker_planck_ou_anal(x_vals, T, kappa, theta, sigma, X0)
        
        # Ensure frame index is within bounds
        if frame >= len(P_xt):
            frame = len(P_xt) - 1

        # Histogram of particle simulations
        ax.hist(X[:, frame], bins=hist_bins, density=True, alpha=0.6, label=f"t = {t[frame]:.2f}")
        # Chang-Cooper Numerical Solution
        #ax.plot(x_vals, P_xt[frame, :], 'r-', label="Chang-Cooper FP Solution")
        # Analytical FP solution 
        ax.plot(x_vals, P_analytical, 'k--', label=f"Analytical Solution (from FP) {func_form}")

        ax.set_title(f"Evolution of OU Probability Distribution\n($\\theta={theta}$, $\\sigma={sigma}$)", fontsize=16)
        ax.set_xlabel(r"$X_t$", fontsize = 16)
        ax.set_xlim(np.min(X), np.max(X))
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability Density", fontsize = 16)
        ax.grid()
        ax.legend(loc = 2, fontsize = 16)

        # len(t) = 500 so choose frames like this 
        save_frames = [0, 250, 500]

        if frame in save_frames:
            filename = f"OU_t-{frame}.png"
            print(f"Saving frame {frame} as {filename}...")
            fig.savefig(filename, dpi=300)
            print("yo frame saved brotha")

    ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, len(t)-1, 200, dtype=int), interval=50)
    plt.show()

#%% Main
##################################### MAIN #########################################


kappa = 1.0    # Mean reversion rate
theta = 1.0   # Equilibrium mean
sigma = 1.0    # Noise strength
X0 = 3.0       # Initial value (delta function start)
T = 5.0        # Total simulation time
dt = 0.01      # Time step
num_samples = 5000  # Number of trajectories for probability estimation

# Choose method (currently only Euler-Maruyama, but easy to extend)
method = "euler_maruyama"

# Plot a single trajectory (example)
plot_ou_trajectory(kappa, theta, sigma, X0, T, dt, method=method)

# Plots many trajectories (just for personal understanding (looks crazy and takes ages to plot))
#plot_multiple_ou_trajectories(kappa, theta, sigma, X0, T, dt, num_samples, method=method)

# Animate the probability distribution evolution
animate_ou_distribution(kappa, theta, sigma, X0, T, dt, num_samples, method=method)

#############################################################################


# %%
