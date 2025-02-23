import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# live love laugh
class Simulation: 

    def __init__(self, num_particles, steps, dt, k=1.0): 

        self.stochastic = StochasticMechanics(k) # stochastic mechanics object
        self.quantum = QuantumMechanics(k) # quantum mechanics object
        self.num_particles = num_particles
        self.steps = steps
        self.dt = dt
        self.dx = np.sqrt(self.dt)  # Step size from requirement dx^2/dt = 1
    
    def simulate_free_particle(self, k, initial_positions = 'gaussian', plotting = True):
        """
        Dear Paul, 
        I am i third year BSc project student, halfway through my term 2 project with Dimitri Vvendesky. 
        I have been having some issues with my project partner Matthew Howarth and would like to do the project alone.
        Please let me know if this is possible, or if i have to live a docstring-less life. 
        Best Wishes, 
        Ria Ranjitkar 
        """

        walk = RandomWalk(self.num_particles, self.steps, self.dt, initial_positions=initial_positions)
        stochastic = StochasticMechanics(k) # stochastic mechanics object
        quantum = QuantumMechanics(k) # quantum mechanics object
        time_grid, positions = walk.wiener_process(stochastic.free_particle_drift) # update 
        x_grid = np.linspace(np.min(positions), np.max(positions), 1000)
        psi_squared = quantum.free_particle_psi_squared(x_grid, time_grid) # psi(x,t)
        if plotting: 
            self.plot_trajectories(time_grid, positions, 5)
            self.plot_std(time_grid, positions, psi_squared)
            self.animate_simulation(time_grid, positions, psi_squared)

    # def simulate_tunneling  # simulation number 2 to be implemented.

    def compute_momentum(self, k, initial_positions='gaussian', plotting=True):
        """
        Compute momentum as q(t)/t using only the final time step, which is valid for t -> infinity

        """
        walk = RandomWalk(self.num_particles, self.steps, self.dt, initial_positions=initial_positions)
        stochastic = StochasticMechanics(k)
        quantum = QuantumMechanics(k)
        # time grid is a small array going from 0 -> dt 
        time_grid, positions = walk.wiener_process(stochastic.free_particle_drift)
    
        # could use the first element of time grid also but then get div 0 error for t = 0 
        momenta = positions / time_grid
    
        p_griddie = np.linspace(-10,10,1000)
        psi_squared_momentum = quantum.free_particle_momentum_psi_squared(p_griddie, time_grid)
        
        if plotting: 
            self.plot_momentum_distribution(momenta)
            self.animate_momentum(time_grid, positions, psi_squared_momentum)
    
    def animate_simulation(self, time_grid, positions, psi_squared, num_bins=30, frame_step = 10, interval=50):  
        """
        Animate the distribution of particle positions over time.

        Parameters:
            int num_bins: Number of bins for the histogram.
            int frame_step: Time interval between frames.
            int interval: Animation speed (in milliseconds).
        """
        time_steps = self.steps
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Position", fontsize=16)
        ax.set_ylabel("Probability Density", fontsize=16) 
        x_min = np.min(positions)
        x_max = np.max(positions)
        x_values = np.linspace(x_min, x_max, 1000)
        # Function to update the animation frame
        def update(frame):
            ax.clear()  # Clear previous frame
            positions_at_time = positions[:, frame]  # Get particle positions at this time step
            # Plot histogram
            hist_data, bin_edges, _ = ax.hist(positions_at_time, bins=num_bins, density = True, alpha=0.7, color='blue', edgecolor='black', label="Particle Distribution")
            # Plot psi-squared
            x_min = np.min(positions_at_time)
            x_max = np.max(positions_at_time)
            psi_values = psi_squared[:, frame]
            ax.plot(x_values, psi_values, color='red', linestyle='dashed', linewidth=2, label=r"$|\psi(x,t)|^2$")
            ax.set_xlabel("Position")
            ax.set_ylabel("Probability Density")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, 1)
            ax.set_title(f"Particle Position Distribution at t = {time_grid[frame]:.3f}")
            ax.legend()

       
        ani = animation.FuncAnimation(fig, update, frames=range(0, time_steps, frame_step), interval=interval)

    
        plt.show()
    
    def plot_trajectories(self, time_grid, positions, number): 
        """
        Plot trajectories of randomly selected particles over time.

        Parameters:
            int number: Number of particles to plot.
        """
        randoms = np.random.randint(0, self.num_particles, number)
        for i in randoms:
            plt.plot(time_grid, positions[i], lw = 3)
            plt.xlabel('$t$', fontsize=16)
            plt.ylabel('$x$', fontsize=16)
        plt.show()
    
    def plot_momentum_distribution(self, time_grid, momenta, time_index, num_bins=50):
        """
        Plot the distribution of particle momenta at a specific time step.

        Parameters:
            int time_index: Index of the time step to plot.
            int num_bins: Number of bins for the histogram.
        """
        if time_index >= self.steps:
            raise ValueError('Invalid time index.')
        momenta_at_time = momenta[:, time_index]
        hist, bin_edges = np.histogram(momenta_at_time, bins=num_bins, density=True)
        plt.figure(figsize=(8, 5))
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7,color='black')
        plt.xlabel("Momentum", fontsize=16)
        plt.ylabel("Probability Density",   fontsize=16)
        plt.title(f"Particle Momentum Distribution at t = {time_grid[time_index]:.3f}")
        plt.show()
    

    def plot_distribution(self, time_grid, positions, time_index, num_bins=50):
        """
        Plot the distribution of particle positions at a specific time step.

        Parameters:
            int time_index: Index of the time step to plot.
            int num_bins: Number of bins for the histogram.
        """
        if time_index >= self.steps:
            raise ValueError('Invalid time index.')
        positions_at_time = positions[:, time_index]
        hist, bin_edges = np.histogram(positions_at_time, bins=num_bins, density=True)
        plt.figure(figsize=(8, 5))
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7,color='black')
        plt.xlabel("Position", fontsize=16)
        plt.ylabel("Probability Density",   fontsize=16)
        plt.title(f"Particle Position Distribution at t = {time_grid[time_index]:.3f}")
        plt.show()

    def plot_std(self, time_grid, positions, psi_squared):
        """
        Plot the standard deviation of particle positions over time.

        Parameters:
            bool overlay_psi: Whether to overlay the quantum mechanical standard deviation.
        """
        std = np.std(positions, axis=0)
        plt.plot(time_grid, std, label='Stochastic STD', color='red')
        std_schr = np.sqrt((1 + time_grid ** 2) / 2)
        plt.plot(time_grid, std_schr, label=r'$|\psi(x,t)|^2$', color='black')
        plt.xlabel('$t$', fontsize=16)
        plt.ylabel('$\\sigma$', fontsize=16)
        plt.legend()
        plt.show()

    def animate_momentum(self, time_grid, positions, psi_sqaured_momentum, num_bins=30, frame_step=10, interval=50):  
        """
        Animate the instantaneous momentum distribution over time.
        Here, we compute momentum as q(t)/t for t>0.
        """
        time_steps = self.steps
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Momentum", fontsize=16)
        ax.set_ylabel("Probability Density", fontsize=16)
        
        def update(frame):
            # Skip the first frame to avoid division by zero.
            if time_grid[frame] == 0:
                return
            ax.clear()
    
            # Compute instantaneous momentum at time t (for each particle)
            current_momenta = positions[:, frame] / time_grid[frame]
            hist_data, bin_edges, _ = ax.hist(current_momenta, bins=num_bins, density=True, 
                                              alpha=0.7, color='green', label="Simulated Momentum")
            p_griddie = np.linspace(-10, 10, 1000)
            ax.plot(p_griddie, psi_sqaured_momentum, 'r--', linewidth=2, label=r'$|\psi(p,t)|^2$')
            ax.set_xlabel("Momentum")
            ax.set_ylabel("Probability Density")
            ax.set_title(f"Momentum Distribution at t = {time_grid[frame]:.3f}")
            ax.set_xlim(-10, 10)
            ax.set_ylim(0,0.7)
            ax.legend()
    
        ani = animation.FuncAnimation(fig, update, frames=range(1, time_steps, frame_step), interval=interval)
        plt.show()
    
    def plot_momentum_distribution(self, momenta, num_bins=50):
        """
        Plot the distribution of particle momenta.
        Here, momenta is a 1D array computed from the final time step.
        """
        plt.figure(figsize=(8, 5))
        hist, bin_edges = np.histogram(momenta, bins=num_bins, density=True)
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7, color='black')
        plt.xlabel("Momentum", fontsize=16)
        plt.ylabel("Probability Density", fontsize=16)
        plt.title("Final Momentum Distribution")
        plt.show()

class QuantumMechanics:
    def __init__(self, k=1.0):
        """
        Parameters:
            float k: Wave number
        """
        self.k = k

    def free_particle_psi_squared(self, x, t):
        """
        Vectorized squared wavefunction (PDF) for a free particle.

        Parameters:
            np.ndarray x: Positions (array)
            np.ndarray t: Times (array)

        Returns:
            np.ndarray: Squared wavefunction evaluated on the grid
        """
        x = np.asarray(x)[:, np.newaxis]  # Reshape x to (Nx, 1) for broadcasting
        t = np.asarray(t)[np.newaxis, :]  # Reshape t to (1, Nt) for broadcasting
        print(t)

        return (1 / np.sqrt(np.pi * (1 + t ** 2))) * np.exp(-(x - self.k * t) ** 2 / (1 + t ** 2))
    
    def free_particle_std(self, t):
        """
        Standard deviation of the wavefunction for a free particle.

        Parameters:
            float t: Time

        Returns:
            float: Standard deviation
        """
        return np.sqrt((1 + t ** 2) / 2)

    def free_particle_momentum_psi_squared(self, p, t):
        """
        Initial Gaussian wave packet: 
            psi(0,x) = 1/(pi^(1/4)) exp(-x^2/2) exp(i*k*x)
        FT:
            psi(tilda)(p) = 1/(pi^(1/4)) exp(-(p - k)^2/2)
        so that mod psi(tilda)(p) squared = 1/sqrt(pi) exp(-(p - k)^2).
        
        """

        p = np.asarray(p)[:, np.newaxis]
       
        return  (1/ np.sqrt(np.pi)) * np.exp((-(p - self.k)**2)) 

class StochasticMechanics:
    def __init__(self, k=1.0, psi_xt = None):
        """
        Parameters:
            float k: Wave number
            array psi_xt: Wave function array in the spatiotemporal grid
        """
        self.k=k
        self.psi_xt = psi_xt # note the free particle does not need this
    
    def free_particle_drift(self,x,t):
        """
        Drift velocity for a free particle.

        Parameters:
            float x: Position
            float t: Time

        Returns:
            float: Drift velocity
        """
        return self.k + (t-1)*(x-self.k*t)/(1+t**2)

class RandomWalk:
    def __init__(self, num_particles, steps, dt, k=1.0, initial_positions='delta'):
        """
        Parameters:
            int num_particles: Number of particles to simulate.
            float steps: Number of time steps.
            float dt: time grid resolution; dx follows.
            float k: wave number
            str initial_positions: Initial position distribution. Options are 'delta' and 'gaussian'.
        """
        self.num_particles = num_particles
        self.dt = dt
        self.steps = steps
        self.dx = np.sqrt(self.dt)  # Step size from requirement dx^2/dt = 1
        self.time_grid = np.arange(0.00000000001, dt * steps, dt)
        if initial_positions == 'delta':
            self.positions = np.zeros((num_particles, steps))
        elif initial_positions == 'gaussian':
            self.positions = np.random.normal(scale = 1/np.sqrt(2), size=(num_particles, steps))

    def wiener_process(self, drift_velocity):
        """
        Simulates a Wiener process with drift for multiple particles.

        Parameters:
            callable drift_velocity (Callable[[float, float], float]): Function returning drift velocity given (x, t).

        Returns:
            tuple: (time_grid, positions) where positions is a 2D array of shape (num_particles, steps).
        """
        for i in range(1, self.steps):  # Start from step 1 since step 0 is initialized to 0
            t = self.time_grid[i]
            # Compute drift velocity for each particle
            prob_drift = np.array([drift_velocity(x, t) for x in self.positions[:, i - 1]]) * (self.dt / self.dx)
            # Ensure consistency condition holds for all particles
            if np.any(np.abs(prob_drift) >= 1):
                raise ValueError('Consistency condition not met for at least one particle.')
            p_left = 0.5 * (1 - prob_drift)  # Compute left probability per particle
            deviate = np.random.uniform(0, 1, num_particles)  
            # Update positions
            move_left = deviate < p_left
            self.positions[:, i] = self.positions[:, i - 1] - self.dx * move_left + self.dx * (~move_left) 
        return self.time_grid, self.positions

class UncertaintyPrinciple:
    def __init__(self, num_particles, steps, dt, k=1.0):
        self.stochastic = StochasticMechanics(k)  # stochastic mechanics object
        self.quantum = QuantumMechanics(k)        # quantum mechanics object
        self.num_particles = num_particles
        self.steps = steps
        self.dt = dt
        self.dx = np.sqrt(self.dt)  # Step size from requirement dx^2/dt = 1

    def compute_statistics(self, k, initial_positions='gaussian'):
        """_summary_

        Args:
            k (_type_): _description_
            initial_positions (str, optional): _description_. Defaults to 'gaussian'.
        """
        walk = RandomWalk(self.num_particles, self.steps, self.dt, initial_positions=initial_positions)
        stochastic = StochasticMechanics(k)
        quantum = QuantumMechanics(k)
        time_grid, positions = walk.wiener_process(stochastic.free_particle_drift)
        final_t = time_grid[-1]
        final_positions = positions[:, -1]
        final_momenta = final_positions / final_t

        # Compute ensemble statistics:
        self.mean_x = np.mean(final_positions)
        self.mean_p = np.mean(final_momenta)
        self.sigma_x = np.std(final_positions)
        self.sigma_p = np.std(final_momenta)
        self.covariance = np.mean((final_positions - self.mean_x) * (final_momenta - self.mean_p))
    
    def schrodinger_uncertainty(self):
        """
        note: h bar = m = 1 
        """
        self.compute_statistics(k=1.0, initial_positions='gaussian')
        variance_product = self.sigma_x**2 * self.sigma_p**2
        lower_bound = 0.25 + self.covariance**2
        return variance_product, lower_bound

    def print_results(self):
        variance_product, lower_bound = self.schrodinger_uncertainty()
        print(rf"$\sigma_x^2\,\sigma_p^2 = {variance_product:.4f}$")
        print(rf"$\text{{Lower bound}} = \frac{{1}}{{4}} + \text{{cov}}^2(x,p) = {lower_bound:.4f}$")
        print(rf"$\sigma_x = {self.sigma_x:.4f}, \quad \sigma_p = {self.sigma_p:.4f}, \quad \text{{cov}}(x,p) = {self.covariance:.4f}$")

num_steps = 2000
dt = 0.001
num_particles = 5000

#simulation = Simulation(num_particles, num_steps, dt)
#simulation.simulate_free_particle(1.0)

simulation = Simulation(num_particles, num_steps, dt)
simulation.compute_momentum(1.0)

#uncertainty = UncertaintyPrinciple(num_particles, num_steps, dt)
#uncertainty.print_results()