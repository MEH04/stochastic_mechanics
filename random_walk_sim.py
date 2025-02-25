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
    
    def simulate_free_particle(self, k, initial_positions = 'gaussian', plotting = None, print_time = None):
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
        momenta = positions / time_grid
        p_grid = np.linspace(-10,10,1000)
        psi_squared_momentum = quantum.free_particle_momentum_psi_squared(p_grid, time_grid)

        if print_time is not None:
            idx = np.argmin(np.abs(time_grid - print_time))
            mean_position = np.mean(positions[:, idx])
            mean_momentum = np.mean(momenta[:, idx])
            std_position = np.std(positions[:, idx])
            std_momentum = np.std(momenta[:, idx])
            print(f"At time {time_grid[idx]:.3f}: Mean Position = {mean_position:.3f}, Mean Momentum = {mean_momentum:.3f}, STD Position = {std_position:.3f}, STD Momentum = {std_momentum:.3f}")

        if plotting: 
            self.plot_trajectories(time_grid, positions, 5)
            self.plot_std(time_grid, positions, psi_squared)
            self.animate_position_momentum(time_grid, positions, psi_squared, psi_squared_momentum)

            

    # def simulate_tunneling  # simulation number 2 to be implemented.
   

    def animate_position_momentum(self, time_grid, positions, psi_squared, psi_squared_momentum, num_bins=30, frame_step=10, interval=50):  
        """
        Animate the position and momentum distribytions over time in a side-by-side subplot.

        Parameters:
            time_grid: Array of time values.
            positions: Array of particle positions over time.
            psi_squared: Array of |ψ(x,t)|² values over time.
            psi_squared_momentum: Array of |ψ(p,t)|² values over time.
            num_bins: Number of bins for the histograms.
            frame_step: Step size for frames in the animation.
            interval: Animation speed (in milliseconds).
        """
        time_steps = self.steps 

        # Create figure and subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Labels and limits
        axs[0].set_xlabel("Position", fontsize=14)
        axs[0].set_ylabel("Probability Density", fontsize=14)
        axs[1].set_xlabel("Momentum", fontsize=14)
        axs[1].set_ylabel("Probability Density", fontsize=14)

        # Define x and momentum grid values
        x_min, x_max = np.min(positions), np.max(positions)
        x_values = np.linspace(x_min, x_max, 1000)
        p_griddie = np.linspace(-10, 10, 1000)
        analytical_var_product_list = []
        lower_bound_list = []
        varaince_product_list = []
        times = []

        # Function to update the animation frame
        def update(frame):
            if time_grid[frame] == 0:
                return  # Skip frame to avoid division by zero in momentum

            for ax in axs:
                ax.clear()  # Clear previous frame

            # --- POSITION DISTRIBUTION (Left subplot) ---
            positions_at_time = positions[:, frame]  # Particle positions at this time step
            hist_data, bin_edges, _ = axs[0].hist(positions_at_time, bins=num_bins, density=True, 
                                                alpha=0.7, color='blue', edgecolor='black', label="Particle Distribution")
            psi_values = psi_squared[:, frame]
            y_min, y_max = np.min(psi_values), np.max(psi_values)
            axs[0].plot(x_values, psi_values, color='red', linestyle='dashed', linewidth=2, label=r"$|\psi(x,t)|^2$")
            axs[0].set_xlim(x_min, x_max)
            axs[0].set_ylim(y_min, y_max*1.1)
            axs[0].set_title(f"Position Distribution at t = {time_grid[frame]:.3f}")
            axs[0].legend()

            # --- MOMENTUM DISTRIBUTION (Right subplot) ---
            current_momenta = positions[:, frame] / time_grid[frame]
            current_positions = positions[:, frame]
            mean_x = np.mean(current_positions)
            mean_p = np.mean(current_momenta)
            sigma_x = np.std(current_positions)
            sigma_p = np.std(current_momenta)
            covariance = np.mean((current_positions - mean_x) * (current_momenta - mean_p))

            variance_product = sigma_x**2 * sigma_p**2
            lower_bound = 0.25 + covariance**2
            analytical_var_product = (1 + time_grid[frame]**2) * (1/ (2 * np.pi))

            varaince_product_list.append(variance_product)
            lower_bound_list.append(lower_bound)
            analytical_var_product_list.append(analytical_var_product)
            times.append(time_grid[frame])

            textstr = (
                r"$\left(\Delta x\right)^2 \left(\Delta p\right)^2 = $" + f"{variance_product:.2f}\n"
                r"$\mathrm{Cov}(x,p)^2 + \frac{1}{4} = $" + f"{lower_bound:.2f} \n"
                "Analytical" + r"$\left(\Delta x\right)^2 \left(\Delta p\right)^2 = $" +  f"{analytical_var_product:.2f}"
                #f"mean_x = {mean_x:.2f}, mean_p = {mean_p:.2f}\n"
                #f"sigma_x = {sigma_x:.2f}, sigma_p = {sigma_p:.2f}\n"
            )

            hist_data, bin_edges, _ = axs[1].hist(current_momenta, bins=num_bins, density=True, 
                                                alpha=0.7, color='green', edgecolor = 'black', label="Simulated Momentum")
            axs[1].plot(p_griddie, psi_squared_momentum, 'r--', linewidth=2, label=r'$|\psi(p,t)|^2$')
            axs[1].set_xlim(-10, 10)
            axs[1].set_ylim(0, 0.7)
            axs[1].set_title(f"Momentum Distribution at t = {time_grid[frame]:.3f}")

            # Add annotation box with variance information
            props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray", alpha=0.7)
            axs[1].text(0.05, 0.95, textstr, transform=axs[1].transAxes, fontsize=10, verticalalignment='top', bbox=props)
            axs[1].legend()

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=range(1, time_steps, frame_step), interval=interval)

        plt.tight_layout()
        plt.show()

        plt.plot(times, varaince_product_list, label = 'Simulated Variance Product')
        plt.plot(times, lower_bound_list, label = 'Lower Bound')
        plt.plot(times, analytical_var_product_list, label = 'Analytical Variance Product')
        plt.legend()
        plt.xlabel('Time')
        plt.show()

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
            current_positions = positions[:, frame]
            mean_x = np.mean(current_positions)
            mean_p = np.mean(current_momenta)
            sigma_x = np.std(current_positions)
            sigma_p = np.std(current_momenta)
            covariance = np.mean((current_positions - mean_x) * (current_momenta - mean_p))

            variance_product = sigma_x**2 * sigma_p**2
            lower_bound = 0.25 + covariance**2

            textstr = (
                r"$\left(\Delta x\right)^2 \left(\Delta p\right)^2 = $" + f"{variance_product:.2f}\n"
                r"$\mathrm{Cov}(x,p)^2 + \frac{1}{4} = $" + f"{lower_bound:.2f}"
            )

            hist_data, bin_edges, _ = ax.hist(current_momenta, bins=num_bins, density=True, 
                                              alpha=0.7, color='green', label="Simulated Momentum")
            
            p_griddie = np.linspace(-10, 10, 1000)
            #psi_frame = psi_sqaured_momentum[:, frame]
            ax.plot(p_griddie, psi_sqaured_momentum, 'r--', linewidth=2, label=r'$|\psi(p,t)|^2$')
            ax.set_xlabel("Momentum")
            ax.set_ylabel("Probability Density")
            ax.set_title(f"Momentum Distribution at t = {time_grid[frame]:.3f}")
            ax.set_xlim(-10, 10)
            ax.set_ylim(0,0.7)
            props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray", alpha=0.7)
            ax.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,fontsize=10, verticalalignment='top', bbox=props)
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
        x = np.linspace(-10, 10, 1000)
        x = np.asarray(x)[:, np.newaxis]
        psi_x = (1/(np.pi)**(1/4)) * 1/np.sqrt(1 + 1j*t) * np.exp((-1 * (1 - 1j*t)*(x - self.k*t)**2 / 2*(1 + 1j*t**2))) + 1j*self.k*(x - (self.k * t/2))
        momenta = np.fft.fftfreq(len(x), d = 1/1000)
        #p = np.fft.fftshift(momenta)
        psi_tilda_p = np.fft.fft(psi_x)
        p = np.asarray(p)[:, np.newaxis]
    
        return  1/np.sqrt(np.pi) * np.exp(-1 * (p - self.k)**2) 

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


num_steps = 10000
dt = 0.01
num_particles = 10000

simulation = Simulation(num_particles, num_steps, dt)
simulation.simulate_free_particle(1.0, plotting = True, print_time = 1000)

