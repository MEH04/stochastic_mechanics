import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Simulation: 
    """Simulation wrapper, including animations and plotting."""
    def __init__(self, num_particles, steps, dt, k=1.0): 
        """
        Initialise simulation. 
        
        Parameters:
            int num_particles: Number of particles to simulate.
            int steps: Number of time steps.
            float dt: Time step size.
            float k: Wave number.
        """
        self.stochastic = StochasticMechanics(k) # stochastic mechanics object
        self.quantum = QuantumMechanics(k) # quantum mechanics object
        self.num_particles = num_particles
        self.steps = steps
        self.dt = dt
        self.dx = np.sqrt(self.dt)  # Step size from requirement dx^2/dt = 1
        self.k = k
        self.time_grid = np.arange(1E-10, dt * steps, dt)
    
    def simulate_free_particle(self, x_max = 10, p_max = 10, initial_positions = 'gaussian', plotting = False, print_time = None):
        """
        Simulate a free particle including position and momentum measures. 

        Parameters:
            float x_max: Mod of maximum position value for the simulation.
            float p_max: Mod of maximum momentum value for the simulation.
            str initial_positions: Initial position distribution. Options are 'delta' and 'gaussian'.
            bool plotting: Whether to plot the simulation.
            float print_time: Time to print the mean and standard deviation of position and momentum.
        """
        x_grid = np.arange(-x_max, x_max, self.dx) # grid to perform random walk on; to initialise initial distribution
        walk = RandomWalk(x_grid, self.num_particles, self.steps, self.dt, initial_positions=initial_positions)
        stochastic = StochasticMechanics(self.k) # stochastic mechanics object
        quantum = QuantumMechanics(self.k) # quantum mechanics object
        time_grid, positions = walk.wiener_process(stochastic.free_particle_drift) # update 
        x_grid = np.linspace(np.min(positions), np.max(positions), 1000) # new grid for computing psi_squared
        p_grid = np.linspace(-p_max, p_max, len(x_grid))
        psi_squared = quantum.free_particle_psi_squared(x_grid, time_grid) # find wave functions for plotting purposes
        momenta = positions / time_grid
        psi_squared_momentum = quantum.free_particle_momentum_psi_squared(p_grid, time_grid)

        if print_time:
            idx = np.argmin(np.abs(time_grid - print_time))
            mean_position = np.mean(positions[:, idx])
            mean_momentum = np.mean(momenta[:, idx])
            std_position = np.std(positions[:, idx])
            std_momentum = np.std(momenta[:, idx])
            print(f"At time {time_grid[idx]:.3f}: Mean Position = {mean_position:.3f}, Mean Momentum = {mean_momentum:.3f}, STD Position = {std_position:.3f}, STD Momentum = {std_momentum:.3f}")

        if plotting: 
            self.plot_trajectories(time_grid, positions, 5)
            self.plot_std(time_grid, positions, psi_squared)
            self.animate_position_momentum(x_grid, time_grid, positions, psi_squared, psi_squared_momentum)
    
    def simulate_tunneling(self, x_max, starting_pos, sigma, speed, V_frac, V_size, initial_positions = 'gaussian', plotting = False):
        """
        Simulate tunneling of a particle in a potential well using the Crank-Nicolson method.

        Parameters:
            float x_max: Maximum position value for the simulation.
            float starting_pos: Starting position of the wave packet
            float sigma: Standard deviation of the initial wave function.
            float speed: Speed of the wave packet.
            float V_frac: Fraction of the grid to apply the potential energy to.
            float V_size: Potential energy size.
            str initial_positions: Initial position distribution. Options are 'delta' and 'gaussian'.
            bool plotting: Whether to plot the simulation.
        """
        # change self.dx to 1000 or something if too slow.
        quantum = QuantumMechanics(self.k)
        x_grid = np.arange(-x_max, x_max, self.dx) # grid to perform random walk on; to initialise initial distribution
        V_x = quantum.create_potential(x_grid, V_size, V_frac)
        psi_zero = (1/np.sqrt(2*np.pi*sigma))*np.exp(-(x_grid-starting_pos)**2/(2*sigma**2))*np.exp(speed*1j*x_grid)
        psi_xt = quantum.evolve_psi_tunneling(self.steps, x_grid, V_x, psi_zero)
        stochastic = StochasticMechanics(self.time_grid, x_grid, psi_xt, self.k)
        walk = RandomWalk(x_grid, self.num_particles, self.steps, self.dt, mean = starting_pos, sigma=sigma, initial_positions=initial_positions)
        time_grid, positions = walk.wiener_process(stochastic.numerical_bf)
        if plotting:
            self.animate_position(x_grid, time_grid, positions, np.abs(psi_xt)**2)

    def animate_position_momentum(self, x_grid, time_grid, positions, psi_squared, psi_squared_momentum, num_bins=30, frame_step=10, interval=50):  
        """
        Animate the position and momentum distribytions over time in a side-by-side subplot.

        Parameters:
            x_grid: Position grid on which psi was calculated.
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
            axs[0].plot(x_grid, psi_values, color='red', linestyle='dashed', linewidth=2, label=r"$|\psi(x,t)|^2$")
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

    def animate_position(self, x_grid, time_grid, positions, psi_squared, num_bins=30, frame_step = 10, interval=50):  
        """
        Animate the distribution of particle positions over time.

        Parameters:
            array x_grid: Position grid.
            array time_grid: Array of time values.
            array positions: Array of particle positions over time.
            array psi_squared: Array of |ψ(x,t)|² values over time.
            int num_bins: Number of bins for the histogram.
            int frame_step: Time interval between frames.
            int interval: Animation speed (in milliseconds).
        """
        time_steps = self.steps
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Position", fontsize=16)
        ax.set_ylabel("Probability Density", fontsize=16)
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
            ax.plot(x_grid, psi_values, color='red', linestyle='dashed', linewidth=2, label=r"$|\psi(x,t)|^2$")
            ax.set_xlabel("Position")
            ax.set_ylabel("Probability Density")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, 1)
            ax.set_title(f"Particle Position Distribution at t = {time_grid[frame]:.3f}")
            ax.legend()
        ani = animation.FuncAnimation(fig, update, frames=range(0, time_steps, frame_step), interval=10)
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
    def __init__(self, dt = None, k=1.0):
        """
        Parameters:
            float dt: Time step size
            float k: Wave number
        """
        self.k = k
        self.dt = dt

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
    
    @staticmethod
    def create_potential(x_grid, V_size, V_frac): 
        """
        Create a potential energy grid for a potential well.

        Parameters:
            array x_grid: Position grid.
            float V_size: Potential energy size.
            float V_frac: Fraction of the grid to apply the potential energy to.

        Returns:
            array: Potential energy grid.
        """
        V_x = np.zeros(len(x_grid)).astype(complex)
        V_x[int(len(x_grid)*(0.5-V_frac/2)):int(len(x_grid)*(0.5+V_frac/2))] = V_size
        return V_x
    
    @staticmethod
    def construct_tridiagonal_matrix(a, b, c):
        """
        Constructs a tridiagonal matrix A from subdiagonal (a), main diagonal (b), and superdiagonal (c).
        
        Args:
            array a Subdiagonal (n-1 elements, a[0] is unused)
            array b Main diagonal (n elements)
            array c Superdiagonal (n-1 elements, c[-1] is unused)
        
        Returns:
            Tridiagonal matrix A of shape (n, n)
        """
        n = len(b)
        A = np.zeros((n, n), dtype=complex)

        # Fill diagonals
        np.fill_diagonal(A, b)  # Main diagonal
        np.fill_diagonal(A[1:], a[1:])  # Subdiagonal
        np.fill_diagonal(A[:, 1:], c[:-1])  # Superdiagonal
        
        return A

    @staticmethod
    def thomas_algorithm(a, b, c, d):
        """
        Solves Ax = d for a tridiagonal matrix A using the Thomas algorithm O(n).
        
        Args:
            array a Subdiagonal (n-1 elements, a[0] is unused)
            array b Main diagonal (n elements)
            array c Superdiagonal (n-1 elements, c[-1] is unused)
            array d Right-hand side vector (n elements)
        
        Returns: 
            Solution vector x (n elements)
        """
        n = len(b)
        x = np.zeros(n, dtype=complex)
        # forwards sub
        c_prime = np.zeros(n-1, dtype=complex)
        d_prime = np.zeros(n, dtype=complex)
        c_prime[0] = c[0]/b[0]
        d_prime[0] = d[0]/b[0]
        for i in range(1,n):
            d_prime[i] = (d[i]-a[i]*d_prime[i-1])/(b[i]-a[i]*c_prime[i-1])
            if i < n-1:
                c_prime[i] = c[i]/(b[i]-a[i]*c_prime[i-1])
        # backwards sub
        x[-1] = d_prime[-1]
        for i in range(n-2,-1,-1):
            x[i] = d_prime[i] - c_prime[i]*x[i+1]
        return x

    def evolve_psi_tunneling(self, steps, x_grid, V_x, psi_0): 
        """
        Simulates tunneling of a particle in a potential well using the Crank-Nicolson method.
        
        Args:
            int steps Number of time steps
            array x_grid Position grid (n elements)
            array V_x Potential energy grid (n elements)
            array psi_0 Initial wave function (n elements)
        
        Returns:
            Wave function at the final time step (n elements)
        """
        n = len(x_grid)
        psi_xt = np.zeros((len(x_grid),steps), dtype=complex)
        psi_xt[:,0] = psi_0
        alpha = 1j
        A_a = -alpha*np.ones(n)
        A_b = 2+2*alpha+1j*self.dt*V_x
        A_c = -alpha*np.ones(n)
        B_a = -A_a
        B_b = 2-2*alpha-1j*dt*V_x
        B_c = -A_c
        B = self.construct_tridiagonal_matrix(B_a,B_b,B_c)
        for i in range(1,steps):
            psi_xt[:,i] = self.thomas_algorithm(A_a,A_b,A_c,np.dot(B,psi_xt[:,i-1]))
            psi_xt[0], psi_xt[-1] = 0, 0
            integral = np.trapz(np.abs(psi_xt[:,i])**2, x_grid)
            psi_xt[:,i] = psi_xt[:,i]/np.sqrt(integral)
        return psi_xt


class StochasticMechanics:
    def __init__(self, time_grid = None, x_grid = None, psi_xt = None, k=1.0):
        """
        Parameters:
            array time_grid: Time grid
            array x_grid: Position grid
            array psi_xt: Wave function array in the spatiotemporal grid
            float k: Wave number
        """
        self.time_grid = time_grid
        self.x_grid = x_grid
        self.psi_xt = psi_xt # note the free particle does not need this
        self.k=k
    
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
    
    def numerical_bf(self, x, t):
        """
        Determine b_f by numerical derivative of psi_xt. 

        Parameters:
            float x: Position
            float t: Time
        """
        nearest_x_index = np.argmin(np.abs(self.x_grid - x))
        nearest_t_index = np.argmin(np.abs(self.time_grid - t))
        if nearest_x_index == 0 or nearest_x_index == len(self.x_grid) - 1:
            b_f = 0
        else:
            psi_x_minus_one = self.psi_xt[nearest_x_index-1, nearest_t_index]
            psi = self.psi_xt[nearest_x_index, nearest_t_index]
            psi_x_plus_one = self.psi_xt[nearest_x_index+1, nearest_t_index]
            dx = self.x_grid[1] - self.x_grid[0]
            nabla_psi = (psi_x_plus_one - psi_x_minus_one) / (2 * dx) # central difference method O(dx^2)
            b_f = np.real(nabla_psi / psi) + np.imag(nabla_psi / psi)
        return b_f


class RandomWalk:
    def __init__(self, x_grid, num_particles, steps, dt, mean = 0, sigma = 1/np.sqrt(2), k=1.0, initial_positions='delta'):
        """
        Parameters:
            array x_grid: Position grid.
            int num_particles: Number of particles to simulate.
            float steps: Number of time steps.
            float dt: time grid resolution; dx follows.
            float k: wave number
            str initial_positions: Initial position distribution. Options are 'delta' and 'gaussian'.
        """
        self.x_grid = x_grid
        self.num_particles = num_particles
        self.dt = dt
        self.steps = steps
        self.dx = np.sqrt(self.dt)  # Step size from requirement dx^2/dt = 1
        self.time_grid = np.arange(1E-10, dt * steps, dt)
        if initial_positions == 'delta':
            self.positions = np.zeros((num_particles, steps))
        elif initial_positions == 'gaussian':
            # want to confine the particles to the x_grid. 
            sampled_positions = np.random.normal(loc=mean, scale=sigma, size=num_particles)
            # Map to the closest values in the x_grid
            self.positions = np.array([x_grid[np.argmin(np.abs(x_grid - pos))] for pos in sampled_positions])
            # Expand positions to store for all time steps
            self.positions = np.tile(self.positions[:, np.newaxis], steps)

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
            prob_drift = np.clip(prob_drift, -1, 1)
            # Ensure consistency condition holds for all particles
            if np.any(np.abs(prob_drift) > 1):
                raise ValueError('Consistency condition not met for at least one particle.')
            p_left = 0.5 * (1 - prob_drift)  # Compute left probability per particle
            deviate = np.random.uniform(0, 1, num_particles)  
            # Update positions
            move_left = deviate < p_left
            self.positions[:, i] = self.positions[:, i - 1] - self.dx * move_left + self.dx * (~move_left) 
        return self.time_grid, self.positions


num_steps = 3000
dt = 0.01
num_particles = 2000
x_max = 10
sigma = 0.3
starting_point = -2
speed = 1
V_size = 1
V_frac = 0.1

simulation = Simulation(num_particles, num_steps, dt)
simulation.simulate_tunneling(x_max, starting_pos = starting_point, sigma = sigma, speed = speed, V_frac = V_frac, V_size = V_size, plotting = True)
# simulation.simulate_free_particle(x_max = 1, initial_positions='gaussian', plotting = True)

