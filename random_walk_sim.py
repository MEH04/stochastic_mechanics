import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline

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
    
    def simulate_free_particle(self, p_max = 10, initial_positions = 'gaussian', plotting = False, print_time = None):
        """
        Simulate a free particle including position and momentum measures. 

        Parameters:
            float p_max: Mod of maximum momentum value for the simulation.
            str initial_positions: Initial position distribution. Options are 'delta' and 'gaussian'.
            bool plotting: Whether to plot the simulation.
            float print_time: Time to print the mean and standard deviation of position and momentum.
        """
        walk = SDE(self.time_grid, self.num_particles, self.steps, initial_positions=initial_positions)
        stochastic = StochasticMechanics(self.k) # stochastic mechanics object
        quantum = QuantumMechanics(self.k) # quantum mechanics object
        time_grid, positions = walk.euler_maruyama(stochastic.free_particle_drift) # update 
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
            self.plot_trajectories(time_grid, positions, 15)
            self.plot_std(time_grid, positions, psi_squared)
            self.animate_position_momentum(x_grid, time_grid, positions, psi_squared, psi_squared_momentum)
    
    def simulate_tunneling(self, x_max, starting_pos, sigma, V_size, V_width, separation = None, initial_positions = 'gaussian', walk_type = 'euler_maruyama', stochastic = True, plotting = False):
        """
        Simulate tunneling of a particle in a potential well using the Crank-Nicolson method.

        Parameters:
            float x_max: Maximum position value for the simulation.
            float starting_pos: Starting position of the wave packet
            float sigma: Standard deviation of the initial wave function.
            float V_frac: Fraction of the grid to apply the potential energy to.
            float V_size: Potential energy size.
            str initial_positions: Initial position distribution. Options are 'delta' and 'gaussian'.
            bool stochastic: Whether to carry out the stochastic simulation. 
            bool plotting: Whether to plot the simulation.
        """
        quantum = QuantumMechanics(self.dt, self.k)
        x_grid = np.arange(-x_max+5, x_max+5, self.dx) # grid to perform random walk on; to initialise initial distribution
        if separation:
            V_x, edges = quantum.create_potential(x_grid, V_size, V_width, type = 'double', separation=separation)
        else: 
            V_x, edges = quantum.create_potential(x_grid, V_size, V_width, type = 'single')
        psi_zero = np.exp(-(x_grid - starting_pos) ** 2 / (2 * sigma ** 2)) * np.exp(1j * self.k * x_grid)
        psi_zero /= np.sqrt(np.trapz(np.abs(psi_zero) ** 2, x_grid))
        psi_xt = quantum.evolve_psi_tunneling(self.steps, x_grid, V_x, psi_zero)
        if stochastic:
            stochastic = StochasticMechanics(x_grid, self.time_grid, psi_xt, self.k)
            walk = SDE(self.time_grid, self.num_particles, self.steps, mean = starting_pos, sigma=sigma, initial_positions=initial_positions)
            if walk_type == 'euler_maruyama':
                time_grid, positions = walk.euler_maruyama(stochastic.numerical_bf)
            elif walk_type == 'random':
                time_grid, positions = walk.random_walk(stochastic.numerical_bf)
            if plotting:
                self.plot_trajectories(time_grid, positions, 40, x_grid, edges)
                #self.animate_position(x_grid, time_grid, positions, np.abs(psi_xt)**2, V_x, V_edges=edges)
                #self.plot_distribution(x_grid, [500,1250,2500], positions, np.abs(psi_xt)**2, V_x, edges)
            return x_grid, time_grid, positions, psi_xt
        else:
            return x_grid, self.time_grid, psi_xt

    def tunneling_separation_experiment(self, x_max, starting_pos, sigma, V_size, V_width, separation_list, num_experiments=5, stochastic=False, stochastic_interval=1):
        walk_types = ['random', 'euler_maruyama']
        colors = {'random': 'lightcoral', 'euler_maruyama': 'lightblue'}

        # Data storage (shared across walk types)
        psi_prob_list = []
        dwell_time_ratio_list = []
        presence_time_ratio_list = []
        separation_list_particle = [] 

        # Data storage (per walk type)
        particle_prob_dict = {walk: [] for walk in walk_types}
        particle_prob_err_dict = {walk: [] for walk in walk_types}
        
        particle_time_ratio_dict = {walk: [] for walk in walk_types}
        particle_time_ratio_err_dict = {walk: [] for walk in walk_types}

        x_grid = np.arange(-x_max + 5, x_max + 5, self.dx)

        for i, sep in enumerate(separation_list):
            _, edges = self.quantum.create_potential(x_grid, V_size, V_width, type='double', separation=sep)

            if i % stochastic_interval == 0 and stochastic:
                separation_list_particle.append(sep)  # Store separation for particle-based data
                
                # Store psi probability and dwell time once
                x_grid, time_grid, psi_xt = self.simulate_tunneling(
                    x_max, starting_pos, sigma, V_size, V_width, sep, plotting=False, stochastic=False
                )
                psi_prob = self.quantum.tunneling_probability(x_grid, time_grid, psi_xt, edges[-1])
                dwell_time_1 = self.quantum.dwell_time(x_grid, time_grid, psi_xt, edges[0:2])
                dwell_time_2 = self.quantum.dwell_time(x_grid, time_grid, psi_xt, edges[2:4])
                dwell_time_ratio = dwell_time_1 / dwell_time_2
                presence_time_1 = self.quantum.presence_time(x_grid, time_grid, psi_xt, edges[0:2])
                presence_time_2 = self.quantum.presence_time(x_grid, time_grid, psi_xt, edges[2:4])
                presence_time_ratio = presence_time_1 / presence_time_2
                print(dwell_time_ratio, presence_time_ratio)

                psi_prob_list.append(psi_prob)
                dwell_time_ratio_list.append(dwell_time_ratio)
                presence_time_ratio_list.append(presence_time_ratio)


                for walk in walk_types:
                    particle_probs = []
                    particle_time_ratios = []

                    for _ in range(num_experiments):
                        x_grid, time_grid, positions, _ = self.simulate_tunneling(
                            x_max, starting_pos, sigma, V_size, V_width, sep, plotting=False, stochastic=True, walk_type=walk
                        )
                        particle_prob = self.stochastic.particle_tunneling_probability(positions, edges[-1])
                        t_1 = self.stochastic.particle_tunneling_time(self.dt, positions, edges[0:2])
                        t_2 = self.stochastic.particle_tunneling_time(self.dt, positions, edges[2:4])
                        particle_time_ratio = t_1 / t_2

                        particle_probs.append(particle_prob)
                        particle_time_ratios.append(particle_time_ratio)

                    # Compute statistics
                    particle_prob_dict[walk].append(np.mean(particle_probs))
                    particle_prob_err_dict[walk].append(np.std(particle_probs) / np.sqrt(num_experiments))

                    particle_time_ratio_dict[walk].append(np.mean(particle_time_ratios))
                    particle_time_ratio_err_dict[walk].append(np.std(particle_time_ratios) / np.sqrt(num_experiments))

                    print(f"{walk} - Separation {sep}: Particle Prob = {particle_prob_dict[walk][-1]:.4f} ± {particle_prob_err_dict[walk][-1]:.4f}, "
                        f"Psi Prob = {psi_prob:.4f}, Particle time = {particle_time_ratio_dict[walk][-1]:.4f} ± {particle_time_ratio_err_dict[walk][-1]:.4f}, "
                        f"Dwell time = {dwell_time_ratio:.4f}")

            else:
                x_grid, time_grid, psi_xt = self.simulate_tunneling(
                    x_max, starting_pos, sigma, V_size, V_width, sep, plotting=False, stochastic=False
                )
                psi_prob = self.quantum.tunneling_probability(x_grid, time_grid, psi_xt, edges[-1])
                dwell_time_1 = self.quantum.dwell_time(x_grid, time_grid, psi_xt, edges[0:2])
                dwell_time_2 = self.quantum.dwell_time(x_grid, time_grid, psi_xt, edges[2:4])
                dwell_time_ratio = dwell_time_1 / dwell_time_2
                presence_time_1 = self.quantum.presence_time(x_grid, time_grid, psi_xt, edges[0:2])
                presence_time_2 = self.quantum.presence_time(x_grid, time_grid, psi_xt, edges[2:4])
                presence_time_ratio = presence_time_1 / presence_time_2
                print(dwell_time_ratio, presence_time_ratio)
                psi_prob_list.append(psi_prob)
                dwell_time_ratio_list.append(dwell_time_ratio)
                presence_time_ratio_list.append(presence_time_ratio)

        # ---- PLOT: Tunneling Probability ----
        fig, ax = plt.subplots()
        ax.plot(separation_list, psi_prob_list, color='black', label='Wave Function', lw=2.5)

        if stochastic:
            for walk in walk_types:
                if walk == 'euler_maruyama':
                    ax.plot(separation_list_particle, particle_prob_dict[walk], color=colors[walk], lw=2, label=f'Particles (Euler Maruyama)')
                else:
                    ax.plot(separation_list_particle, particle_prob_dict[walk], color=colors[walk], lw=2, label=f'Particles (Random Walk)')
                ax.fill_between(
                    separation_list_particle, 
                    np.array(particle_prob_dict[walk]) - np.array(particle_prob_err_dict[walk]),
                    np.array(particle_prob_dict[walk]) + np.array(particle_prob_err_dict[walk]),
                    color=colors[walk], alpha=0.6
                )

        ax.set_xlabel('Barrier Separation, $\\ell$ (a.u)', fontsize=16)
        ax.set_ylabel('Tunneling Probability', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(ls=':')
        ax.legend(fontsize=16)
        fig.savefig('tunneling_prob.pdf')
        plt.show()

        # ---- PLOT: time ratio ----
        fig, ax = plt.subplots()
        ax.plot(separation_list, dwell_time_ratio_list, lw=2.5, color='black', label='Dwell Time')
        ax.plot(separation_list, presence_time_ratio_list, lw=2.5, color='blue', alpha = 0.7, label='Presence Time')

        if stochastic:
            for walk in walk_types:
                if walk == 'euler_maruyama':
                    ax.plot(separation_list_particle, particle_time_ratio_dict[walk], color=colors[walk], lw=2, label=f'Particle Time (Euler Maruyama)')
                else: 
                    ax.plot(separation_list_particle, particle_time_ratio_dict[walk], color=colors[walk], lw=2, label=f'Particle Time (Random Walk)')
                ax.fill_between(
                    separation_list_particle, 
                    np.array(particle_time_ratio_dict[walk]) - np.array(particle_time_ratio_err_dict[walk]),
                    np.array(particle_time_ratio_dict[walk]) + np.array(particle_time_ratio_err_dict[walk]),
                    color=colors[walk], alpha=0.6
                )

        ax.set_xlabel('Barrier Separation, $\\ell$ (a.u)', fontsize=16)
        ax.set_ylabel('$t_1/t_2$', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(ls=':')
        ax.legend(fontsize=14)
        fig.savefig('time_ratios.pdf')
        plt.show() 
        # ---- SAVE DATA ----
        np.savetxt("psi_tunnel.txt", np.column_stack((separation_list, psi_prob_list)), 
                header="Separation  Psi_Prob", fmt="%.6f")
        np.savetxt("time_ratios_psi.txt", np.column_stack((separation_list, dwell_time_ratio_list, presence_time_ratio_list)), 
                header="Separation  Dwell_Time_Ratio  Presence_Time_Ratio", fmt="%.6f")
        if stochastic:
            np.savetxt("particle_tunnel.txt", 
                    np.column_stack((separation_list_particle, 
                                        particle_prob_dict['random'], particle_prob_err_dict['random'], 
                                        particle_prob_dict['euler_maruyama'], particle_prob_err_dict['euler_maruyama'])),
                    header="Separation  Particle_Prob(Random)  Error(Random)  Particle_Prob(Euler-Maruyama)  Error(Euler-Maruyama)", 
                    fmt="%.6f")

            np.savetxt("t_ratios.txt", 
                    np.column_stack((separation_list_particle, 
                                        particle_time_ratio_dict['random'], particle_time_ratio_err_dict['random'], 
                                        dwell_time_ratio_list, presence_time_ratio_list, 
                                        particle_time_ratio_dict['euler_maruyama'], particle_time_ratio_err_dict['euler_maruyama'], 
                                        dwell_time_ratio_list, presence_time_ratio_list)),
                    header="Separation  Particle_Time_Ratio(Random)  Error(Random)  Dwell_Time_Ratio Particle_Time_ratio  Particle_Time_Ratio(Euler-Maruyama)  Error(Euler-Maruyama)  Dwell_Time_Ratio Particle_Time_Ratio", 
                    fmt="%.6f")


    def tunneling_energy_experiment(self, x_max, starting_pos, sigma, V_size, V_width, separation, E_list, initial_positions='gaussian', stochastic=False, stochastic_interval=3):
        psi_prob_array = []
        particle_prob_array = []
        particle_std_array = []
        
        steps_original = self.steps
        k_original = self.k
        time_grid_original = self.time_grid

        x_grid = np.arange(-x_max+5, x_max+5, self.dx)
        _, edges = self.quantum.create_potential(x_grid, V_size, V_width, type='double', separation=separation)
        E_list_particle = []
        for i, E in enumerate(E_list):
            self.k = np.sqrt(2*E)
            self.steps = 4500 # will change
            self.time_grid = np.arange(1E-10, dt * self.steps, dt)
            if i % stochastic_interval == 0 and stochastic:
                E_list_particle.append(E)
                particle_probs = []
                psi_probs = []
                t_ratios = []
                for _ in range(5):  # 5 independent simulations
                    x_grid, time_grid, positions, psi_xt = self.simulate_tunneling(x_max, starting_pos, sigma, V_size, V_width, separation, 
                                                                                   initial_positions=initial_positions, plotting=False, stochastic=True)
                    psi_probs.append(self.quantum.tunneling_probability(x_grid, time_grid, psi_xt, edges[-1]))
                    particle_probs.append(self.stochastic.particle_tunneling_probability(positions, edges[-1]))
                psi_prob_mean = np.mean(psi_probs)
                particle_prob_mean = np.mean(particle_probs)
                particle_prob_std = np.std(particle_probs, ddof=1)
                # Store results
                psi_prob_array.append(psi_prob_mean)
                particle_prob_array.append(particle_prob_mean)
                particle_std_array.append(particle_prob_std)
            else:
                x_grid, time_grid, psi_xt = self.simulate_tunneling(x_max, starting_pos, sigma, V_size, V_width, separation,
                                                                    initial_positions=initial_positions, plotting=False, stochastic=False)
                psi_prob = self.quantum.tunneling_probability(x_grid, time_grid, psi_xt, edges[-1])
                psi_prob_array.append(psi_prob)
                print(psi_prob)
        # ---- PLOT: Tunneling Probability with Error Bars ----
        fig, ax = plt.subplots()
        ax.plot(E_list, psi_prob_array, color='black', label='Wave Function', lw=2.5)
        
        if stochastic:
            ax.errorbar(E_list_particle, particle_prob_array, yerr=particle_std_array, fmt='o', ms=3, color='green', label='Particles', capsize=3)

        ax.set_xlabel('Barrier Separation, $\\ell$ (a.u)', fontsize=16)
        ax.set_ylabel('Tunneling Probability', fontsize=16)
        ax.grid()
        ax.legend(fontsize=14)
        fig.savefig('tunneling_prob.pdf')
        plt.show()

        # ---- SAVE DATA ----
        data_psi = np.column_stack((E_list, psi_prob_array))
        data_particle = np.column_stack((E_list, particle_prob_array, particle_std_array))

        np.savetxt("psi_energy.txt", data_psi, fmt="%.6f")
        np.savetxt("particle_energy.txt", data_particle, fmt="%.6f")

        self.steps = steps_original
        self.k = k_original
        self.time_grid = time_grid_original

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

    def animate_position(self, x_grid, time_grid, positions, psi_squared, V_x = None, V_edges = None, num_bins=70, frame_step = 10, interval=50):  
        """
        Animate the distribution of particle positions over time.

        Parameters:
            array x_grid: Position grid.
            array time_grid: Array of time values.
            array positions: Array of particle positions over time.
            array psi_squared: Array of |ψ(x,t)|² values over time.
            array V_x: potential energy grid (obeying x_grid).
            array V_edges: x positions of V_x edges.
            int num_bins: Number of bins for the histogram.
            int frame_step: Time interval between frames.
            int interval: Animation speed (in milliseconds).
        """
        time_steps = self.steps
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Position", fontsize=18)
        ax.set_ylabel("Probability Density", fontsize=18)
        ax.set_xlim(-15, 20)
        # Function to update the animation frame
        def update(frame):
            ax.clear()  # Clear previous frame
            positions_at_time = positions[:, frame]  # Get particle positions at this time step

            # Dynamically scale the number of bins based on the spread of positions
            x_min, x_max = np.min(positions_at_time), np.max(positions_at_time)
            spread = x_max - x_min
            dynamic_bins = max(15, int(spread / 0.2))  # Adjust 0.1 to control bin scaling

            # Plot histogram
            hist_data, bin_edges, _ = ax.hist(
                positions_at_time, bins=dynamic_bins, density=True, 
                color='blue', edgecolor='black', alpha=0.6, label="Particle Distribution"
            )

            # Plot psi-squared
            psi_values = psi_squared[:, frame]
            ax.plot(x_grid, psi_values, color='black', linewidth=2.5, label=r"$|\psi(x,t)|^2$")

            # Draw potential barriers if provided
            if V_x is not None:
                for edge in V_edges:
                    ax.axvline(edge, lw=2, color='black')
                for i in range(0, len(V_edges), 2):  # Step by 2 to get pairs of edges
                    ax.axvspan(V_edges[i], V_edges[i+1], color='lightgray', alpha=0.5)

            ax.set_xlabel("$x$ (a.u)", fontsize = 18)
            ax.set_ylabel("Probability Density", fontsize = 18)
            ax.set_title(f"t = {time_grid[frame]:.3f}", fontsize = 18)
            ax.set_ylim(0, 0.65)
            ax.tick_params(axis = 'both', labelsize = 14)
            ax.legend(fontsize = 18)

        ani = animation.FuncAnimation(fig, update, frames=range(0, time_steps, frame_step), interval=interval)
        ani.save("tunnel_animation.gif", writer=animation.PillowWriter(fps=20), dpi=300)
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

    def plot_trajectories(self, time_grid, positions, number, x_grid=None, V_edges=None, index=2000):
        """
        Plot trajectories of randomly selected particles over time with a side-by-side plot.

        Parameters:
            array time_grid: Time grid.
            array positions: Array of particle positions over time.
            int number: Number of trajectories to plot.
            array x_grid: Position grid (for plotting the potential energy).
            array V_edges: Potential energy edge.
            int index: Time index to display particle distribution.
        """
        fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(30, 10), gridspec_kw={'width_ratios': [3, 1]})
        randoms = np.random.randint(0, self.num_particles, number)

        # Plot potential barriers
        if V_edges is not None:
            for edge in V_edges:
                if edge == V_edges[-1]:
                    ax.axhline(edge, color='gray', label='Barrier', linewidth=2)
                else:
                    ax.axhline(edge, color='gray', linewidth=2)
            for i in range(0, len(V_edges), 2):
                ax.axhspan(V_edges[i], V_edges[i + 1], color='lightgray', alpha=0.5)

        # Plot trajectories
        for i in randoms:
            ax.plot(time_grid, positions[i], color='black', lw=1)

        ax.set_xlabel('$t$ (a.u)', fontsize=50)
        ax.set_ylabel('$x$ (a.u)', fontsize=50)
        ax.legend(fontsize=30)
        ax.tick_params(axis='both', labelsize=35)
        ax.set_ylim(-8, 8)

        # Plot distribution in the side subplot with light gray background
        if index is not None and 0 <= index < len(time_grid):
            positions_at_time = positions[:, index]
            ax_hist.hist(positions_at_time, bins=30, density = True, color='blue', alpha=0.7)
            ax_hist.set_xlabel('$x$ (a.u)', fontsize=50)
            ax_hist.set_ylabel('PDF', fontsize=50)
            ax_hist.set_title(f'Final Particle Distribution', fontsize=50)
            ax_hist.tick_params(axis='both', labelsize=35)

        fig.tight_layout()
        plt.subplots_adjust(left=0.10, right=0.95, top=0.9, bottom=0.2, wspace=0.3)
        fig.savefig('Trajectories_with_SidePlot.pdf')
        plt.show()


    def plot_distribution(self, x_grid, time_index, positions, psi_squared, V_x=None, V_edges=None):
        """
        Plot the distribution of particle positions at specific time steps in separate subplots.
        
        Parameters:
            array time_grid: time grid
            array positions: particle positions
            array time_index: Indexes of the time steps to plot.
            int num_bins: Number of bins for the histogram.
        """
        fig, axes = plt.subplots(len(time_index), 1, figsize=(8, 1.8 * len(time_index)), sharex=True)
        colour = 'blue'  # Use the same color for all histograms
        bin_nums = [100, 150, 200]
        
        for i, index in enumerate(time_index): 
            ax = axes[i] if len(time_index) > 1 else axes  # Handle case when there's only one subplot
            ax.set_xlim(-10, 10)
            ax.set_ylim(0, 0.6)
            
            positions_at_time = positions[:, index]  # Get particle positions at this time step
            
            # Plot histogram
            ax.hist(positions_at_time, bins=bin_nums[i], density=True, color=colour, alpha=0.5, 
                    edgecolor='black', label = 'Particle Distribution')
            
            # Plot psi-squared
            psi_values = psi_squared[:, index]
            ax.plot(x_grid, psi_values, color='black', linewidth=2.5, label=r"$|\psi(x,t)|^2$")
            
            # Add potential barriers if provided
            if V_x is not None and V_edges is not None:
                for edge in V_edges:
                    ax.axvline(edge, lw=2, color='black')
                for j in range(0, len(V_edges), 2):  # Step by 2 to get pairs of edges
                    ax.axvspan(V_edges[j], V_edges[j+1], color='lightgray', alpha=0.5)  
            
            ax.set_yticks([0, 0.6])  # Only 0 and 0.6 on y-axis
            ax.set_yticklabels(["0", "0.6"], fontsize=12)
            
            if i == len(time_index) // 2:  # Middle subplot gets y-label
                ax.set_ylabel("Probability Density", fontsize=18)
            
            if i == 0:  # Only the top plot gets a legend
                ax.legend(fontsize=16)
            
            # Annotate time on the plot
            ax.annotate(f't = {round(index * self.dt, 3)}', xy=(0.05, 0.6), xycoords='axes fraction',
                        fontsize=16, color='black', weight='bold')
            
            ax.tick_params(axis='x', labelsize=12)
            ax.grid(ls=':')
        
        axes[-1].set_xlabel("$x$ (a.u)", fontsize=18)  # Set x-label on the last subplot
        plt.tight_layout()
        fig.savefig('Tunneling_dist.pdf')
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
        self.dt = dt
        self.k = k

    def mean_psi_squared(self, x_grid, psi_xt):
        """
        Compute the mean position expectation value <x> over time.

        Parameters:
            array x_grid: Spatial grid points.
            array psi_xt: Wave function array with positions along axis 0 and time along axis 1.

        Returns:
            array: Expectation value <x> over time.
        """
        # Compute probability density
        prob_density = np.abs(psi_xt) ** 2
        
        # Compute expectation value <x> over time
        mean_x = np.trapz(x_grid[:, np.newaxis] * prob_density, x_grid, axis=0)
        
        return mean_x

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
    def tunneling_probability(x_grid, time_grid, psi_xt, x_max): 
        """
        Compute the tunneling probability for a given wave function.

        Parameters:
            array x_grid: Position grid.
            array psi_x: Wave function.
            float x_max: Maximum position of the potential. 

        Returns:
            float: Tunneling probability.
        """
        idx = np.argmin(np.abs(x_grid-x_max))+5 # make sure you're not exactly on the barrier edge
        dx = x_grid[1] - x_grid[0]  # Assume uniform grid
        dpsi_dx = (psi_xt[idx+1, :] - psi_xt[idx-1, :]) / (2 * dx)
        # Compute probability current J(x0, t)
        J = np.imag(np.conj(psi_xt[idx, :]) * dpsi_dx)
        # Integrate J over time to get total flux
        F = np.trapz(J, time_grid)  # Numerically integrate J over time
        return F 

    @staticmethod
    def dwell_time(x_grid, time_grid, psi_xt, edges):
        # Find indices for spatial slicing
        i1 = np.searchsorted(x_grid, edges[0], side='left')
        i2 = np.searchsorted(x_grid, edges[1], side='right')
        
        # Slice the relevant x values and psi data
        x_slice = x_grid[i1:i2]
        psi_slice = psi_xt[i1:i2,:]
        
        # Compute squared magnitude
        psi_squared = np.abs(psi_slice)**2
        
        # Integrate over x for each time step (axis=0 is position)
        integral_over_x = np.trapz(psi_squared, x=x_slice, axis=0)
        
        # Integrate over time
        tau_d = np.trapz(integral_over_x, x=time_grid)
        
        return tau_d 

    @staticmethod
    def presence_time(x_grid, time_grid, psi_xt, edges): 
        i1 = np.searchsorted(x_grid, edges[0], side='left')
        i2 = np.searchsorted(x_grid, edges[1], side='right')

        # Compute the probability density as a function of time for x1
        prob_density_t_x1 = np.abs(psi_xt[i1, :])**2
        rho_t_x1 = prob_density_t_x1 / np.trapz(prob_density_t_x1, time_grid)
        presence_time_x1 = np.trapz(time_grid * rho_t_x1, time_grid)

        prob_density_t_x2 = np.abs(psi_xt[i2, :])**2
        rho_t_x2 = prob_density_t_x2 / np.trapz(prob_density_t_x2, time_grid)
        presence_time_x2 = np.trapz(time_grid * rho_t_x2, time_grid)


        # Calculate the presence time difference
        tau_p = presence_time_x2 - presence_time_x1

        return tau_p



    @staticmethod
    def create_potential(x_grid, V_size, V_width, type='single', separation=None, 
                        use_CAP=True, W0=20, L_CAP=5):
        """
        Create a potential energy grid, optionally with a Complex Absorbing Potential (CAP).

        Parameters:
            x_grid : array
                Position grid.
            V_size : float
                Potential energy size.
            V_width : float
                Potential well width.
            type : str, optional
                Type of potential well. Options: 'single', 'double'. Default is 'single'.
            separation : float, optional
                Separation between two potential wells (for double well only).
            use_CAP : bool, optional
                Whether to include a Complex Absorbing Potential (CAP). Default is True.
            W0 : float, optional
                Strength of the CAP. Default is 20.
            L_CAP : float, optional
                Width of the CAP region. Default is 5.
            alpha : float, optional
                Controls the smoothness of the Fermi CAP. Default is 2.

        Returns:
            V_x : array
                Potential energy grid (real + imaginary).
            edges : array
                Edges of the potential region.
        """
        def snap_to_grid(value, x0, dx):
            return np.round((value - x0) / dx) * dx + x0

        # Initialize potential with zeros (complex type)
        V_x = np.zeros(len(x_grid), dtype=complex)
        dx = x_grid[1] - x_grid[0]  # assume uniform spacing
        x0 = x_grid[0]  # starting point of grid

        # Define potential well(s) with snapped edges
        if type == 'single':
            # Theoretical edges for single well
            left_edge_theory  = -V_width / 2
            right_edge_theory =  V_width / 2

            # Snap the edges to grid points
            left_edge  = snap_to_grid(left_edge_theory, x0, dx)
            right_edge = snap_to_grid(right_edge_theory, x0, dx)

            # Set potential: include points between (and including) the snapped edges
            V_x[(x_grid >= left_edge) & (x_grid <= right_edge)] = V_size
            edges = [left_edge, right_edge]

        elif type == 'double':
            # The theoretical edges for the barriers:
            left_edge1_theory  = -(separation / 2 + V_width)
            right_edge1_theory = -separation / 2
            left_edge2_theory  = separation / 2
            right_edge2_theory = separation / 2 + V_width

            # Snap edges to grid points:
            left_edge1  = snap_to_grid(left_edge1_theory, x0, dx)
            right_edge1 = snap_to_grid(right_edge1_theory, x0, dx)
            left_edge2  = snap_to_grid(left_edge2_theory, x0, dx)
            right_edge2 = snap_to_grid(right_edge2_theory, x0, dx)

            # Set potentials for each barrier
            V_x[(x_grid >= left_edge1) & (x_grid <= right_edge1)] = V_size
            V_x[(x_grid >= left_edge2) & (x_grid <= right_edge2)] = V_size

            edges = [left_edge1, right_edge1, left_edge2, right_edge2]
        if use_CAP:
            x_min, x_max = x_grid[0], x_grid[-1]  # Domain limits
            x_cap_min, x_cap_max = x_min + L_CAP, x_max - L_CAP  # Start of CAP regions

            for i, x in enumerate(x_grid):
                if x < x_cap_min:
                    V_x[i] -= 1j * W0 * np.sin(np.pi * (x_cap_min - x) / L_CAP)**2
                elif x > x_cap_max:
                    V_x[i] -= 1j * W0 * np.sin(np.pi * (x - x_cap_max) / L_CAP)**2

        return V_x, edges

    
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
        psi_xt = np.zeros((n,steps), dtype=complex)
        psi_xt[:,0] = psi_0
        alpha = 1j/2
        A_a = -alpha*np.ones(n)
        A_b = 2+2*alpha+1j*self.dt*V_x
        A_c = -alpha*np.ones(n)
        B_a = -A_a
        B_b = 2-2*alpha-1j*self.dt*V_x
        B_c = -A_c
        B = self.construct_tridiagonal_matrix(B_a,B_b,B_c)
        for i in range(1,steps):
            psi_xt[:,i] = self.thomas_algorithm(A_a,A_b,A_c,np.dot(B,psi_xt[:,i-1]))
            psi_xt[0], psi_xt[-1] = 0, 0
        return psi_xt


class StochasticMechanics:
    def __init__(self, x_grid = None, time_grid = None, psi_xt = None, k=1.0):
        """
        Parameters:
            array time_grid: Time grid
            array x_grid: Position grid
            array psi_xt: Wave function array in the spatiotemporal grid
            float k: Wave number
        """
        self.x_grid = x_grid
        self.time_grid = time_grid
        self.psi_xt = psi_xt # note the free particle does not need this
        # Create a Cubic Spline for each time slice
        if x_grid is not None and time_grid is not None and psi_xt is not None:
            self.psi_splines = {}
            for t_idx in range(len(time_grid)):
                # Create a cubic spline for each time slice
                self.psi_splines[t_idx] = CubicSpline(x_grid, psi_xt[:, t_idx])

    
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
    
    def numerical_bf(self, x_array, t):
        """
        Determine b_f by numerical derivative of psi_xt, using spline interpolation.

        Parameters:
            np.array x_array: Array of positions
            float t: Time
        """
        # Find the nearest time index for t
        nearest_t_index = np.argmin(np.abs(self.time_grid - t))
        
        # Interpolate psi_xt at x_array and its neighbors for derivative calculation
        psi_values = self.psi_splines[nearest_t_index](x_array)  # Interpolated psi(x,t)
        
        # Calculate the derivative of psi using central differences (via spline interpolation)
        psi_prime = self.psi_splines[nearest_t_index].derivative()(x_array)
        
        # Compute the b_f based on the derivative
        nabla_psi = psi_prime
        b_f_array = np.real(nabla_psi / psi_values) + np.imag(nabla_psi / psi_values)

        return b_f_array
    
    def particle_tunneling_probability(self, positions, x_max):
        """
        Computes the proportion of time that particles spend beyond x_max.
        
        Parameters:
            dt : float
                Time step size.
            positions : array
                Particle positions over time (shape: num_particles x num_steps).
            x_max : float
                Final barrier location.

        Returns:
            float: Normalized probability of tunneling.
        """
        N_avg = 10
        proportion_exceeding = 0
        for i in range(N_avg):
            final_positions = positions[:,-(i+1)]
            particles_exceed = final_positions > x_max
            proportion_exceeding += np.mean(particles_exceed)
        proportion_exceeding /= 10
        return proportion_exceeding

    @staticmethod
    def particle_tunneling_time(dt, positions, V_edges):
        """
        Measures how long each particle is inside each barrier (for double barrier).
    
        Parameters:
            float dt: Time step size.
            array positions: Array of particle positions over time (shape: num_particles x num_steps).
            array V_edges: barrier edges
    
        Returns:
            float time: mean time particle spends inside the non-zero region of the potential.
        """ 
        mask = (positions > V_edges[0]) & (positions < V_edges[1])
        time = np.mean(np.sum(mask, axis = 1)*dt)
        return time



class SDE:
    def __init__(self, time_grid, num_particles, steps, mean = 0, sigma = 1/np.sqrt(2), k=1.0, initial_positions='delta'):
        """
        Parameters:
            array time_grid: Time grid
            int num_particles: Number of particles to simulate.
            float steps: Number of time steps.
            float dt: time grid resolution; dx follows.
            float k: wave number
            str initial_positions: Initial position distribution. Options are 'delta' and 'gaussian'.
        """
        self.time_grid = time_grid
        self.dt = time_grid[1] - time_grid[0]
        self.num_particles = num_particles
        self.steps = steps
        self.dx = np.sqrt(self.dt)  # Step size from requirement dx^2/dt = 1
        if initial_positions == 'delta':
            self.positions = np.zeros((num_particles, steps))
        elif initial_positions == 'gaussian':
            self.positions = np.random.normal(loc=mean, scale=sigma, size=(num_particles,steps))


    def random_walk(self, drift_velocity):
        """
        Simulates a Wiener process with drift for multiple particles.

        Parameters:
            callable drift_velocity (Callable[[float, float], float]): Function returning drift velocity given (x, t).

        Returns:
            tuple: (time_grid, positions) where positions is a 2D array of shape (num_particles, steps).
        """
        for i in range(1, self.steps):  # Start from step 1 since step 0 is initialized to 0
            t = self.time_grid[i-1]
            # Compute drift velocity for each particle
            drift = drift_velocity(self.positions[:, i - 1], t)
            prob_drift = drift * (self.dt / self.dx)
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

    def euler_maruyama(self, drift_velocity): 
        for i in range(1, self.steps):
            t = self.time_grid[i-1]
            dW = np.random.normal(0, np.sqrt(self.dt), size = self.num_particles)
            drift = drift_velocity(self.positions[:, i - 1], t)
            self.positions[:, i] = self.positions[:, i - 1] + drift * self.dt + dW
        return self.time_grid, self.positions


# previously 3000 w/ 1E-3
dt = 1E-3
num_particles = 1000
x_max = 20
speed = 5
V_size = 10
V_width = 30*np.sqrt(dt)
separation = 1.5
starting_point = -4
sigma = 0.5
separation_list = np.arange(0.25,3,np.sqrt(dt))
num_steps = 3000
# defaults: steps = 3000 dt = 1E-3 x_max = 20 sigma = 0.5 starting point = -4 speed = 4 v_size = 5 v_width = 6dt^{1/2}, separation = linspace(0.25,3,sqrt(dt)) 

simulation = Simulation(num_particles, num_steps, dt, speed)
simulation.simulate_tunneling(x_max, starting_pos = starting_point, sigma = sigma, V_width = V_width, V_size = V_size, separation=separation, initial_positions='gaussian',plotting = True)
#simulation.tunneling_dt_experiment(x_max, starting_pos = starting_point, sigma = sigma, V_frac = V_frac, V_size = V_size, N_dt=5, plotting = True)
#simulation.tunneling_separation_experiment(x_max, starting_point, sigma, V_size, V_width, separation_list, stochastic = False)
#simulation.tunneling_energy_experiment(x_max,starting_point,sigma,V_size,V_width,separation,E_list)