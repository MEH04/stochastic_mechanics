import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# hello 

def discrete_random_walk(N, steps, drift=0):
    """Simulate N particles undergoing a random walk with given drift.
    
    Args:

        N: number of particles
        steps: number ofsteps
        drift: probability shift to the right

    Returns:
        positions(array): positions of each particle at each time
        sigma(array): standard deviation of particle positions
    """
    p_right = 0.5 + drift  # Probability of going right
    p_left = 1 - p_right
    positions = np.zeros((steps + 1, N))  # Store positions at each step
    sigma = np.zeros(steps+1)
    for t in range(steps):
        rand_vals = np.random.uniform(0, 1, N)
        step_changes = np.where(rand_vals < p_left, -1, 1)
        positions[t + 1] = positions[t] + step_changes
        sigma[t + 1] = np.std(positions[t+1])
    return positions, sigma

def wiener_process(steps, dt, drift_velocity, num_particles):
    """
    Simulates a Wiener process with drift for multiple particles.

    Parameters:
        steps (int): Number of time steps.
        dt (float): Time step size.
        drift_velocity (Callable[[float, float], float]): Function returning drift velocity given (x, t).
        num_particles (int): Number of particles to simulate.

    Returns:
        tuple: (time_grid, positions) where positions is a 2D array of shape (num_particles, steps).
    """
    time_grid = np.arange(0, dt * steps, dt)  # Time array
    dx = np.sqrt(dt)  # Step size from requirement dx^2/dt = 1
    positions = np.zeros((num_particles, steps))  # Each row is a particle, each column a time step

    for i in range(1, steps):  # Start from step 1 since step 0 is initialized to 0
        t = time_grid[i]
        # Compute drift velocity for each particle
        prob_drift = np.array([drift_velocity(x, t) for x in positions[:, i - 1]]) * (dt / dx)
        # Ensure consistency condition holds for all particles
        if np.any(np.abs(prob_drift) >= 1):
            raise Exception('Consistency condition not met for at least one particle.')
        p_left = 0.5 * (1 - prob_drift)  # Compute left probability per particle
        deviate = np.random.uniform(0, 1, num_particles)  # Generate random numbers for each particle
        # Update positions
        move_left = deviate < p_left
        positions[:, i] = positions[:, i - 1] - dx * move_left + dx * (~move_left) 

    return time_grid, positions

#%% Wiener plotting
def drift_velocity(x,t):
    return -5


num_steps = 5000
dt = 0.001
num_particles = 3000
t, pos = wiener_process(num_steps,dt,drift_velocity,num_particles)
for i in range(3):
    plt.plot(t,pos[i])
plt.show()
z

time_index = 1000
positions_at_time = pos[:, time_index]
num_bins = 50
hist, bin_edges = np.histogram(positions_at_time, bins=num_bins, density=True)
plt.figure(figsize=(8, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7,color='black')
plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.title(f"Particle Position Distribution at t = {t[time_index]:.3f}")
plt.show()

#%% wiener animate
# Parameters
num_bins = 30  # Number of bins for histogram
interval = 50   # Animation speed (in milliseconds)
time_steps = pos.shape[1]  # Number of time steps in the simulation

# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlabel("Position")
ax.set_ylabel("Probability Density")
ax.set_title("Particle Position Distribution Over Time")
ax.set_xlim(-2,2)
ax.set_ylim(0,1)
# Initialize histogram (empty)
hist_data, bin_edges, patches = ax.hist([], bins=num_bins, density=True, alpha=0.7, color='blue', edgecolor='black')

# Function to update the histogram for each frame
def update(frame):
    ax.clear()  # Clear the previous frame
    positions_at_time = pos[:, frame]  # Get particle positions at this time step
    hist_data, bin_edges, _ = ax.hist(positions_at_time, bins=num_bins, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Update labels
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Particle Position Distribution at t = {t[frame]:.3f}")

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(0, time_steps, 10), interval=interval)

# Display the animation
plt.show()