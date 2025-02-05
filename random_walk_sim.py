import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def random_walk(N, steps, drift=0):
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


# Parameters
N = 5000
steps = 1000
drift = 0.0  # Bias to the right
positions_over_time, sigma = random_walk(N, steps, drift)

# plot sigma
t = np.arange(0,steps+1,1)
plt.plot(t,sigma,label='Simulated',color='black')
plt.plot(t,np.sqrt(t),label='$\sqrt{t}$',alpha=0.6,color='red')
plt.xlabel('$t$',fontsize=16)
plt.ylabel('$\sigma$',fontsize=16)
plt.legend()
plt.show()

# plot some particle trajectories
particle_names = ['partrick', 'wanseo', 'penelope', 'isaac', 'albert', 'tommy', 'deborah', 'fanny', 'plutarch', 'faye']
for i in range(10):
    plt.plot(t, positions_over_time[:,i],label=particle_names[i])
plt.legend()
plt.xlabel('$t$',fontsize=16)
plt.ylabel('Position',fontsize=16)
plt.axhline(0,ls=':',color='red',lw=5)
plt.show()

animate = False
# animate
if animate: 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Position")
    ax.set_ylabel("Count")
    ax.set_title("Evolution of Random Walk Distribution")
    ax.set_xlim(-steps, steps)
    ax.set_ylim(0, N )  # Adjust the height for better visualization

    histogram, = ax.plot([], [], 'o-', color='blue')

    def update(frame):
        ax.clear()
        ax.set_xlim(-steps, steps)
        ax.set_ylim(0, N / 10)
        ax.set_xlabel("Position")
        ax.set_ylabel("Count")
        ax.set_title(f"Evolution of Random Walk Distribution (Step {frame})")

        # Bin edges centered around 0
        bins = np.arange(min(positions_over_time[frame]) - 1, 
                        max(positions_over_time[frame]) + 2, 2) + 1

        counts, bin_edges = np.histogram(positions_over_time[frame], bins=bins)
        ax.bar(bin_edges[:-1], counts, width=1.8, edgecolor='black', alpha=0.7)

    ani = animation.FuncAnimation(fig, update, frames=steps + 1, interval=10, blit = False)

    plt.show()
