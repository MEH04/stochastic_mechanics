import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def evolve_psi_tunneling(steps, dt, dx, x, V_x, psi_0): 
    """
    Simulates tunneling of a particle in a potential well using the Crank-Nicolson method.
    
    Args:
        array psi_0 Initial wave function (n elements)
        int steps Number of time steps
        float dt Time step size
        float dx Position step size
        array x Position grid (n elements)
        array V_x Potential energy grid (n elements)
        array psi_0 Initial wave function (n elements)
    
    Returns:
        Wave function at the final time step (n elements)
    """
    n = len(x)
    psi_xt = np.zeros((len(x),steps), dtype=complex)
    psi_xt[:,0] = psi_0
    alpha = 1j
    A_a = -alpha*np.ones(n)
    A_b = 2+2*alpha+1j*dt*V_x
    A_c = -alpha*np.ones(n)
    B_a = -A_a
    B_b = 2-2*alpha-1j*dt*V_x
    B_c = -A_c
    B = construct_tridiagonal_matrix(B_a,B_b,B_c)
    for i in range(1,steps):
        psi_xt[:,i] = thomas_algorithm(A_a,A_b,A_c,np.dot(B,psi_xt[:,i-1]))
        psi_xt[0], psi_xt[-1] = 0, 0
    return psi_xt

def animate_tunneling(psi_xt, x, V_frac):
    fig, ax = plt.subplots()
    n=len(x)
    ax.set_xlim(x[0], x[-1])
    ax.axvline(x[int(n*(0.5-V_frac/2))])
    ax.axvline(x[int(n*(0.5+V_frac/2))])
    line, = ax.plot([], [], lw=2)  # Initialize empty line

    def update(frame):
        line.set_data(x, np.abs(psi_xt[:, frame])**2)  # Update data
        ax.set_ylim(0,1)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=psi_xt.shape[1], interval=1, blit=True)
    plt.xlabel('Position')
    plt.ylabel('Probability Density')
    plt.title('Quantum Tunneling Animation')
    plt.show()

# Example usage
steps = 6000
dt = 1E-4
dx = 0.01
x = np.arange(-10,10,dx)
n=len(x)
sigma = 0.3
starting_point = -3
V_size = 3
V_frac = 0.1
psi_0 = (1/np.sqrt(2*np.pi*sigma))*np.exp(-(x-starting_point)**2/(2*sigma**2))*np.exp(3j*1*x)
psi_0 /= np.sqrt(np.trapz(np.abs(psi_0)**2, x))
V_x = np.zeros(n).astype(complex)
V_x[int(n*(0.5-V_frac/2)):int(n*(0.5+V_frac/2))] = V_size
V_x[0], V_x[-1] = 10000, 10000

psi_xt = evolve_psi_tunneling(steps, dt, dx, x, V_x, psi_0)
animate_tunneling(psi_xt,x,V_frac)
