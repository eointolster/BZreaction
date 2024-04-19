import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
D = 0.1  # diffusion coefficient
k1 = 0.06  # reaction rate constant for u
k2 = 0.07  # reaction rate constant for v
L = 100  # length of the domain
dx = 0.02  # spatial step
T = 400  # total time increased for longer animation
dt = 0.1  # reduced time step for more detailed simulation
frames = 2000  # more frames for a longer, smoother animation

# Initial conditions
u = np.random.rand(L, L)  # initial concentration of reactant 1
v = np.random.rand(L, L)  # initial concentration of reactant 2

# PDEs
def pde(u, v):
    laplacian_u = np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) + \
                  np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4 * u
    laplacian_v = np.roll(v, -1, axis=0) + np.roll(v, 1, axis=0) + \
                  np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 4 * v
    du_dt = D * laplacian_u - u * v**2 + k1 * (1 - u)
    dv_dt = D * laplacian_v + u * v**2 - (k2 + 1) * v
    return du_dt, dv_dt

# Set up the figure and axis
fig, ax = plt.subplots()

# Initialize the plot
im = ax.imshow(u, cmap='inferno', vmin=0, vmax=1)  # using inferno for better color visibility

# Define the animation update function
def update(frame):
    global u, v
    du_dt, dv_dt = pde(u, v)
    u += du_dt * dt
    v += dv_dt * dt
    u = np.clip(u, 0, 1)  # prevent values from growing too large
    v = np.clip(v, 0, 1)  # prevent values from growing too large
    im.set_array(u)
    return [im,]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# Save the animation if needed
ani.save('BZ_reaction.mp4', writer='ffmpeg')

# Show the animation
plt.show()
