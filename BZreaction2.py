import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
D = 0.1  # diffusion coefficient
k1 = 0.06  # reaction rate constant for u
k2 = 0.07  # reaction rate constant for v
L = 100  # length of the domain
dx = 0.02  # spatial step
T = 800  # extended total time for a longer simulation
dt = 0.05  # smaller time step for a finer detail in simulation
frames = 4000  # more frames for a slower, longer animation

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

# Set up the figure and axis for a higher resolution output
fig, ax = plt.subplots(figsize=(10, 10))

# Initialize the plot with an enhanced color map
im = ax.imshow(u, cmap='plasma', vmin=0, vmax=1)  # using plasma for a more dramatic color contrast

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
ani = animation.FuncAnimation(fig, update, frames=frames, interval=25, blit=True)

# Save the animation as a high-quality video
ani.save('BZ_reaction_enhanced.mp4', writer='ffmpeg', dpi=300)

# Show the animation
plt.show()
