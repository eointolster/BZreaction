import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Create a custom dynamic color map
colors = ["darkblue", "purple", "red", "orange", "yellow"]
nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
custom_cmap = LinearSegmentedColormap.from_list("custom", list(zip(nodes, colors)))

# Parameters
D = 0.1  # diffusion coefficient
k1 = 0.03  # reaction rate constant for u
k2 = 0.06  # reaction rate constant for v
L = 100  # length of the domain
T = 800  # total time for the simulation
dt = 0.01  # time step
frames = 2000  # total frames

# Initial conditions with added noise
u = np.random.rand(L, L)
v = np.random.rand(L, L)

# PDEs with noise added for more dynamism
def pde(u, v):
    laplacian_u = np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) + \
                  np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4 * u
    laplacian_v = np.roll(v, -1, axis=0) + np.roll(v, 1, axis=0) + \
                  np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 4 * v
    noise_u = 0.1 * np.random.randn(L, L)
    noise_v = 0.1 * np.random.randn(L, L)
    du_dt = D * laplacian_u - u * v**2 + k1 * (1 - u) + noise_u
    dv_dt = D * laplacian_v + u * v**2 - (k2 + 1) * v + noise_v
    return du_dt, dv_dt

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Initialize the plot
im = ax.imshow(u, cmap=custom_cmap, vmin=0, vmax=1)

# Define the animation update function
def update(frame):
    global u, v
    du_dt, dv_dt = pde(u, v)
    u += du_dt * dt
    v += dv_dt * dt
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    im.set_array(u)
    return [im,]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True)

# Save the animation as a high-quality video
ani.save('BZ_reaction_dynamic.mp4', writer='ffmpeg', dpi=300)

# Show the animation
plt.show()