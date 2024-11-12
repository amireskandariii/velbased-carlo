import numpy as np
import taichi as ti
import time

# Initialize Taichi
ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)

# Simulation parameters
resolution = 128
dt = 0.01
rho = 1.0

# Fields for smoke density and velocity components
smoke_density = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))
velocity_x = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))
velocity_y = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))
velocity_z = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))

# Temporary fields for storing new velocity and smoke density after advection
new_velocity_x = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))
new_velocity_y = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))
new_velocity_z = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))
new_smoke_density = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))

# Initialize smoke density in a specific area
@ti.kernel
def initialize_smoke():
    for i, j, k in smoke_density:
        if (resolution // 4 < i < 3 * resolution // 4) and (resolution // 4 < j < 3 * resolution // 4):
            smoke_density[i, j, k] = 50.0  # Set a higher initial smoke density value for visibility

@ti.func
def trilinear_interpolate(field, x, y, z):
    ix, iy, iz = int(ti.floor(x)), int(ti.floor(y)), int(ti.floor(z))
    dx, dy, dz = x - ix, y - iy, z - iz

    # Clamp indices to valid bounds for safe field access
    ix = max(0, min(ix, resolution - 2))
    iy = max(0, min(iy, resolution - 2))
    iz = max(0, min(iz, resolution - 2))

    # Interpolate along each dimension
    c00 = field[ix, iy, iz] * (1 - dx) + field[ix + 1, iy, iz] * dx
    c01 = field[ix, iy, iz + 1] * (1 - dx) + field[ix + 1, iy, iz + 1] * dx
    c10 = field[ix, iy + 1, iz] * (1 - dx) + field[ix + 1, iy + 1, iz] * dx
    c11 = field[ix, iy + 1, iz + 1] * (1 - dx) + field[ix + 1, iy + 1, iz + 1] * dx

    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy

    return c0 * (1 - dz) + c1 * dz

@ti.kernel
def apply_external_forces():
    for i, j, k in velocity_x:
        buoyancy_force = 50.0 * smoke_density[i, j, k] 
        velocity_y[i, j, k] += buoyancy_force * dt

@ti.kernel
def advect_velocity():
    for i, j, k in velocity_x:
        # Trace back the particle position using the velocity field components
        x = i - velocity_x[i, j, k] * dt
        y = j - velocity_y[i, j, k] * dt
        z = k - velocity_z[i, j, k] * dt

        # Interpolate each component separately
        new_velocity_x[i, j, k] = trilinear_interpolate(velocity_x, x, y, z)
        new_velocity_y[i, j, k] = trilinear_interpolate(velocity_y, x, y, z)
        new_velocity_z[i, j, k] = trilinear_interpolate(velocity_z, x, y, z)

    # Copy the new values back to the original fields
    for i, j, k in velocity_x:
        velocity_x[i, j, k] = new_velocity_x[i, j, k]
        velocity_y[i, j, k] = new_velocity_y[i, j, k]
        velocity_z[i, j, k] = new_velocity_z[i, j, k]

@ti.kernel
def advect_smoke():
    for i, j, k in smoke_density:
        # Calculate the backward-traced position using velocity components
        x = i - velocity_x[i, j, k] * dt
        y = j - velocity_y[i, j, k] * dt
        z = k - velocity_z[i, j, k] * dt

        # Use trilinear interpolation to calculate the new smoke density
        new_smoke_density[i, j, k] = trilinear_interpolate(smoke_density, x, y, z)

    # Copy the updated values back to the main smoke density field
    for i, j, k in smoke_density:
        smoke_density[i, j, k] = new_smoke_density[i, j, k]

# Main simulation loop
def run_simulation(time_steps):
    initialize_smoke()
    gui = ti.GUI("Smoke Simulation", (resolution, resolution))

    for step in range(time_steps):
        print(f"Step {step + 1}/{time_steps}")

        advect_velocity()
        advect_smoke()
        apply_external_forces()

        # avg_density = np.mean(smoke_density.to_numpy())
        # avg_velocity_x = np.mean(velocity_x.to_numpy())
        # avg_velocity_y = np.mean(velocity_y.to_numpy())
        # avg_velocity_z = np.mean(velocity_z.to_numpy())

        # print(f"Average smoke density: {avg_density}")
        # print(f"Average velocity_x: {avg_velocity_x}")
        # print(f"Average velocity_y: {avg_velocity_y}")
        # print(f"Average velocity_z: {avg_velocity_z}")

        img = np.mean(smoke_density.to_numpy(), axis=2) * 100  # Increase brightness for visibility
        # print("Max density in img:", np.max(img))
        gui.set_image(img)
        gui.show()
        time.sleep(0.05)  # Optional delay for easier visualization

    while gui.running:
        gui.show()

# Run the simulation with a set number of time steps
time_steps = 200
run_simulation(time_steps)
