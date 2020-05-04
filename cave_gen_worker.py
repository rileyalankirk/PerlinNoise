import celery
import numpy as np
import pickle
from perlin_noise import perlin_noise_3d
from marching_cubes import generate_mesh

app = celery.Celery('perlin_noise_worker')
app.config_from_object('config')

@app.task
def generate_perlin_noise(x, y, z, frequency, amplitude, seed=None):
    # Generates 3D Perlin noise
    return pickle.dumps(perlin_noise_3d(x, y, z, frequency, seed=seed)*amplitude)

@app.task
def build_marching_cube_mesh(grid, threshold, offset):
    # Generates 3D Perlin noise
    vertices, indices = generate_mesh(grid, threshold)
    vertices = np.array(vertices)
    indices = np.array(indices)
    vertices += np.array(offset)
    return pickle.dumps((vertices, indices))
