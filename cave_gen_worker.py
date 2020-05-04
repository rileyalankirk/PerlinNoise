import celery
import numpy as np
import pickle
from perlin_noise import perlin_noise_3d
from marching_cubes import generate_mesh

app = celery.Celery('perlin_noise_worker')
app.config_from_object('config')

@app.task
def generate_perlin_noise(data):
    seed = None
    if len(data) >= 6:
        seed = data[5]
    # Generates 3D Perlin noise
    return pickle.dumps(perlin_noise_3d(data[0], data[1], data[2], data[3], seed=seed)*data[4])

@app.task
def build_marching_cube_mesh(data):
    grid = data[0]
    threshold = data[1]
    offset = data[2]
    # Generates 3D Perlin noise
    vertices, indices = generate_mesh(grid, threshold)
    vertices = np.array(vertices)
    indices = np.array(indices)
    vertices += np.array(offset)
    return pickle.dumps((vertices, indices))
