import celery
import numpy as np
import pickle
from perlin_noise import perlin_noise_3d
from marching_cubes import generate_mesh

app = celery.Celery('perlin_noise_worker')
app.config_from_object('config')

@app.task
def generate_perlin_noise(data):
    # Since seed is optional, check if it was given or not
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
    # Generating mesh with marching cubes
    vertices, indices = generate_mesh(grid, threshold)
    # Adding offset to vertices, since each task won't know which area the subpspace it's working on actually comes from
    vertices = np.array(vertices)
    vertices += np.array(offset)
    return pickle.dumps((vertices, indices))
