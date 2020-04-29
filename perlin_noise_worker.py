import celery
import numpy as np
from perlin_noise import perlin_noise_3d

app = celery.Celery('perlin_noise_worker')
app.config_from_object('config')

@app.task
def generate_perlin_noise(n):
    # Generates 3D Perlin noise

    x, y, z = 200, 200, 200
    seed = 69420
    frequency = 2

    return (perlin_noise_3d(x, y, z, frequency, seed=seed)*255).round().astype(np.int8)
