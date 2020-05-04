import celery
import numpy as np
from perlin_noise import perlin_noise_3d

app = celery.Celery('perlin_noise_worker')
app.config_from_object('config')

@app.task
def generate_perlin_noise(x, y, z, frequency, amplitude, seed=None):
    # Generates 3D Perlin noise
    return (perlin_noise_3d(x, y, z, frequency, seed=seed)*amplitude).round().astype(np.int8).tobytes()
