"""A python program for generating 2D and 3D Perlin noise, including banded forms of both.

Author: Riley Kirkpatrick
Reference: Tomáš Bouda -- https://medium.com/100-days-of-algorithms/day-88-perlin-noise-96d23158a44c
"""


import imageio
import numpy as np


# *---------------*
#  2D Perlin Noise
# *---------------*

def generate_gradient_2d(width, height, seed=None):
    """Creates a random 2D gradient.

    Arguments:
        width {int} -- The width of the gradient.
        height {int} -- The height of the gradient.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        numpy.ndarray -- The random 2D gradient.
    """

    if seed: np.random.seed(seed)
    return np.random.rand(width, height, 2)*2 - 1

def linear_spacing_2d(width, height, frequency):
    """Generates a 2D linear spacing.

    Arguments:
        width {int} -- The width of the gradient.
        height {int} -- The height of the gradient.
        frequency {float} -- The frequency of the linear spacing.

    Returns:
        numpy.ndarray -- The 3D linear spacing.
    """
    # Linear spacing by frequency
    x = np.tile(np.linspace(0, frequency, width, endpoint=False), height)
    y = np.repeat(np.linspace(0, frequency, height, endpoint=False), width)

    # Gradient coordinates
    x0, y0 = x.astype(int), y.astype(int)

    # Local coordinates
    x -= x0; y -= y0

    return x, y, x0, y0

def generate_gradient_projections_2d(width, height, frequency, seed=None):
    """Creates a 2D gradient and calculates the projections into the gradient that are linearly spaced.

    Arguments:
        width {int} -- The width of the gradient.
        height {int} -- The height of the gradient.
        frequency {float} -- The frequency of the linear spacing.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        tuple -- Returns the gradient projections (list of numpy.ndarray) and each of the local coordinates from the
                 generated linear spacing (numpy.ndarray).
    """

    gradient = generate_gradient_2d(width, height, seed=seed)
    x, y, x0, y0 = linear_spacing_2d(width, height, frequency)

    # Gradient projections [g00, g10, g01, g11]
    gradient_projs = [gradient[x0 + i, y0 + j] for j in [0, 1] for i in [0, 1]]

    return gradient_projs, x, y

def perlin_noise_2d(width, height, frequency, seed=None):    
    """Generates 2D Perlin noise.

    Arguments:
        width {int} -- The width of the gradient.
        height {int} -- The height of the gradient.
        frequency {float} -- The frequency of the linear spacing.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        numpy.ndarray -- An array of floating point numbers.
    """

    gradient_projs, x, y = generate_gradient_projections_2d(width, height, frequency, seed=seed)
    g00, g10, g01, g11 = gradient_projs

    # X-fade
    t = (3 - 2*x) * x*x

    # Linear interpolation
    r = g00[:, 0] * x + g00[:, 1] * y
    s = g10[:, 0] * (x - 1) + g10[:, 1] * y
    g0 = r + t * (s - r)

    r = g01[:, 0] * x + g01[:, 1] * (y - 1)
    s = g11[:, 0] * (x - 1) + g11[:, 1] * (y - 1)
    g1 = r + t * (s - r)

    # Y-fade
    t = (3 - 2*y) * y*y

    # Bilinear interpolation
    g = g0 + t * (g1 - g0)

    return g.reshape(height, width)

def banded_perlin_noise_2d(width, height, frequencies, amplitudes, seed=None):
    """Generates 3D banded Perlin noise by averaging together 3D Perlin noise of different frequencies and amplitudes.

    Arguments:
        width {int} -- The width of the gradient.
        height {int} -- The height of the gradient.
        frequency {float} -- The frequency of the linear spacing.
        frequencies {iterable} -- Frequencies for linear spacing (floats).
        amplitudes {iterable} -- Amplitudes for each Perlin noise (floats).

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        numpy.ndarray -- An array of floating point numbers.
    """

    image = np.zeros((height, width))
    for f, a in zip(frequencies, amplitudes):
        image += perlin_noise_2d(width, height, f, seed=seed) * a
    image -= image.min()
    image /= image.max()
    return image





# *---------------*
#  3D Perlin Noise
# *---------------*

def generate_gradient_3d(x, y, z, seed=None):
    """Creates a random 3D gradient.

    Arguments:
        x {int} -- The size of the gradient in the x-direction.
        y {int} -- The size of the gradient in the y-direction.
        z {int} -- The size of the gradient in the z-direction.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        numpy.ndarray -- The random 3D gradient.
    """

    if seed: np.random.seed(seed)
    return np.random.rand(x, y, z, 3)*2 - 1

def linear_spacing_3d(x, y, z, frequency):
    """Generates a 3D linear spacing.

    Arguments:
        x {int} -- The size of the gradient in the x-direction.
        y {int} -- The size of the gradient in the y-direction.
        z {int} -- The size of the gradient in the z-direction.
        frequency {float} -- The frequency of the linear spacing.

    Returns:
        numpy.ndarray -- The 3D linear spacing.
    """

    # Linear spacing by frequency
    x_vals = np.tile(np.tile(np.linspace(0, frequency, x, endpoint=False), y), z)
    y_vals = np.repeat(np.repeat(np.linspace(0, frequency, y, endpoint=False), x), z)
    z_vals = np.repeat(np.tile(np.linspace(0, frequency, z, endpoint=False), y), x)

    # Gradient coordinates
    x0, y0, z0 = x_vals.astype(np.int64), y_vals.astype(np.int64), z_vals.astype(np.int64)
    x1, y1, z1 = np.ceil(x_vals), np.ceil(y_vals), np.ceil(z_vals)

    # Local coordinates
    x_vals -= x0; y_vals -= y0; z_vals -= z0

    return x_vals, y_vals, z_vals, x0, y0, z0, x1, y1, z1

def generate_gradient_projections_3d(x, y, z, frequency, seed=None):
    """Creates a 3D gradient and calculates the projections into the gradient that are linearly spaced.

    Arguments:
        x {int} -- The size of the gradient in the x-direction.
        y {int} -- The size of the gradient in the y-direction.
        z {int} -- The size of the gradient in the z-direction.
        frequency {float} -- The frequency of the linear spacing.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        tuple -- Returns the gradient projections (list of numpy.ndarray) and each of the local coordinates from the
                 generated linear spacing (numpy.ndarray).
    """

    gradient = generate_gradient_3d(x, y, z, seed=seed)
    x, y, z, x0, y0, z0, x1, y1, z1 = linear_spacing_3d(x, y, z, frequency)

    # Gradient projections [g000, g100, g010, g110, g001, g101, g011, g111]
    gradient_projs = [gradient[x0 + i, y0 + j, z0 + k] for k in [0, 1] for j in [0, 1] for i in [0, 1]]

    return gradient_projs, x, y, z

def perlin_noise_3d(x, y, z, frequency, seed=None):
    """Generates 3D Perlin noise.

    Arguments:
        x {int} -- The size of the gradient in the x-direction.
        y {int} -- The size of the gradient in the y-direction.
        z {int} -- The size of the gradient in the z-direction.
        frequency {float} -- The frequency of the linear spacing.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        numpy.ndarray -- An array of floating point numbers.
    """

    gradient_projs, x_vals, y_vals, z_vals = generate_gradient_projections_3d(x, y, z, frequency, seed=seed)
    g000, g100, g010, g110, g001, g101, g011, g111 = gradient_projs

    # X, Y, and Z fades
    tx = (3 - 2*x_vals) * x_vals*x_vals
    ty = (3 - 2*y_vals) * y_vals*y_vals
    tz = (3 - 2*z_vals) * z_vals*z_vals

    # Compute dot product of distance and gradient vectors
    g000 =     x_vals*g000[:, 0] +       y_vals*g000[:, 1] +     z_vals*g000[:, 2]
    g100 = (x_vals-1)*g100[:, 0] +       y_vals*g100[:, 1] +     z_vals*g100[:, 2]
    g010 =     x_vals*g010[:, 0] + (y_vals - 1)*g010[:, 1] +     z_vals*g010[:, 2]
    g110 = (x_vals-1)*g110[:, 0] + (y_vals - 1)*g110[:, 1] +     z_vals*g110[:, 2]
    g001 =     x_vals*g001[:, 0] +       y_vals*g001[:, 1] + (z_vals-1)*g001[:, 2]
    g101 = (x_vals-1)*g101[:, 0] +       y_vals*g101[:, 1] + (z_vals-1)*g101[:, 2]
    g011 =     x_vals*g011[:, 0] + (y_vals - 1)*g011[:, 1] + (z_vals-1)*g011[:, 2]
    g111 = (x_vals-1)*g111[:, 0] + (y_vals - 1)*g111[:, 1] + (z_vals-1)*g111[:, 2]

    # X-direction linear interpolations
    g00 = g000 + tx*(g100 - g000)
    g01 = g001 + tx*(g101 - g001)
    g10 = g010 + tx*(g110 - g010)
    g11 = g011 + tx*(g111 - g011)

    # Y-direction linear interpolations
    g0 = g00 + ty*(g10 - g00)
    g1 = g01 + ty*(g11 - g01)

    # Z-direction linear interpolation
    g = g0 + tz*(g1 - g0)

    return g.reshape(x, y, z)

def banded_perlin_noise_3d(x, y, z, frequencies, amplitudes, seed=None):
    """Generates 3D banded Perlin noise by averaging together 3D Perlin noise of different frequencies and amplitudes.

    Arguments:
        x {int} -- The size of the gradient in the x-direction.
        y {int} -- The size of the gradient in the y-direction.
        z {int} -- The size of the gradient in the z-direction.
        frequency {float} -- The frequency of the linear spacing.
        frequencies {iterable} -- Frequencies for linear spacing (floats).
        amplitudes {iterable} -- Amplitudes for each Perlin noise (floats).

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        numpy.ndarray -- An array of floating point numbers.
    """
    image = np.zeros((x, y, z))
    for f, a in zip(frequencies, amplitudes):
        image += perlin_noise_3d(x, y, z, f, seed=seed) * a
    image -= image.min()
    image /= image.max()
    return image





def main():
    # Generates three images of different layers of the banded 3D Perlin noise (layers 0, 5, and 10)

    x, y, z = 200, 200, 200
    seed, num_bands = 69420, 4
    frequencies = [2**i for i in range(1, num_bands+1)]
    amplitudes = [2**i for i in range(num_bands-1, -1, -1)]

    # image_array = (perlin_noise_3d(x, y, z, 5, seed=seed)*255).round().astype(np.uint8)
    image_array = (banded_perlin_noise_3d(x, y, z, frequencies, amplitudes, seed=seed)*255).round().astype(np.uint8)
    for i in range(3):
        imageio.imwrite('perlin_noise/' + str(seed) + '_' + str(i) + '.png', image_array[i*5], format='png')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('You interrupted the program, you jerk.')