"""A python program for generating ND Perlin noise.

Author: Riley Kirkpatrick
Reference: Tomáš Bouda -- https://medium.com/100-days-of-algorithms/day-88-perlin-noise-96d23158a44c
"""


import numpy as np


# *----------------------*
#  Helpers and Exceptions
# *----------------------*

class ShapeException(Exception):
    pass

def permute(arr1, arr2):
    # Permutes lists of the form [a1, a2,...], [b1, b2,...] to [[c1, c2,...], [d1, d2,...],...]
    # Thus, the elements maintain order, but we get all permutations of elements from both lists
    if len(arr1) != len(arr2):
        return None

    out1 = [[arr1[0]] for _ in range(2**(len(arr1)-1))]
    out2 = [[arr2[0]] for _ in range(2**(len(arr2)-1))]

    return _permute(arr1, arr2, 1, len(arr1)-1, out1) + _permute(arr1, arr2, 1, len(arr1)-1, out2)

def _permute(arr1, arr2, ind, n, out):
    # Recursive permute helper function
    if n <= 0:
        return out

    out1 = out[:len(out)//2]
    out2 = out[len(out)//2:]
    for i in range(len(out1)):
        out1[i].append(arr1[ind])
    for i in range(len(out2)):
        out2[i].append(arr2[ind])
    
    return _permute(arr1, arr2, ind+1, n-1, out1) + _permute(arr1, arr2, ind+1, n-1, out2)

def lerp(g0, g1, t):
    # Returns the linear interpolation of two values/arrays with a fade
    return g0 + t*(g1 - g0)

def nlerp(g, t):
    # Returns the linear interpolation with a fade of each pair of values in the list
    # The list of values should have a length that is a multiple of 2
    # The first len(g)//2 values are used as the first value in the pair and the following len(g)//2 as the second
    return [lerp(g[i], g[len(g)//2 + i], t) for i in range(len(g)//2)]

def recurse_nlerp(n, g, t):
    return _recurse_nlerp(n, 0, g, t)

def _recurse_nlerp(n, ind, g, t):
    if n <= 0:
        return g
    return _recurse_nlerp(n-1, ind+1, nlerp(g, t[ind]), t)





# *---------------*
#  ND Perlin Noise
# *---------------*

def generate_gradient(n, shape, seed=None):
    """Creates a random ND gradient.

    Arguments:
        n {int} -- The number of dimensions of the gradient.
        shape {int} -- The shape of the gradient.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        numpy.ndarray -- The random ND gradient.
    """

    # Check that the shape is n-dimensional and has at least 1 dimension
    if (len(shape) != n or n <= 0): raise ShapeException

    if seed: np.random.seed(seed)
    return np.random.rand(*shape, n)*2 - 1

def linear_spacing(n, shape, frequency):
    """Generates a ND linear spacing.

    Arguments:
        n {int} -- The number of dimensions of the gradient.
        shape {int} -- The shape of the gradient.
        frequency {float} -- The frequency of the linear spacing.

    Returns:
        tuple -- The local and gradient coordinates (both lists of numpy.ndarray).
    """

    # Check that the shape is n-dimensional and has at least 1 dimension
    if (len(shape) != n or n <= 0): raise ShapeException

    # Linear spacing by frequency
    n_vals = [np.linspace(0, frequency, _n, endpoint=False) for _n in shape]
    for i in range(n):
        for j in range(i):
            n_vals[i] = np.tile(n_vals[i], shape[j])
        for j in range(i + 1, n):
            n_vals[i] = np.repeat(n_vals[i], shape[j])

    # Gradient coordinates
    n0 = [_n_vals.astype(np.int64) for _n_vals in n_vals]

    # Local coordinates
    for _n_vals, _n0 in zip(n_vals, n0):
        _n_vals -= _n0

    return n_vals, n0

def generate_gradient_projections(n, shape, frequency, seed=None):
    """Creates a ND gradient and calculates the projections into the gradient that are linearly spaced.

    Arguments:
        n {int} -- The number of dimensions of the gradient.
        shape {int} -- The shape of the gradient.
        frequency {float} -- The frequency of the linear spacing.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        tuple -- Returns the gradient projections (list of numpy.ndarray) and the local coordinates from the
                 generated linear spacing (list of numpy.ndarray).
    """

    # Check that the shape is n-dimensional and has at least 1 dimension
    if (len(shape) != n or n <= 0): raise ShapeException

    gradient = generate_gradient(n, shape, seed=seed)
    n_vals, n0 = linear_spacing(n, shape, frequency)
    n1 = [_n0 + 1 for _n0 in n0]

    # Values are in big endian e.g. [[n0[0], n0[1], n0[2]], [n0[0], n0[1], n1[2]], [n0[0], n1[1], n0[2]], [n0[0], n1[1], n1[2]],
    #                                [n1[0], n0[1], n0[2]], [n1[0], n0[1], n1[2]], [n1[0], n1[1], n0[2]], [n1[0], n1[1], n1[2]]] for 3D
    n0_n1 = np.array(permute(n0, n1))

    # Gradient projections
    gradient_projs = [gradient[_n0_n1] for _n0_n1 in n0_n1]

    return gradient_projs, n_vals

def perlin_noise(n, shape, frequency, seed=None):
    """Generates 3D Perlin noise.

    Arguments:
        n {int} -- The number of dimensions of the gradient.
        shape {int} -- The shape of the gradient.
        frequency {float} -- The frequency of the linear spacing.

    Keyword Arguments:
        seed {int} -- The seed for random number generation. (default: {None})

    Returns:
        numpy.ndarray -- An array of floating point numbers.
    """

    gradient_projs, n_vals = generate_gradient_projections(n, shape, frequency, seed=seed)

    # Coordinate fades
    t = [(3 - 2*vals) * vals*vals for vals in n_vals]

    # Compute dot product of distance and gradient vectors
    g = []
    for i in range(len(gradient_projs)):
        g.append(n_vals[0]*gradient_projs[i][0])
        for j in range(1, n):
            g[i] += n_vals[j]*gradient_projs[i][j]

    # N-directional linear interpolations
    g = recurse_nlerp(n, g, t)

    return g.reshape(shape)





def main():
    shape = (20, 20, 20)
    n = len(shape)
    seed, num_bands = 69420, 4
    frequency = 2

    image_array = perlin_noise(n, shape, frequency, seed=seed)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('You interrupted the program, you jerk.')