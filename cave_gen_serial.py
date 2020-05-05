from cave_gen_worker import generate_perlin_noise
from cave_gen_worker import build_marching_cube_mesh
import numpy as np
import pickle
import trimesh
import skimage.io as io
from time import monotonic

def main():

    cube_tasks = 10
    perlin_iterations = 10000
    size = [20,20,20]
    perlin_time = 0
    cubes_time = 0

    # Create tasks to do banded perlin noise
    perlin_results = []
    for i in range(perlin_iterations):
        start = monotonic()
        array = generate_perlin_noise([size[0], size[1], size[2], 2.0, 1.0])
        perlin_time += monotonic() - start
    print("Total Time for Perlin Noise Calls:", perlin_time, "seconds")
    print("Perlin Noise Serial Average Time:", (perlin_time / perlin_iterations), "seconds")

    # Getting all data to be above 0 for marching cubes
    environment = np.random.uniform(size=(size[0],size[1],size[2]))
    marching_cubes_results = []

    # Getting threshold to render marching cubes at.
    threshold = np.mean(environment)

    # Now we need to split the data up into slices to send out as separate tasks
    group_size = size[0]//cube_tasks
    if group_size < 2:
        group_size = 2
    cube_iterations = 0
    for j in range(100):
        for i in range(cube_tasks):
            cube_iterations += 1
            start_index = i*group_size
            end_index = (i+1)*group_size + 1
            start = monotonic()
            if end_index >= size[0]-2:
                marching_cubes_results.append(build_marching_cube_mesh([environment[start_index:], threshold, [start_index, 0, 0]]))
                break
            marching_cubes_results.append(build_marching_cube_mesh([environment[start_index:end_index], threshold, [start_index, 0, 0]]))
            cubes_time += monotonic() - start
            print(cubes_time)
    print("Total Time for Marching Squares Calls:", (cubes_time / 60), "minutes")
    print("Marching Squares Serial Average Time:", (cubes_time / cube_iterations), "seconds")

if __name__ == "__main__":
    main()