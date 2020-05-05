import cave_gen_worker
import numpy as np
import pickle
import trimesh
import skimage.io as io
from time import monotonic

def main():

    cube_tasks = 10
    perlin_iterations = 100
    size = [20,20,20]
    perlin_time = 0
    cubes_time = 0

    # Create tasks to do banded perlin noise
    perlin_results = []
    data = [[size[0], size[1], size[2], 2.0, 1.0]]*perlin_iterations
    start = monotonic()
    for result in ~cave_gen_worker.generate_perlin_noise.starmap(zip(data)):
        perlin_results.append(result)
    perlin_time += monotonic() - start
    print("Total Time for Perlin Noise Calls:", perlin_time, "seconds")
    print("Perlin Noise Average Time:", (perlin_time / perlin_iterations), "seconds")

    # Getting all data to be above 0 for marching cubes
    environment = np.random.uniform(size=(size[0],size[1],size[2]))
    marching_cubes_results = []
    cubes_data = []

    # Getting threshold to render marching cubes at.
    threshold = np.mean(environment)

    # Now we need to split the data up into slices to send out as separate tasks
    group_size = size[0]//cube_tasks
    if group_size < 2:
        group_size = 2
    for i in range(cube_tasks):
        start_index = i*group_size
        end_index = (i+1)*group_size + 1
        if end_index >= size[0]-2:
            cubes_data.append([environment[start_index:], threshold, [start_index, 0, 0]])
            break
        cubes_data.append([environment[start_index:end_index], threshold, [start_index, 0, 0]])
    start = monotonic()
    for result in ~cave_gen_worker.build_marching_cube_mesh.starmap(zip(cubes_data)):
        marching_cubes_results.append(result)    
    cubes_time += monotonic() - start
    print("Total Time for Marching Squares Calls:", (cubes_time / 60), "minutes (" + str(cubes_time) + " seconds)")
    print("Marching Squares Average Time:", (cubes_time / cube_tasks), "seconds")

if __name__ == "__main__":
    main()
