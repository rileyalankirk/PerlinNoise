import cave_gen_worker
import numpy as np
import pickle
import pyrender
from pyrender.constants import RenderFlags
import trimesh
import skimage.io as io


def main():

    cube_tasks = 10
    perlin_iterations = 10
    size = [20,20,20]

    # Create tasks to do banded perlin noise
    perlin_results = []
    for i in range(perlin_iterations):
        perlin_results.append(cave_gen_worker.generate_perlin_noise.delay(size[0], size[1], size[2], 2.0, 1.0, seed=2556))

    # Now go through all tasks, average results together to finish banded perlin noise
    environment = np.zeros((size[0],size[1],size[2]))
    for result in perlin_results:
        environment += np.array(pickle.loads(result.get())) / perlin_iterations
    print(environment.shape)
    # Getting all data to be above 0 for marching cubes
    environment = environment + np.abs(np.min(environment))
    environment = environment.reshape(size[0],size[1],size[2])


    marching_cubes_results = []

    # Getting threshold to render marching cubes at.
    threshold = np.mean(environment)

    # Now we need to split the data up into slices to send out as separate tasks
    group_size = size[0]//cube_tasks
    if group_size < 2:
        group_size = 2
    for i in range(cube_tasks):
        start_index = i*group_size
        end_index = (i+1)*group_size + 1

        '''
        It's possible that the way the slicing is done, we could end up with a single 2D
        slice as the last thing sent to a worker. This of course wouldn't work too well
        with marching cubes, so we check for this below and simply add that would-be 1-layer
        slice on to the slice that comes before it
        '''
        if end_index >= size[0]-2:
            marching_cubes_results.append(cave_gen_worker.build_marching_cube_mesh.delay(environment[start_index:], threshold, [start_index, 0, 0]))
            break
        marching_cubes_results.append(cave_gen_worker.build_marching_cube_mesh.delay(environment[start_index:end_index], threshold, [start_index, 0, 0]))

    vertices = []
    indices = []

    # Now go through all marching cubes results and get vertices and indices data, add them all on
    for result in marching_cubes_results:
        new_vertices, new_indices = pickle.loads(result.get())
        new_indices = np.array(new_indices)

        new_indices += len(vertices)
        vertices += list(new_vertices)
        indices += list(new_indices)

    # Now render the scene

    # Setting up meshes
    colors = [[0.25, 0.5, 0.25]] * len(vertices)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=indices, vertex_colors=colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene()
    scene.add(mesh)


    # Setting up camera object
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2)/2
    camera_pose = np.array([
       [0.0, -s,   s,   size[0]*1.25],
       [1.0,  0.0, 0.0, size[1]/2.5],
       [0.0,  s,   s,   size[2]*1.25],
       [0.0,  0.0, 0.0, 1.0],
    ])

    scene.add(camera, pose=camera_pose)


    # Setting up lighting
    lights = []

    for a in range(2):
        for b in range(2):
            for c in range(2):
                lights.append(np.array([
                   [0.0, -s,   s,   size[0] * (-1**a)],
                   [1.0,  0.0, 0.0, size[1] * (-1**b)],
                   [0.0,  s,   s,   size[2] * (-1**c)],
                   [0.0,  0.0, 0.0, 1.0],
                ]))

    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1000.0, range=1000.0)

    for l in lights:
        scene.add(light, pose=l)

    # Finally, render. This will open an interactive viewer.
    pyrender.Viewer(scene, render_flags={"cull_faces":False})

    # This will save the mesh as an OBJ file
    # with open("test_mesh.obj", "w") as f:
    #     f.write(trimesh.exchange.export.export_obj(tri_mesh))



if __name__ == "__main__":
    main()
