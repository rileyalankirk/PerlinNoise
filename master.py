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

    # Create tasks to do banded perlin noise, and then through all results and average them together to finish banded perlin noise
    environment = np.zeros((size[0],size[1],size[2]))
    data = [[size[0], size[1], size[2], 2.0, 1.0]]*perlin_iterations
    for result in ~cave_gen_worker.generate_perlin_noise.starmap(zip(data)):
        environment += np.array(pickle.loads(result)) / perlin_iterations

    # Getting all data to be above 0 for marching cubes program
    environment = environment + np.abs(np.min(environment))
    environment = environment.reshape(size[0],size[1],size[2])


    cubes_data = []

    # Getting threshold to render marching cubes at.
    threshold = np.mean(environment)

    # Now we need to split the data up into slices for sending out as separate tasks
    group_size = size[0]//cube_tasks
    # Want to make sure the group size is at least 2, since if it were 1 or 0 we wouldn't be sending any actual cubes
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
            cubes_data.append([environment[start_index:], threshold, [start_index, 0, 0]])
            break
        cubes_data.append([environment[start_index:end_index], threshold, [start_index, 0, 0]])

    vertices = []
    indices = []

    # Create marching cubes tasks and go through all results
    for result in ~cave_gen_worker.build_marching_cube_mesh.starmap(zip(cubes_data)):
        # Get vertices and indices data generated
        new_vertices, new_indices = pickle.loads(result)
        new_indices = np.array(new_indices)

        # Add on the vertices and indices to our lists
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
