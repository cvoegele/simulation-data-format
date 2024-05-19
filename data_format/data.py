from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pyvista as pv
import vtk
import vtkmodules.util.numpy_support as numpy_support


def create_file(filename: str, x_dim, y_dim, z_dim, chunk_size=100, additional_dimension=7):
    with h5py.File(f'{filename}.h5', 'w') as f:
        total_dimension = 3 + additional_dimension
        dset = f.create_dataset('data', (x_dim, y_dim, z_dim, total_dimension), dtype='float32',
                                chunks=(chunk_size, chunk_size, chunk_size, total_dimension))

        dset.attrs['x_dim'] = x_dim
        dset.attrs['y_dim'] = y_dim
        dset.attrs['z_dim'] = z_dim

        def write_chunk(i, j, k):
            # Define the chunk slice
            x_slice = slice(i, min(i + chunk_size, x_dim))
            y_slice = slice(j, min(j + chunk_size, y_dim))
            z_slice = slice(k, min(k + chunk_size, z_dim))

            # Fill in the coordinate data
            xx, yy, zz = np.meshgrid(np.arange(x_slice.start, x_slice.stop),
                                     np.arange(y_slice.start, y_slice.stop),
                                     np.arange(z_slice.start, z_slice.stop), indexing='ij')

            # Create an array to hold the data for this chunk
            chunk_data = np.zeros((xx.shape[0], yy.shape[1], zz.shape[2], total_dimension), dtype='float32')

            # Fill the first three channels with coordinates
            chunk_data[..., 0] = xx
            chunk_data[..., 1] = yy
            chunk_data[..., 2] = zz

            # Fill the rest with random data
            chunk_data[..., 3:] = np.random.rand(xx.shape[0], yy.shape[1], zz.shape[2], additional_dimension)

            # Write the chunk to the dataset
            dset[x_slice, y_slice, z_slice, :] = chunk_data

        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(0, x_dim, chunk_size):
                for j in range(0, y_dim, chunk_size):
                    for k in range(0, z_dim, chunk_size):
                        print(f"writing chunk: {i} {j} {k}")
                        executor.submit(write_chunk, i, j, k)


def get_sub_cube_from_file(filename: str, start_x, start_y, start_z, x_size, y_size, z_size):
    with h5py.File(f'{filename}.h5', 'r') as f:
        dataset = f["data"]
        data = dataset[start_x:(start_x + x_size), start_y:(start_y + y_size), start_z:(start_z + z_size), :]
        return data


def write_sub_cube_to_file(filename: str, cube_to_replace: np.array, start_x, start_y, start_z):
    with h5py.File(f'{filename}.h5', 'r+') as f:
        dataset = f["data"]
        shape_of_cube = cube_to_replace.shape
        x_size = shape_of_cube[0]
        y_size = shape_of_cube[1]
        z_size = shape_of_cube[2]
        dataset[start_x:(start_x + x_size), start_y:(start_y + y_size), start_z:(start_z + z_size), :] = cube_to_replace


def numpy_to_vtk(data, output_file):
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data[:, :, :, 5].flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])

    # Save the VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(f"{output_file}.vti")
    writer.SetInputData(img)
    writer.Write()


def sub_cube_to_vtk(data, out_filename):
    grid = pv.ImageData()

    # Set the grid dimensions
    grid.dimensions = np.array(data.shape) + 1

    # Edit the origin and spacing of the grid
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # Add the NumPy array as the grid's cell data
    grid.cell_data["values"] = data.flatten(order="F")  # Flatten the array

    # Save the grid to a VTK file
    grid.save(f"{out_filename}.vtk")
