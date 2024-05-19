from data_format.data import *

if __name__ == "__main__":
    file_name = "test_file"
    vtk_file_name = "test_vtk_file"

    create_file(file_name, 250, 250, 250, 50)
    data = get_sub_cube_from_file(file_name, 0, 0, 0, 10, 10, 10)
    print(data)

    replacement_cube = np.ones((10, 10, 10, 10))
    write_sub_cube_to_file(file_name, replacement_cube, 0, 0, 0)

    data = get_sub_cube_from_file(file_name, 0, 0, 0, 250, 250, 250)
    print(data)

    numpy_to_vtk(data, vtk_file_name)
