import numpy as np
from h5py import File
import SimpleITK as sitk
from pathlib import Path


def save_by_ITK(filepath, data, voxel_size=4., format='dcm'):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    shape = np.array(data.shape)
    data = data.ravel(order='F').reshape(shape[::-1])
    data = np.rot90(data, k=1, axes=(1, 2))
    spacing = [voxel_size]*3
    origin = (1 - shape)*voxel_size/2
    print(np.unique(data))
    dicom_image = sitk.GetImageFromArray(data)
    dicom_image.SetSpacing(spacing)
    dicom_image.SetOrigin(origin)
    dicom_image.SetMetaData('0010|0010', filepath.split('/')[-1])
    sitk.WriteImage(dicom_image, filepath + f'.{format}')

def from_dicom_to_npy(filepath):
    dicom_image = sitk.ReadImage(filepath + '.dcm')
    data = sitk.GetArrayFromImage(dicom_image)
    return data

def from_mhd_to_npy(filepath):
    dicom_image = sitk.ReadImage(filepath + '.mhd')
    data = sitk.GetArrayFromImage(dicom_image)
    return data

def save_as_npy(filepath, data):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath + '.npy', data)

def save_as_dat(filepath, data, split=False):
    if split:
        Path(filepath).mkdir(parents=True, exist_ok=True)
        for i, slice in enumerate(data, 1):
            np.savetxt(filepath + f'/{i}.dat', slice, fmt='\t%.6E')
    else:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = data.ravel(order='F')
        np.savetxt(filepath + '.dat', data, fmt='\t%.6E')

def from_dat_to_npy(filepath, size):
    data = np.loadtxt(filepath + '.dat').reshape(size, order='F')
    return data

def from_hdf_to_npy(filepath, distribution='Dose'):
    data = File(filepath + '.hdf')
    distribution = data[distribution + ' distribution']
    volume = np.array(distribution['Volume'])[::-1]
    voxel_size = np.array(distribution['Voxel size'])
    return volume, voxel_size

def change_values(data):

    # data[data == 8] /= 5
    # data[data == 10] /= 5
    # data[data == 12] /= 5    

    # data[data == 2] = 6

    # data[data == 0.28] = 0.15
    
    data[data==40] = 10
    data[data==30] = 20
    data[data==70] = 40
    data[data==80] = 40
    data[data==89] = 50
    data[data==140] = 200

    # data[data == 2] = 0.        #Воздух
    # data[data == 400] = 0.035     #Лёгкие
    # data[data == 1500] = 0.146   #Мягкие ткани
    # data[data == 1650] = 0.162   #Мягкие ткани
    # data[data == 2900] = 0.298    #Кости

    # data[data == 0.] = 0 #Воздух
    # data[data == 10.] = 1 #Лёгкие
    # data[data == 1.] = 2 #Мягкие ткани
    # data[(data >= 2.)*(data <= 8)] = 3 #Кости




size = np.array([
    100,
    100,
    50
])
phantom_name = 'hoffman_tumor_activity'
filepathDAT = lambda:f'Dat phantoms/{phantom_name}'
filepathHDF = lambda:f'Raw data/{phantom_name}'
filepathNPY = lambda:f'Numpy phantoms/{phantom_name}'
filepathDICOM = lambda:f'DICOM phantoms/{phantom_name}'
filepathMHD = lambda:f'MHD phantoms/{phantom_name}'


data = from_dat_to_npy(filepathDAT(), size)
# data *= /data.max()
# data = data.astype(np.int16)
save_by_ITK(filepathMHD(), data, voxel_size=4., format='mhd')
# save_by_ITK(filepathDICOM(), data, format='dcm')
save_as_npy(filepathNPY(), data)


# image = sitk.ReadImage(filepathMHD() + '.mhd')
# # sitk.WriteImage(image, filepathDICOM() + '.vtk')
# # exit()

# image_array = sitk.GetArrayFromImage(image)
# print(np.unique(image_array))

# shape = np.array(image_array.shape)
# image_array = image_array.ravel(order='F').reshape(shape[::-1])
# image_array = np.rot90(image_array, k=1, axes=(0, 1))

# change_values(image_array)

# # image_array = image_array.astype(int)
# print(np.unique(image_array))
# save_as_npy(filepathNPY(), image_array)
# save_as_dat(filepathDAT(), image_array)






# data = []
# for i in range(1, 129):
#     proj = np.loadtxt(f'Dat data/{phantom_name}/proj_{i}.dat').reshape((128, 128), order='F')
#     proj = np.rot90(proj, k=-1)
#     data.append(proj)
# data = np.stack(data, axis=2)
# data = from_dicom_to_npy(filepathDICOM())
# data = from_mhd_to_npy(filepathMHD())
# data = data.astype(float)
# data = data[::-1, :, :]
# data = from_dat_to_npy(filepathDAT(), size)
# for i, x in enumerate(np.unique(data)):
#     data[data == x] = i
# print(np.unique(data))
# data = data[:, :, 75:243-56]

# change_values(data)
# data = data.astype(np.uint)
# print(np.unique(data))

# data = (data*300*10**6)/data.sum()
# print(np.unique(data))

# phantom_name = 'efg3cut'

# data = np.rot90(data, k=-1, axes=(0, 1))
# data = np.rot90(data, k=1, axes=(0, 2))

# save_by_ITK(filepathDICOM(), data)
# save_by_ITK(filepathMHD(), data, format='mhd')
# unique = np.unique(data)
# for value, new_value in zip(unique, np.linspace(0, unique.size - 1, unique.size, endpoint=True)):
#     data[data == value] = new_value
# data = data.astype(np.ubyte)
# save_by_ITK(filepathMHD(), data, format='mhd')

# save_as_dat(filepathDAT(), data, split=False)

