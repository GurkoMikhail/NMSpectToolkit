from pathlib import Path
import numpy as np
from raw_data_processing import DataConverter, DataExtractor, DataSaver
import pyvista as pv
from hepunits import*


if __name__ == '__main__':
    filename = 'nema1-10'
    
    views = 40
    gamma_cameras = 2

    angles = np.linspace(0, 2*pi, views, endpoint=False)[:views//gamma_cameras]
    name_list = [f'{round(angle/degree, 1)} deg' for angle in angles]
    name_list = [f'Raw data/{filename}/' + name + '.hdf' for name in name_list]

    data_extractor = DataExtractor(max_processes=15, time_interval=[0*s, 15*s])
    
    data = data_extractor.extract_data(name_list)

    new_data = []
    angles = []
    for volumes in data:
        for volume_name, volume_data in volumes.items():
            angle = float(volume_name.split(' ')[2])
            angles.append(angle)
            new_data.append({'interactions_data': volume_data})
    data = [new_data[i] for i in np.argsort(angles)]

    data_converter = DataConverter(max_processes=40)
    data_converter.processing_parameters = {
            "decay_time": 300 * ns,
            "spatial_resolution": 4.0 * mm,
            "energy_resolution": 9.9,
            "reference_energy": 140.5 * keV,
            "energy_channels": 1024,
            "energy_range": [0, 300 * keV],
            "energy_windows": [
                # [1.0, [126 * keV, 154 * keV]],
                # [-0.5, [98 * keV, 126 * keV]],
                [1.0, [143 * keV, 175 * keV]],
                # [-0.5, [111 * keV, 143 * keV]],
                ],
            "image_range": None,
            "pixel_size": 2.5 * mm,
            "matrix": [128, 128],
            "use_distance_traveled": True,
            "voxel_size": 4.0 * mm,
            "emission_ROI": [[-600.0, 600.0 * cm], [-600.0, 600.0 * cm], [-400.0 * cm, 400.0 * cm]],
        }

    if True:
        images = data_converter.convert_to_image(data)
        images = np.roll(images, views//2, axis=0)
        # images = images[::-1]
        images[images < 0] = 0
 
        pixel_size = data_converter.processing_parameters['pixel_size']
        data_saver = DataSaver(images, filename + "_MEW", pixel_size=pixel_size)
        data_saver.save_as_numpy(rot=True)
        data_saver.save_as_dicom()
        data_saver.save_as_dat()
        
        exit()
    
    for i, point_cloud in enumerate(data_converter.convert_to_vtk_point_cloud(data), 1):
        dir = f"VTK data/{filename}_MEW6/"
        Path(dir).mkdir(parents=True, exist_ok=True)
        point_cloud.save(dir + f"/proj_processed {i}.vtp")
    
    # exit()
    
    data_extractor.translator.update({
            "particle_ID": "particle_ID",
            "process_name": "process_name",
            "scattering_angles": "scattering_angles",
    })
    data = data_extractor.extract_data(name_list)
    
    new_data = []
    angles = []
    for volumes in data:
        for volume_name, volume_data in volumes.items():
            angle = float(volume_name.split(' ')[2])
            angles.append(angle)
            new_data.append({'interactions_data': volume_data})
    data = [new_data[i] for i in np.argsort(angles)]
    
    for i, dat in enumerate(data, 1):
        interactions_data = dat["interactions_data"]
        point_cloud = pv.PolyData(interactions_data["global_position"])
        for key, value in interactions_data.items():
            if key == "process_name":
                process_id = np.zeros_like(value, dtype = int)
                process_id[value == b"ComptonScattering"] = 1
                process_id[value == b"CoherentScattering"] = 2
                point_cloud["process_id"] = process_id
            point_cloud[key] = value
        point_cloud.save(f"VTK data/{filename}/proj {i}.vtp")
        
    