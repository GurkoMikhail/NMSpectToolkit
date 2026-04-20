from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pyvista as pv
import SimpleITK as sitk
from h5py import File
from hepunits import *
from numba import jit
from pathlib import Path


class DataExtractor:
    def __init__(self, max_processes=32, time_interval=None, events_limit=None):
        self.max_processes = max_processes
        self.time_interval = time_interval
        self.events_limit = events_limit
        self.translator = {
            "local_position": "local_position",
            "global_position": "global_position",
            "energy_deposit": "energy_deposit",
            "distance_traveled": "distance_traveled",
            "particle_ID": "particle_ID",
            "process_name": "process_name",
            "scattering_angles": "scattering_angles",
            "species": "species",
            "material_id": "material_id",
            "volume_id": "volume_id"
        }

    def extract_data(self, filepath):
        if isinstance(filepath, list):
            with Pool(self.max_processes) as pool:
                return pool.map(self._extract_data, filepath)
        return self._extract_data(filepath)

    def _extract_data(self, filepath):
        filename = filepath.split(sep="/")[-1]
        print(f"Reading {filename}")
        with File(filepath, "r") as file:
            data = self._data_extraction(file)
        return data

    def _data_extraction(self, file):
        data = {}
        interactions = file.get("interactions")
        if not interactions:
            print("Warning: No 'interactions' group found in file.")
            return data

        initial_states = file.get("initial_states")
        has_initial = initial_states is not None

        # Подготавливаем индексы для быстрого маппинга данных из initial_states
        if has_initial:
            init_pids = initial_states['particle_ID'][:]
            sort_idx_init = np.argsort(init_pids)
            sorted_init_pids = init_pids[sort_idx_init]

        for volume_name, volume_group in interactions.items():
            vol_data = {}
            for key in volume_group.keys():
                if key in self.translator:
                    vol_data[self.translator[key]] = np.copy(volume_group[key])
                else:
                    vol_data[key] = np.copy(volume_group[key])

            pids = vol_data.get('particle_ID')
            n_events = len(pids) if pids is not None else 0

            # 1. Извлекаем начальные состояния, если это HistoryAssemblerHandler
            if has_initial and pids is not None:
                idx = np.searchsorted(sorted_init_pids, pids)
                # Защита от выхода за границы
                idx = np.clip(idx, 0, len(sorted_init_pids) - 1)
                valid_mask = (sorted_init_pids[idx] == pids)
                mapped_idx = sort_idx_init[idx]

                if 'emission_position' in initial_states:
                    vol_data['emission_position'] = initial_states['emission_position'][:][mapped_idx]
                    vol_data['emission_position'][~valid_mask] = 0.0
                if 'emission_direction' in initial_states:
                    vol_data['emission_direction'] = initial_states['emission_direction'][:][mapped_idx]
                    vol_data['emission_direction'][~valid_mask] = 0.0
                
                time_key = 'emission_time' if 'emission_time' in initial_states else 'time'
                if time_key in initial_states:
                    vol_data['emission_time'] = initial_states[time_key][:][mapped_idx]
                    vol_data['emission_time'][~valid_mask] = 0.0
                
                energy_key = 'emission_energy' if 'emission_energy' in initial_states else 'energy'
                if energy_key in initial_states:
                    vol_data['emission_energy'] = initial_states[energy_key][:][mapped_idx]
                    vol_data['emission_energy'][~valid_mask] = 0.0

            # 2. Заглушки, если это SensitiveVolumeHandler (без initial_states)
            if 'emission_time' not in vol_data:
                vol_data['emission_time'] = np.zeros(n_events, dtype=float)
            if 'emission_energy' not in vol_data:
                vol_data['emission_energy'] = np.zeros(n_events, dtype=float)
            if 'emission_position' not in vol_data:
                vol_data['emission_position'] = np.zeros((n_events, 3), dtype=float)
            if 'emission_direction' not in vol_data:
                vol_data['emission_direction'] = np.zeros((n_events, 3), dtype=float)

            # 3. Сортировка и фильтрация по времени (или по ID частицы)
            has_real_time = np.any(vol_data['emission_time'] != 0)
            sort_key = vol_data['emission_time'] if has_real_time else vol_data['particle_ID']
            indices_sort = np.argsort(sort_key)

            if self.time_interval is not None and has_real_time:
                sorted_time = vol_data['emission_time'][indices_sort]
                lower_idx = np.searchsorted(sorted_time, self.time_interval[0], side="left")
                upper_idx = np.searchsorted(sorted_time, self.time_interval[1], side="right")
                indices_sort = indices_sort[lower_idx:upper_idx]

            if self.events_limit is not None and indices_sort.size > self.events_limit:
                indices_sort = indices_sort[: int(self.events_limit)]

            for key in vol_data.keys():
                vol_data[key] = vol_data[key][indices_sort]

            data[volume_name] = vol_data

        return data


class DataProcessor:
    def __init__(self):
        self.rng = np.random.default_rng()

    def cut_emission_ROI(self, data, emission_ROI):
        emission_position = data["emission_position"]
        emission_ROI = np.asarray(emission_ROI)
        ROI = np.all((emission_position >= emission_ROI[:, 0]) * (emission_position < emission_ROI[:, 1]), axis=1)
        indices = np.nonzero(ROI)[0]
        for key in data.keys():
            data[key] = data[key][indices]

    def add_energy_deviation(self, data, energy_resolution, reference_energy):
        energy = data["energy_deposit"]
        coeff = np.sqrt(reference_energy) * energy_resolution / 100
        resolution_distribution = coeff / np.sqrt(energy)
        sigma = resolution_distribution * energy / 2.355
        data["registrated_energy"] = self.rng.normal(energy, sigma)
        return data

    def add_position_deviation(self, data, spatial_resolution):
        local_position = data["local_position"]
        sigma = spatial_resolution / 2.35
        data["registrated_position"] = self.rng.normal(local_position, sigma)

    def unite_acts(self, data, decay_time=0.0, use_distance_traveled=True):
        local_position = data["local_position"]
        global_position = data["global_position"]
        energy_deposit = data["energy_deposit"]
        emission_time = data["emission_time"]
        emission_energy = data["emission_energy"]
        emission_position = data["emission_position"]
        emission_direction = data["emission_direction"]
        distance_traveled = data["distance_traveled"]
        particle_ID = data["particle_ID"]

        has_real_time = np.any(emission_time != 0)
        registration_time = np.zeros_like(particle_ID, dtype=float)

        # Выбираем ключ для группировки актов
        if has_real_time:
            registration_time = emission_time + (distance_traveled / c_light if use_distance_traveled else 0)
            registration_time = self.culc_registration_time(registration_time, decay_time)
            grouping_key = registration_time
        else:
            # Если времени нет, группируем строго по Particle ID (никаких pile-up)
            grouping_key = particle_ID

        # Важно: сортируем массивы перед вызовом np.unique, 
        # иначе np.arange захватит неверные смежные элементы
        sort_idx = np.argsort(grouping_key)
        grouping_key = grouping_key[sort_idx]
        registration_time = registration_time[sort_idx]
        
        local_position = local_position[sort_idx]
        global_position = global_position[sort_idx]
        energy_deposit = energy_deposit[sort_idx]
        emission_time = emission_time[sort_idx]
        emission_energy = emission_energy[sort_idx]
        emission_position = emission_position[sort_idx]
        emission_direction = emission_direction[sort_idx]
        distance_traveled = distance_traveled[sort_idx]
        particle_ID = particle_ID[sort_idx]

        unique_keys, indices, counts = np.unique(grouping_key, return_index=True, return_counts=True)
        events_number = indices.size
        events_indices = [np.arange(indices[i], indices[i] + counts[i]) for i in range(events_number)]
        
        averaged_local_position = np.zeros((events_number, 3), dtype=float)
        averaged_global_position = np.zeros((events_number, 3), dtype=float)
        primary_position = np.zeros((events_number, 3), dtype=float)
        averaged_distance_traveled = np.zeros(events_number, dtype=float)
        primary_distance_traveled = np.zeros(events_number, dtype=float)
        averaged_emission_time = np.zeros(events_number, dtype=float)
        averaged_emission_energy = np.zeros(events_number, dtype=float)
        sum_energy_deposit = np.zeros(events_number, dtype=float)
        averaged_emission_position = np.zeros((events_number, 3), dtype=float)
        averaged_emission_direction = np.zeros((events_number, 3), dtype=float)
        averaged_particle_ID = np.zeros(events_number, dtype=np.uint64)
        
        del_indices = []
        for i, acts_indices in enumerate(events_indices):
            weights = energy_deposit[acts_indices]
            if weights.sum() > 0:
                averaged_local_position[i] = np.average(local_position[acts_indices], axis=0, weights=weights)
                averaged_global_position[i] = np.average(global_position[acts_indices], axis=0, weights=weights)
                primary_position[i] = global_position[acts_indices[0]]
                averaged_distance_traveled[i] = np.average(distance_traveled[acts_indices], weights=weights)
                primary_distance_traveled[i] = distance_traveled[acts_indices[0]]
                averaged_emission_time[i] = np.average(emission_time[acts_indices], weights=weights)
                averaged_emission_energy[i] = np.amax(emission_energy[acts_indices])
                sum_energy_deposit[i] = np.sum(energy_deposit[acts_indices])
                averaged_emission_position[i] = np.average(emission_position[acts_indices], axis=0, weights=weights)
                averaged_emission_direction[i] = np.average(emission_direction[acts_indices], axis=0, weights=weights)
                averaged_particle_ID[i] = np.amax(particle_ID[acts_indices])
            else:
                del_indices.append(i)
                
        del_indices = np.array(del_indices, dtype=int)

        data["registration_time"] = np.delete(unique_keys if has_real_time else registration_time[indices], del_indices)
        data["local_position"] = np.delete(averaged_local_position, del_indices, axis=0)
        data["global_position"] = np.delete(averaged_global_position, del_indices, axis=0)
        data["primary_position"] = np.delete(primary_position, del_indices, axis=0)
        data["distance_traveled"] = np.delete(averaged_distance_traveled, del_indices)
        data["primary_distance_traveled"] = np.delete(primary_distance_traveled, del_indices)
        data["emission_time"] = np.delete(averaged_emission_time, del_indices)
        data["emission_energy"] = np.delete(averaged_emission_energy, del_indices)
        data["emission_position"] = np.delete(averaged_emission_position, del_indices, axis=0)
        data["emission_direction"] = np.delete(averaged_emission_direction, del_indices, axis=0)
        data["energy_deposit"] = np.delete(sum_energy_deposit, del_indices)
        data["particle_ID"] = np.delete(averaged_particle_ID, del_indices)
        return data

    @staticmethod
    @jit(nopython=True, cache=True)
    def culc_registration_time(time, decay_time):
        time_with_decay = np.zeros_like(time)
        countdown_time = 0.0
        for i, t in enumerate(time):
            if (t - countdown_time) > decay_time:
                countdown_time = t
            time_with_decay[i] = countdown_time + decay_time
        return time_with_decay


class DataConverter:
    def __init__(self, max_processes=32):
        self.max_processes = max_processes
        self.processing_parameters = {
            "decay_time": 300 * ns,
            "spatial_resolution": 4.0 * mm,
            "energy_resolution": 9.9,
            "reference_energy": 140.5 * keV,
            "energy_channels": 1024,
            "energy_range": [0, 300 * keV],
            "energy_windows": [
                [1.0, [126 * keV, 154 * keV]],
                [-0.5, [98 * keV, 126 * keV]]],
            "image_range": None,
            "pixel_size": 4.0 * mm,
            "matrix": [128, 128],
            "use_distance_traveled": True,
            "voxel_size": 4.0 * mm,
            "emission_ROI": [[-600.0, 600.0 * cm], [-600.0, 600.0 * cm], [-400.0 * cm, 400.0 * cm]],
        }
        self.vtk_points_parameters = {
            "decay_time": 300.0 * ns,
            "spatial_resolution": 4.0 * mm,
            "energy_resolution": 9.9,
            "reference_energy": 140.5 * keV,
            "use_distance_traveled": True,
            "emission_ROI": False,
        }

    def _get_matrix_and_image_range(self, parameters):
        if parameters["image_range"] is None:
            matrix = np.asarray(parameters["matrix"], dtype=int)
            pixel_size = np.asarray(parameters["pixel_size"])
            image_range = matrix * pixel_size
            image_range = np.column_stack([-image_range / 2, image_range / 2])
        else:
            image_range = np.asarray(parameters["image_range"])
        if parameters["matrix"] is None:
            matrix = np.round(((image_range[:, 1] - image_range[:, 0]) / parameters["pixel_size"])).astype(int)
        else:
            matrix = np.asarray(parameters["matrix"], dtype=int)
        parameters["pixel_size"] = (image_range[:, 1] - image_range[:, 0]) / matrix
        return matrix, image_range

    def convert_to_vtk_point_cloud(self, data, processing_parameters={}):
        print("\tConverting to VTK point cloud")
        self.update_parameters(self.vtk_points_parameters, processing_parameters)
        if isinstance(data, list):
            with Pool(min(len(data), self.max_processes)) as pool:
                return pool.map(self._convert_to_vtk_point_cloud, data)
        return self._convert_to_vtk_point_cloud(data)

    def convert_to_image(self, data, processing_parameters={}):
        print("\tConverting to image")
        self.update_parameters(self.processing_parameters, processing_parameters)
        if isinstance(data, list):
            with Pool(min(len(data), self.max_processes)) as pool:
                return pool.map(self._convert_to_image, data)
        return self._convert_to_image(data)

    def _convert_to_vtk_point_cloud(self, data):
        processed_data = self.process_data(data["interactions_data"], self.vtk_points_parameters)
        point_cloud = pv.PolyData(processed_data["global_position"])
        for key, value in processed_data.items():
            point_cloud[key] = value
        return point_cloud

    def _convert_to_image(self, data):
        processed_data = self.process_data(data["interactions_data"], self.processing_parameters)
        matrix, image_range = self._get_matrix_and_image_range(self.processing_parameters)
        image_array = np.zeros(matrix)
        for coeff, energy_window in self.processing_parameters["energy_windows"]:
            valid_events = self.cut_to_energy_window(processed_data["registrated_energy"], energy_window)
            registrated_position = processed_data["registrated_position"][valid_events]
            image_array += coeff * np.histogram2d(
                registrated_position[:, 0], registrated_position[:, 1], bins=matrix, range=image_range
            )[0]
        return image_array

    @staticmethod
    def update_parameters(parameters, new_parameters):
        for key, value in new_parameters.items():
            if key in parameters:
                parameters[key] = value

    @staticmethod
    def cut_to_energy_window(energy_deposit, energy_window):
        return np.nonzero((energy_deposit >= energy_window[0]) * (energy_deposit <= energy_window[1]))[0]

    @staticmethod
    def process_data(interactions_data, processing_parameters):
        dataProcessor = DataProcessor()
        interactions_data = deepcopy(interactions_data)
        if processing_parameters["emission_ROI"]:
            dataProcessor.cut_emission_ROI(interactions_data, processing_parameters["emission_ROI"])
        if processing_parameters["decay_time"]:
            dataProcessor.unite_acts(
                interactions_data, processing_parameters["decay_time"], processing_parameters["use_distance_traveled"]
            )
        if processing_parameters["energy_resolution"]:
            dataProcessor.add_energy_deviation(
                interactions_data,
                processing_parameters["energy_resolution"],
                processing_parameters["reference_energy"],
            )
        if processing_parameters["spatial_resolution"]:
            dataProcessor.add_position_deviation(interactions_data, processing_parameters["spatial_resolution"])
        return interactions_data


class DataSaver:
    def __init__(self, data, filename, data_type=None, pixel_size=4):
        self.data = np.asarray(data)
        self._filename = filename
        self.data_type = data_type
        self.pixel_size = pixel_size

    @property
    def filename(self):
        if self.data.ndim > 2:
            return self._filename.split("/")[0] + ("" if self.data_type is None else "_" + self.data_type)
        return self._filename.split("/")[-1] + ("" if self.data_type is None else "_" + self.data_type)

    def save_as_numpy(self, rot=False):
        print(f"Saving {self.filename} as Numpy")
        data = self.data
        if rot:
            if self.data.ndim > 2:
                data = np.rot90(self.data, k=-1, axes=(1, 2))
            else:
                data = np.rot90(self.data, k=-1)
                
        Path("Numpy data").mkdir(parents=True, exist_ok=True)
        
        np.save(f"Numpy data/{self.filename}.npy", data)

    def save_as_dicom(self):
        print(f"Saving {self.filename} as Dicom")
        if self.data.ndim > 2:
            data = np.rot90(self.data, k=-1, axes=(1, 2))
            data = data[:, ::-1]
        else:
            data = np.rot90(self.data, k=-1)
            data = data[::-1]
        data = data.astype(np.uint16)
        
        image = sitk.GetImageFromArray(data)
        shape = np.array(data.shape)
        origin = [*((1 - shape) * self.pixel_size / 2), 0.5]
        spacing = [self.pixel_size, self.pixel_size, 1]
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetMetaData("0010|0010", self.filename)
        image.SetMetaData("0018|0070", str(data.sum()))
        
        Path("DICOM data").mkdir(parents=True, exist_ok=True)
        
        sitk.WriteImage(image, f"DICOM data/{self.filename}.dcm")

    def save_as_dat(self):
        print(f"Saving {self.filename} as Dat")
        data = self.data
        
        if self.data.ndim > 2:
            Path(f"Dat data/{self.filename}").mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(data, 1):
                image = image[::-1]
                np.savetxt(f"Dat data/{self.filename}/{i}.dat", image, fmt="%i", delimiter="\t")
        else:
            Path("Dat data").mkdir(parents=True, exist_ok=True)
            data = data[::-1]
            np.savetxt(f"Dat data/{self.filename}.dat", data, fmt="%i", delimiter="\t")