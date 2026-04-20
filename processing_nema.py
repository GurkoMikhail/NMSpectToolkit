import re
from pathlib import Path
import numpy as np
from hepunits import *

from raw_data_processing import DataConverter, DataExtractor, DataSaver

if __name__ == '__main__':
    # Базовые настройки папок
    filename = 'nema_4.2'
    data_dir = Path(f'Raw data/{filename}')
    
    # 1. АВТООПРЕДЕЛЕНИЕ ФАЙЛОВ
    # Ищем все файлы с расширением .hdf или .hdf5
    file_paths = list(data_dir.glob('*.hdf*'))
    if not file_paths:
        raise FileNotFoundError(f"В директории {data_dir} не найдено файлов HDF/HDF5!")
    
    # Переводим пути в строки для DataExtractor
    name_list = [str(p) for p in file_paths]
    print(f"Обнаружено файлов с данными: {len(name_list)}")

    # 2. ИЗВЛЕЧЕНИЕ ДАННЫХ
    data_extractor = DataExtractor(max_processes=15, time_interval=[0*s, 15*s])
    data = data_extractor.extract_data(name_list)

    # 3. АВТОМАТИЧЕСКАЯ СБОРКА И СОРТИРОВКА ПРОЕКЦИЙ
    new_data = []
    angles = []
    
    for volumes in data:
        for volume_name, volume_data in volumes.items():
            # Умный поиск угла в названии (например, "Detector at 180.0 deg")
            match = re.search(r'at ([\d\.]+) deg', volume_name)
            if match:
                angle = float(match.group(1))
            else:
                print(f"Предупреждение: Не удалось извлечь угол из {volume_name}. Пропуск.")
                continue
                
            angles.append(angle)
            new_data.append({'interactions_data': volume_data})
            
    # Сортируем все проекции по абсолютному углу поворота
    sort_indices = np.argsort(angles)
    data = [new_data[i] for i in sort_indices]
    sorted_angles = np.array(angles)[sort_indices]
    
    # Автоматическое определение количества проекций (views)
    views = len(data)
    print(f"Успешно собрано и отсортировано проекций: {views}")

    # 4. НАСТРОЙКИ КОНВЕРТЕРА
    data_converter = DataConverter(max_processes=40)
    data_converter.processing_parameters = {
        "decay_time": 300 * ns,
        "spatial_resolution": 4.0 * mm,
        "energy_resolution": 9.9,
        "reference_energy": 140.5 * keV,
        "energy_channels": 1024,
        "energy_range": [0, 300 * keV],
        "energy_windows": [
            [1.0, [143 * keV, 175 * keV]], # Основное окно (например, I-123 или Tc-99m)
        ],
        "image_range": None,
        "pixel_size": 2.5 * mm,
        "matrix": [128, 128],
        "use_distance_traveled": True,
        "voxel_size": 4.0 * mm,
        "emission_ROI": [[-600.0, 600.0 * cm], [-600.0, 600.0 * cm], [-400.0 * cm, 400.0 * cm]],
    }

    # 5. ГЕНЕРАЦИЯ И СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ (СИНОГРАММЫ)
    images = data_converter.convert_to_image(data)
    
    # Сдвигаем массив на половину от автоматически найденного числа проекций
    images = np.roll(images, views // 2, axis=0)
    images[images < 0] = 0

    pixel_size = data_converter.processing_parameters['pixel_size']
    data_saver = DataSaver(images, filename + "_MEW", pixel_size=pixel_size)
    
    # Создаем необходимые директории внутри DataSaver (если нужно) или убеждаемся, что они есть
    data_saver.save_as_numpy(rot=True)
    data_saver.save_as_dicom()
    data_saver.save_as_dat()
    
    print("Изображения успешно сохранены.")