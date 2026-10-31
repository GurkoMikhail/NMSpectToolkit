# Техническая документация NMSpectToolkit (`nmspect`)

Каталог `docs/` содержит инженерные спецификации форматов данных, физических параметров детектирующей системы и метаданных Монте-Карло симуляций, обрабатываемых пакетом `nmspect`.

---

## Оглавление спецификаций

1. **[Спецификация паспортных метаданных симуляции (`simulation_metadata_spec.md`)](simulation_metadata_spec.md)**
   * Иерархия групп `/metadata/{simulation,source,gantry,geometry,protocol,tasks}` во входных HDF5-файлах симулятора `NMSimToolkit`.
2. **[Спецификация параметров детектирующей системы (`detection_system_parameters_spec.md`)](detection_system_parameters_spec.md)**
   * Физические параметры сцинтилляционного кристалла, коллиматора, энергетического и пространственного разрешения.
3. **[Оптимизация хранения HDF5 (`hdf5_storage_optimization_spec.md`)](hdf5_storage_optimization_spec.md)**
   * Структура групп `/initial_states` и `/interactions` и соглашения о размерностях `hepunits`.
4. **[Обратная связь по интеграции метаданных (`metadata_feedback.md`)](metadata_feedback.md)**
   * Протокол сопряжения между симулятором `NMSimToolkit` и пакетом постобработки `NMSpectToolkit`.

---

## Архитектурные модули пакета `nmspect`

* [`nmspect/io/`](../nmspect/io/__init__.py) — чтение сырых актов взаимодействия из HDF5 (`SimulationDataExtractor`) и экспорт вспомогательных массивов (`legacy_formats.py`).
* [`nmspect/spectrometry/`](../nmspect/spectrometry/__init__.py) — конфигурации энергетических окон (`EnergyWindowConfig`), расчёт весовых коэффициентов вычитания рассеяния DEW и TEW (`scatter.py`), реестр пресетов радионуклидов $^{161}\text{Tb}$ и $^{99\text{m}}\text{Tc}$ (`presets.py`) и трёхуровневый анализатор спектров (`SpectrumProcessor`).
* [`nmspect/detector/`](../nmspect/detector/__init__.py) — агрегация микроскопических актов во вспышки, энергетическое и пространственное размытие (`ScintillationProcessor`) и 2D-гистограммирование проекций (`ProjectionBuilder`).
* [`nmspect/exporters/`](../nmspect/exporters/__init__.py) — экспорт в научный архив HDF5 NeXus (`NeXusExporter`), медицинский стандарт DICOM NM (`DicomNMExporter`) и 3D-сцену VTK ParaView (`VtkParaviewExporter`).
* [`nmspect/reconstruction/`](../nmspect/reconstruction/__init__.py) — прямое и обратное проецирование (`TomographicProjector`), моделирование ослабления фотонов и метод Чанга (`attenuation.py`), аналитическая реконструкция (`FBPReconstructor`) и итерационные алгоритмы (`OSEMReconstructor`, `MLEMReconstructor`).
* [`nmspect/pipelines/`](../nmspect/pipelines/__init__.py) — сквозной конвейер обработки проекционных исследований ОФЭКТ (`projection_pipeline.py`).
* [`nmspect/visualization/`](../nmspect/visualization/__init__.py) — интерактивный просмотрщик спектров на PyQt6 (`spectrum_viewer.py`), 3D-сцена VTK (`vtk_viewer.py`) и инспектор NeXus (`nexus_inspector.py`).
* [`nmspect/cli/`](../nmspect/cli/__init__.py) — точки входа командной строки (`nmspect-process`, `nmspect-reconstruct`, `nmspect-nexus`, `nmspect-view-vtk`, `nmspect-view-spectrum`).
