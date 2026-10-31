# Спецификация параметров системы детекции и электронной постобработки (`nmspect.detector`, `nmspect.spectrometry`)

## 1. Концепция и границы ответственности
Данная спецификация описывает параметры, алгоритмы и математические модели **системы электронной детекции, сцинтилляционного отклика и спектрометрической обработки актов взаимодействия**, реализованные в подпакетах `nmspect.detector` (`ScintillationProcessor`, `ProjectionBuilder`) и `nmspect.spectrometry` (`EnergyWindowConfig`, `SpectrumProcessor`).

### Граница ответственности системы детекции:
* **Входные данные:** Сырые микроскопические акты взаимодействия Монте-Карло (размерность $N$: `global_position`, `energy_deposit`, `distance_traveled`, `emission_time`, `particle_ID`), извлекаемые классом `SimulationDataExtractor` и описанные в [`simulation_metadata_spec.md`](simulation_metadata_spec.md).
* **Назначение:** Моделирование физических процессов преобразования ионизационных потерь энергии в сцинтилляторе $\text{NaI(Tl)}$ в аппаратно регистрируемый отклик электронной системы (фотоумножители, АЦП, схемы позиционирования Ангера, временная агрегация в макроскопические вспышки $M \le N$, многооконная энергетическая селекция DEW/TEW и биннинг планарных проекций).
* **Ключевое свойство:** Все параметры данного набора **не зафиксированы в симуляции**, а настраиваются и варьируются исследователем в `ScintillationProcessor`, `ProjectionBuilder` и `SpectrumProcessor` (или через конвейер `run_isotope_nema_pipeline`) для тестирования различных аппаратных протоколов и алгоритмов реконструкции.

---

## 2. Реестр варьируемых параметров системы детекции

Все физические величины задаются строго в базовых единицах библиотеки `hepunits` (`mm`, `ns`, `keV`, `MeV`):

```python
from hepunits import keV, mm, ns
from nmspect.detector import ProjectionBuilder, ScintillationProcessor
from nmspect.spectrometry import SpectrumProcessor, get_energy_window_preset

scintillation_processor = ScintillationProcessor(
    time_resolution=230.0 * ns,
    energy_resolution=0.095,              # относительное разрешение FWHM на опорной энергии (9.5%)
    intrinsic_spatial_resolution=3.8 * mm,
    reference_energy=74.6 * keV,
    rng=42,
)

energy_windows = get_energy_window_preset("peak75_tew", isotope="161Tb")

projection_builder = ProjectionBuilder(
    pixel_size=4.2 * mm,
    energy_windows=energy_windows,
    detector_size=(540.0 * mm, 400.0 * mm, 9.5 * mm),
    clip_negative=True,
)

spectrum_processor = SpectrumProcessor(
    energy_range=(0.0 * keV, 300.0 * keV),
    num_bins=1024,
)
```

---

## 3. Математические модели физических эффектов детектора

### 3.1. Временное слияние микроскопических актов во вспышки (`ScintillationProcessor.unite_acts`)
В кристалле сцинтиллятора один первичный гамма-квант может испытать несколько последовательных микроактов (комптоновское рассеяние + последующий фотоэффект).

1. **Время регистрации акта взаимодействия:**
   $$t_{\text{act}} = t_{\text{emit}} + \frac{d_{\text{travel}}}{c}$$
   где $d_{\text{travel}}$ — `distance_traveled`, $c$ — скорость света в вакууме (`c_light`).
2. **Временное окно высвечивания (`time_resolution`):**
   * Все акты с одинаковым `particle_ID`, интервал времени между которыми не превышает `time_resolution` (для $\text{NaI(Tl)}$ $\tau \approx 230\text{ нс}$), векторно агрегируются (`np.add.reduceat`) в единую макроскопическую сцинтилляционную вспышку детектора (размерность $M \le N$).
   * Суммарная выделенная энергия вспышки:
     $$E_{\text{sum}} = \sum_{j \in \text{event}} \Delta E_j$$
   * Координаты центра тяжести светового пятна вспышки (средневзвешенные по энерговыделению):
     $$\vec{r}_{\text{event}} = \frac{\sum_{j} \Delta E_j \cdot \vec{r}_j}{\sum_j \Delta E_j}$$
   * Кратность предварительного рассеяния вспышки в фантоме (`united_scattering_orders`) определяется через `np.maximum.reduceat`.

### 3.2. Энергетическое разрешение детектора (`ScintillationProcessor.apply_energy_resolution`)
Флуктуации числа сцинтилляционных фотонов и фотоэлектронов на фотокатодах ФЭУ подчиняются статистике Пуассона. Относительное разрешение кристалла масштабируется обратно пропорционально квадратному корню из энергии:

$$\frac{\text{FWHM}(E)}{E} = R_{\text{ref}} \cdot \sqrt{\frac{E_{\text{ref}}}{E}}$$

Среднеквадратическое отклонение гауссова размытия:
$$\sigma_E = \frac{\text{FWHM}(E)}{2\sqrt{2\ln 2}} \approx \frac{\text{FWHM}(E)}{2.3548}$$

Зарегистрированная энергия вспышки:
$$E_{\text{reg}} \sim \mathcal{N}(E_{\text{sum}}, \sigma_E^2)$$
Именно массив $E_{\text{reg}}$ размерности $M$ формирует аппаратный энергетический спектр детектора.

### 3.3. Пространственное разрешение электроники (`ScintillationProcessor.apply_spatial_resolution`)
Оптическое светораспределение в кристалле и схема позиционирования логики Ангера вносят аппаратную погрешность определения локальных координат $(x, y)$ вспышки:

$$\sigma_{\text{spatial}} = \frac{\text{FWHM}_{\text{spatial}}}{2\sqrt{2\ln 2}}$$
$$\vec{r}_{\text{reg}} \sim \mathcal{N}(\vec{r}_{\text{event}}, \sigma_{\text{spatial}}^2)$$

### 3.4. Дискретизация проекций и коррекция рассеяния (`ProjectionBuilder.build_projections`)
Формирование 2D-проекционной матрицы пикселей с учетом многооконной селекции и весового вычитания комптоновского рассеяния (DEW / TEW):

$$I(u, v) = \text{clip}_{[0, +\infty)}\!\left( \text{Hist2D}\Big( \vec{r}_{\text{reg}} \;\Big|\; E_{\text{reg}} \in [E_{\text{low}}, E_{\text{high}}] \Big) + \sum_{k} w_k \cdot \text{Hist2D}\Big( \vec{r}_{\text{reg}} \;\Big|\; E_{\text{reg}} \in [E_{k,\text{low}}, E_{k,\text{high}}] \Big) \right)$$

где:
* $w_k$ — отрицательный весовой коэффициент `scatter_weight` связанного окна рассеяния из `EnergyWindowConfig.associated_windows` (рассчитывается функциями `calculate_dew_coefficient` и `calculate_tew_coefficients`);
* $[E_{\text{low}}, E_{\text{high}}]$ — границы `lower_bound` и `upper_bound` энергетического окна;
* $\text{Hist2D}$ — двумерное гистограммирование `np.histogram2d` зарегистрированных локальных координат на рабочем поле кристалла `detector_size` с шагом `pixel_size`.

---

## 4. Спецификация выходных структур системы детекции

После выполнения `ScintillationProcessor.process_detector_acts`, `ProjectionBuilder.build_projections` и `SpectrumProcessor.analyze` формируются согласованные структуры данных:

### 4.1. Матрица зарегистрированных сцинтилляционных вспышек (Размерность $M \times 5$)
* Столбец `0` (`particle_id`): идентификатор первичного кванта.
* Столбец `1` (`time`): средневзвешенное время регистрации вспышки в `hepunits` (`ns`).
* Столбец `2` (`energy`): аппаратно размытая зарегистрированная энергия вспышки в `hepunits` (`MeV`).
* Столбец `3` (`local_x`): латеральная координата вспышки на кристалле с аппаратным размытием в `hepunits` (`mm`).
* Столбец `4` (`local_y`): аксиальная координата вспышки на кристалле с аппаратным размытием в `hepunits` (`mm`).
* Сопутствующий вектор `scattering_orders` (`int32 [M]`): кратность комптоновского рассеяния фотона в фантоме до попадания в детектор (`0` — первичный нерассеянный квант, $\ge 1$ — рассеянный).

### 4.2. Трёхуровневый спектр и 6 метрик качества (`SpectrumResult`)
* Спектральные векторы: `energy_bins`, `bin_edges`, `total_spectrum`, `primary_spectrum`, `scatter_spectrum`, `compton_continuum`.
* Диагностические метрики: `primary_fraction`, `scatter_to_primary_ratio`, `compton_to_primary_ratio`, `photopeak_efficiency`, `mean_scatter_order`, `window_purity`.

### 4.3. 3D Стек планарных проекций
* `projections` (`float64 [views, num_bins_x, num_bins_y]`): стек планарных проекций ОФЭКТ для экспорта в **NeXus HDF5** (`NeXusExporter`), **DICOM NM** (`DicomNMExporter`), **VTK ParaView** (`VtkParaviewExporter`) и 3D томографической реконструкции (`FBPReconstructor`, `OSEMReconstructor`, `MLEMReconstructor`).
