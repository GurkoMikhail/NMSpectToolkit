# NMSpectToolkit (Nuclear Medicine SPECT Processing & Reconstruction Toolkit)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

**NMSpectToolkit** (пакет `nmspect`) — программный комплекс для физической постобработки данных Монте-Карло симуляций систем однофотонной эмиссионной компьютерной томографии (**ОФЭКТ / SPECT**), трёхуровневой спектрометрии, коррекции комптоновского рассеяния (DEW / TEW), экспорта в международные научные/медицинские стандарты и 3D томографической реконструкции.

---

## Ключевые возможности

* **Физическая модель сцинтилляционного детектора (`nmspect.detector`):**
  * Строгое разделение микроскопических актов взаимодействия в кристалле ($N$) и зарегистрированных сцинтилляционных вспышек ($M \le N$).
  * Векторизованная временная агрегация актов во вспышки (`unite_acts`), расчёт средневзвешенного центра тяжести светового пятна, моделирование энергетического разрешения ФЭУ ($\propto 1/\sqrt{E}$) и внутреннего пространственного разрешения кристалла (FWHM).
* **Трёхуровневая спектрометрия и коррекция рассеяния (`nmspect.spectrometry`):**
  * Декомпозиция энергетического спектра на полный (`total`), первичный нерассеянный (`primary`) и рассеянный в фантоме (`scatter`) компоненты, а также оценка комптоновского континуума.
  * Автоматический расчёт 6 диагностических метрик качества энергетического окна (`primary_fraction`, `scatter_to_primary_ratio`, `compton_to_primary_ratio`, `photopeak_efficiency`, `mean_scatter_order`, `window_purity`).
  * Единый реестр пресетов энергетических окон для $^{161}\text{Tb}$ (фотопики $48.9\text{ кэВ}$ и $74.6\text{ кэВ}$) и $^{99\text{m}}\text{Tc}$ ($140.5\text{ кэВ}$) с поддержкой методов двойного (DEW, Jaszczak) и тройного (TEW, Ogawa) энергетических окон.
* **Экспорт в международные стандарты (`nmspect.exporters`):**
  * **HDF5 NeXus (`.nxs`):** мастер-архив стандарта `NXtomo` с проекциями, спектрами, 6 метриками качества и полным деревом паспортных метаданных симуляции.
  * **DICOM Nuclear Medicine (`.dcm`):** многокадровый клинический стандарт `TOMO` с заполненными последовательностями `EnergyWindowInformationSequence`, `DetectorInformationSequence` и `RotationInformationSequence` (совместим с 3D Slicer, Horos, OsiriX).
  * **VTK ParaView (`.vtm` + `.vti`):** мультиблочная 3D-сборка геометрии гантри, коллиматоров, корпусов детекторов, фантома и текстурированных проекций в мировых координатах.
* **3D Томографическая реконструкция (`nmspect.reconstruction`):**
  * **Аналитическая реконструкция (`FBPReconstructor`):** метод фильтрованных обратных проекций с фильтрами Рама-Лак (`ramp`), Шеппа-Логана (`shepp-logan`), Косинусным, Хэмминга, Ханна и Баттерворта + итерационная коррекция затухания по методу Чанга (`Chang attenuation correction`).
  * **Итерационная статистическая реконструкция (`OSEMReconstructor`, `MLEMReconstructor`):** максимизация правдоподобия по упорядоченным подмножествам с матричным учётом неоднородных 3D-карт затухания $\mu(x, y, z)$, моделированием функции размытия точки детектора (PSF) и байесовской регуляризацией OSL (Total Variation / квадратичный штраф).

---

## Установка

### Через `uv` (рекомендуется)
```bash
uv sync --extra dev
```

### Стандартный `pip` и виртуальное окружение `.venv`
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

---

## Командный интерфейс (CLI)

После установки пакета в окружении доступны 5 CLI-команд:

### 1. Полный конвейер постобработки симуляций (`nmspect-process`)
```bash
# Просмотр всех доступных пресетов энергетических окон для 161Tb и 99mTc:
nmspect-process --list-presets

# Обработка всех пресетов для 161Tb (по умолчанию из data/raw/output_tb161_dual_cam):
nmspect-process --isotope 161Tb

# Обработка конкретных пресетов для 99mTc:
nmspect-process --isotope 99mTc --preset peak140 --preset peak140_tew
```

### 2. 3D Томографическая реконструкция (`nmspect-reconstruct`)
```bash
# Итерационная реконструкция OSEM с коррекцией затухания по автоконтуру и TV-регуляризацией:
nmspect-reconstruct results/nexus/tb161_nema_120views_peak75_tew.nxs --method osem --iterations 8 --subsets 8 --attenuation-mode auto-contour --prior tv

# Аналитическая реконструкция FBP с фильтром Шеппа-Логана:
nmspect-reconstruct results/dicom/tc99m_nema_120views_peak140.dcm --method fbp --filter shepp-logan
```

### 3. Инспекция и графики файлов NeXus (`nmspect-nexus`)
```bash
nmspect-nexus results/nexus/tb161_nema_120views_peak75.nxs --tree
nmspect-nexus results/nexus/tb161_nema_120views_peak75.nxs --plot-spectrum results/spectrum.png --plot-projections results/projections.png
```

### 4. Интерактивные визуализаторы (`nmspect-view-vtk`, `nmspect-view-spectrum`)
```bash
nmspect-view-vtk results/paraview/tb161_nema_120views_peak75.vtm
nmspect-view-spectrum results/nexus/tb161_nema_120views_peak75.nxs
```

---

## Структура репозитория

```text
NMSpectToolkit/
├── nmspect/                 # Корневой вычислительный пакет
│   ├── io/                  # Извлечение сырых данных HDF5 (SimulationDataExtractor) и конвертация форматов
│   ├── spectrometry/        # Модели окон (EnergyWindowConfig), коэффициенты DEW/TEW, пресеты 161Tb/99mTc, SpectrumProcessor
│   ├── detector/            # Физика сцинтиллятора (ScintillationProcessor) и проекционный биннинг (ProjectionBuilder)
│   ├── exporters/           # Экспортеры в стандарты NeXus (.nxs), DICOM NM (.dcm) и VTK ParaView (.vtm/.vti)
│   ├── reconstruction/      # 3D реконструкция: TomographicProjector, карты затухания, FBP, OSEM, MLEM
│   ├── pipelines/           # Единый параметризованный конвейер постобработки исследований ОФЭКТ
│   ├── visualization/       # Графический анализатор спектров (PyQt6), 3D VTK вьювер и инспектор NeXus
│   └── cli/                 # Точки входа командной строки (nmspect-*)
├── data/                    # Входные данные (игнорируются Git)
│   ├── raw/                 # HDF5-файлы симуляций Монте-Карло (output_tb161_dual_cam, output_tc99m_dual_cam)
│   └── phantoms/            # Цифровые фантомы активности и ослабления (dat, dicom, numpy)
├── results/                 # Выходные артефакты (nexus, dicom, paraview, reconstruction, numpy, spr, png)
├── docs/                    # Техническая документация и спецификации метаданных
├── tests/                   # Модульные и интеграционные тесты по доменным подпакетам
└── pyproject.toml           # Конфигурация сборки, зависимостей, CLI, pytest, ruff и mypy
```

---

## Тестирование и контроль качества кода

```bash
.venv\Scripts\pytest.exe -v
.venv\Scripts\ruff.exe check nmspect tests
.venv\Scripts\mypy.exe nmspect tests
```
