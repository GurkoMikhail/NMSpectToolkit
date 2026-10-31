"""
Геометрическое ядро прямого и обратного томографического проецирования ОФЭКТ.

Содержит функции нормализации проекционных углов, построения круговой маски поля
зрения (FOV), 2D/3D гауссовой фильтрации объёмов, структуры предвычисленных
таблиц билинейной интерполяции и класс `TomographicProjector` с поддержкой
лучевого ослабления гамма-излучения и пространственного отклика детектора (PSF).

Все физические величины (размеры вокселей, углы поворота гантри, линейные
коэффициенты ослабления, ширина гауссианы FWHM) передаются и обрабатываются
строго в системе единиц `hepunits`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from hepunits import deg, mm, rad
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix


def normalize_projection_angles(
    angles: NDArray[np.float64] | Sequence[float] | None,
    num_views: int,
    start_angle: float = 0.0 * deg,
    orbit_span: float = 360.0 * deg,
    rotation_direction: str = "ccw",
) -> NDArray[np.float64]:
    """
    Приводит массив углов поворота детектора к стандарту `hepunits` (где `rad == 1.0`).

    Если массив углов не передан, генерируется равномерная сетка на дуге `orbit_span`,
    начиная со стартового ракурса `start_angle` в направлении `rotation_direction`
    (`'ccw'` — против часовой стрелки, `'cw'` — по часовой стрелке).
    Если переданы численные значения в градусах без множителя `deg`, выполняется
    автоматический перевод в систему `hepunits` умножением на константу `deg`.

    :param angles: Вектор углов проекций или `None`.
    :param num_views: Ожидаемое число угловых ракурсов.
    :param start_angle: Начальный ракурс в `hepunits` (используется при `angles is None`).
    :param orbit_span: Угловой диапазон орбиты в `hepunits` (по умолчанию `360 * deg`).
    :param rotation_direction: Направление вращения (`'ccw'` или `'cw'`).
    :return: Одномерный массив углов в единицах `hepunits` формы `(num_views,)`.
    """
    if int(num_views) <= 0:
        raise ValueError(
            f"Число проекционных ракурсов должно быть положительным, получено: {num_views}."
        )

    if angles is None:
        dir_norm = rotation_direction.strip().lower()
        if dir_norm in ("cw", "clockwise", "-1", "minus"):
            direction_sign = -1.0
        elif dir_norm in ("ccw", "counterclockwise", "+1", "1", "plus"):
            direction_sign = 1.0
        else:
            raise ValueError(
                f"Неизвестное направление вращения '{rotation_direction}', допустимы 'ccw' или 'cw'."
            )
        step = direction_sign * float(orbit_span) / float(num_views)
        grid = float(start_angle) + np.arange(num_views, dtype=np.float64) * step
        return np.mod(grid, 360.0 * deg)

    angle_array = np.asarray(angles, dtype=np.float64).ravel()
    if angle_array.size != num_views:
        raise ValueError(
            f"Размерность вектора углов ({angle_array.size}) не совпадает "
            f"с числом проекционных ракурсов ({num_views})."
        )

    if angle_array.size > 0:
        max_abs = float(np.max(np.abs(angle_array)))
        span = float(np.max(angle_array) - np.min(angle_array)) if angle_array.size > 1 else 0.0
        min_step = (
            float(np.min(np.abs(np.diff(angle_array)))) if angle_array.size > 1 else max_abs
        )
        # Углы в градусах без множителя deg имеют размах > 2*pi либо шаг >= 0.4 при модуле > 2*pi
        if span > (2.0 * np.pi + 1e-2) * rad or (
            max_abs > (2.0 * np.pi + 1e-2) * rad and min_step > 0.4 * rad
        ):
            angle_array = angle_array * deg

    return angle_array


def create_circular_fov_mask(
    transaxial_size: int,
    radius_fraction: float = 1.0,
) -> NDArray[np.bool_]:
    """
    Создаёт булеву маску кругового поля зрения (FOV) в трансаксиальной плоскости.

    :param transaxial_size: Число вокселей вдоль трансаксиальной оси (`nx`).
    :param radius_fraction: Относительный радиус круга относительно вписанной окружности.
    :return: Двумерный булев массив формы `(nx, nx)`.
    """
    if int(transaxial_size) <= 0:
        raise ValueError(
            f"Трансаксиальный размер сетки должен быть положительным, получено: {transaxial_size}."
        )
    center = 0.5 * (transaxial_size - 1)
    max_radius = max(center * float(radius_fraction), 0.5)
    coords = np.arange(transaxial_size, dtype=np.float64) - center
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")
    return (grid_x * grid_x + grid_y * grid_y) <= (max_radius * max_radius + 1e-6)


def gaussian_smooth_volume(
    volume: NDArray[np.float64],
    fwhm: float,
    voxel_size: float,
    axes: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """
    Выполняет сепарабельную самосопряжённую гауссову фильтрацию массива с шириной
    на полувысоте `fwhm` (в единицах `hepunits`).

    :param volume: Входной массив (например, 2D срез `(nx, nx)`, 3D объём `(nx, nx, ny)` или стек проекций).
    :param fwhm: Ширина гауссианы на полувысоте (FWHM) в системе `hepunits`.
    :param voxel_size: Физический размер вокселя в системе `hepunits`.
    :param axes: Опциональная последовательность осей, вдоль которых выполняется сглаживание (по умолчанию все оси).
    :return: Сглаженный массив той же формы.
    """
    fwhm_val = float(fwhm)
    if fwhm_val <= 0.0:
        return np.asarray(volume, dtype=np.float64).copy()

    step = float(voxel_size)
    if step <= 0.0:
        raise ValueError("Размер вокселя voxel_size должен быть строго положительным.")

    # Перевод FWHM в среднеквадратичное отклонение в единицах вокселей: sigma = FWHM / (2 * sqrt(2 * ln(2)))
    sigma_voxels = (fwhm_val / 2.3548200450309493) / step
    if sigma_voxels < 1e-3:
        return np.asarray(volume, dtype=np.float64).copy()

    radius = max(int(np.ceil(3.5 * sigma_voxels)), 1)
    truncate_factor = float(radius) / sigma_voxels

    result = np.asarray(volume, dtype=np.float64).copy()
    target_axes = tuple(range(result.ndim)) if axes is None else tuple(int(ax) for ax in axes)
    for axis_idx in target_axes:
        # Граничное условие 'reflect' в scipy.ndimage соответствует симметричному отражению ('symmetric' в np.pad),
        # гарантируя строгую самосопряжённость оператора фильтрации G^T = G
        result = gaussian_filter1d(
            result,
            sigma=sigma_voxels,
            axis=axis_idx,
            mode="reflect",
            truncate=truncate_factor,
        )
    return result


@dataclass
class _RotationLookup:
    """
    Предвычисленные индексы и веса билинейной интерполяции для поворота
    трансаксиальной сетки `(nx, nx)` на заданный угол.
    """

    x0: NDArray[np.intp]
    x1: NDArray[np.intp]
    y0: NDArray[np.intp]
    y1: NDArray[np.intp]
    w00: NDArray[np.float64]
    w01: NDArray[np.float64]
    w10: NDArray[np.float64]
    w11: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    forward_csr: csr_matrix | None = None
    transpose_csr: csr_matrix | None = None
    ray_proj_csr: csr_matrix | None = None
    ray_proj_transpose_csr: csr_matrix | None = None


@dataclass
class _BackprojectionLookup:
    """
    Предвычисленные индексы и веса 1D-интерполяции вдоль координаты детектора
    для обратного проецирования на трансаксиальную сетку `(nx, nx)`.
    """

    u0: NDArray[np.intp]
    u1: NDArray[np.intp]
    w0: NDArray[np.float64]
    w1: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]


_MAX_CSR_CACHE_SIZE: int = 1024
_ROTATION_CSR_CACHE: dict[tuple[int, float], _RotationLookup] = {}


class TomographicProjector:
    """
    Векторизованное геометрическое ядро прямого и обратного томографического
    проецирования для 2D срезов и 3D объёмов с поддержкой лучевой коррекции
    затухания гамма-излучения в среде и функции размытия точки (PSF).
    """

    @staticmethod
    def _build_forward_rotation_lookup(nx: int, angle: float) -> _RotationLookup:
        """
        Строит таблицу билинейной интерполяции для перевода из лабораторной
        системы координат `(x, y)` в систему лучей детектора под углом `angle`.
        """
        cache_key = (int(nx), round(float(angle), 9))
        cached = _ROTATION_CSR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        center = 0.5 * (nx - 1)
        coords = np.arange(nx, dtype=np.float64) - center
        grid_det, grid_ray = np.meshgrid(coords, coords, indexing="ij")

        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))

        # Лабораторные координаты точки для заданных лучевых координат детектора
        src_x = grid_det * cos_angle - grid_ray * sin_angle + center
        src_y = grid_det * sin_angle + grid_ray * cos_angle + center

        valid = (src_x >= 0.0) & (src_x <= nx - 1.0) & (src_y >= 0.0) & (src_y <= nx - 1.0)
        src_x_clipped = np.clip(src_x, 0.0, nx - 1.0)
        src_y_clipped = np.clip(src_y, 0.0, nx - 1.0)

        x0 = np.floor(src_x_clipped).astype(np.intp)
        y0 = np.floor(src_y_clipped).astype(np.intp)
        x1 = np.minimum(x0 + 1, nx - 1)
        y1 = np.minimum(y0 + 1, nx - 1)

        delta_x = src_x_clipped - x0
        delta_y = src_y_clipped - y0

        w00 = (1.0 - delta_x) * (1.0 - delta_y) * valid
        w01 = (1.0 - delta_x) * delta_y * valid
        w10 = delta_x * (1.0 - delta_y) * valid
        w11 = delta_x * delta_y * valid

        lookup = _RotationLookup(
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            w00=w00,
            w01=w01,
            w10=w10,
            w11=w11,
            valid_mask=valid,
        )
        if len(_ROTATION_CSR_CACHE) >= _MAX_CSR_CACHE_SIZE:
            oldest_key = next(iter(_ROTATION_CSR_CACHE))
            del _ROTATION_CSR_CACHE[oldest_key]
        _ROTATION_CSR_CACHE[cache_key] = lookup
        return lookup

    @staticmethod
    def _build_inverse_rotation_lookup(nx: int, angle: float) -> _RotationLookup:
        """
        Строит таблицу билинейной интерполяции для обратного перевода из системы
        лучей детектора в лабораторную систему координат.
        """
        center = 0.5 * (nx - 1)
        coords = np.arange(nx, dtype=np.float64) - center
        grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")

        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))

        # Лучевые координаты детектора для каждого лабораторного вокселя
        det_coord = grid_x * cos_angle + grid_y * sin_angle + center
        ray_coord = -grid_x * sin_angle + grid_y * cos_angle + center

        valid = (
            (det_coord >= 0.0)
            & (det_coord <= nx - 1.0)
            & (ray_coord >= 0.0)
            & (ray_coord <= nx - 1.0)
        )
        det_clipped = np.clip(det_coord, 0.0, nx - 1.0)
        ray_clipped = np.clip(ray_coord, 0.0, nx - 1.0)

        u0 = np.floor(det_clipped).astype(np.intp)
        ray0 = np.floor(ray_clipped).astype(np.intp)
        u1 = np.minimum(u0 + 1, nx - 1)
        ray1 = np.minimum(ray0 + 1, nx - 1)

        delta_det = det_clipped - u0
        delta_ray = ray_clipped - ray0

        w00 = (1.0 - delta_det) * (1.0 - delta_ray) * valid
        w01 = (1.0 - delta_det) * delta_ray * valid
        w10 = delta_det * (1.0 - delta_ray) * valid
        w11 = delta_det * delta_ray * valid

        return _RotationLookup(
            x0=u0,
            x1=u1,
            y0=ray0,
            y1=ray1,
            w00=w00,
            w01=w01,
            w10=w10,
            w11=w11,
            valid_mask=valid,
        )

    @staticmethod
    def _build_backprojection_1d_lookup(nx: int, angle: float) -> _BackprojectionLookup:
        """
        Строит таблицу 1D линейной интерполяции вдоль оси детектора
        для быстрого и чёткого обратного проецирования.
        """
        center = 0.5 * (nx - 1)
        coords = np.arange(nx, dtype=np.float64) - center
        grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")

        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))

        det_coord = grid_x * cos_angle + grid_y * sin_angle + center
        valid = (det_coord >= 0.0) & (det_coord <= nx - 1.0)
        det_clipped = np.clip(det_coord, 0.0, nx - 1.0)

        u0 = np.floor(det_clipped).astype(np.intp)
        u1 = np.minimum(u0 + 1, nx - 1)
        delta_det = det_clipped - u0

        w0 = (1.0 - delta_det) * valid
        w1 = delta_det * valid

        return _BackprojectionLookup(u0=u0, u1=u1, w0=w0, w1=w1, valid_mask=valid)

    @staticmethod
    def _ensure_rotation_csr(lookup: _RotationLookup, nx: int) -> tuple[csr_matrix, csr_matrix]:
        """Строит и кэширует пару сопряжённых CSR-матриц билинейного поворота `(R_theta, R_theta^T)`."""
        if lookup.forward_csr is not None and lookup.transpose_csr is not None:
            return lookup.forward_csr, lookup.transpose_csr

        num_pixels = nx * nx
        rows_rot = np.broadcast_to(
            np.arange(num_pixels, dtype=np.intp)[:, np.newaxis],
            (num_pixels, 4),
        ).ravel()
        cols_rot = np.column_stack(
            (
                (lookup.x0 * nx + lookup.y0).ravel(),
                (lookup.x0 * nx + lookup.y1).ravel(),
                (lookup.x1 * nx + lookup.y0).ravel(),
                (lookup.x1 * nx + lookup.y1).ravel(),
            )
        ).ravel()
        vals_rot = np.column_stack(
            (
                lookup.w00.ravel(),
                lookup.w01.ravel(),
                lookup.w10.ravel(),
                lookup.w11.ravel(),
            )
        ).ravel()

        fwd_csr = csr_matrix((vals_rot, (rows_rot, cols_rot)), shape=(num_pixels, num_pixels))
        tr_csr = fwd_csr.T.tocsr()
        lookup.forward_csr = fwd_csr
        lookup.transpose_csr = tr_csr
        return fwd_csr, tr_csr

    @staticmethod
    def build_ray_projector_csr_pair(
        forward_lookup: _RotationLookup,
        nx: int,
        ray_att_2d: NDArray[np.float64] | None = None,
    ) -> tuple[csr_matrix, csr_matrix]:
        """
        Строит пару строго сопряжённых разреженных CSR-матриц `(P_theta, P_theta^T)`
        для прямого и обратного лучевого проецирования ракурса.

        Матрица `P_theta` имеет форму `(nx, nx * nx)` и объединяет билинейный поворот
        `R_theta`, умножение на коэффициенты ослабления `ray_att_2d` и суммирование вдоль луча.
        """
        if (
            ray_att_2d is None
            and forward_lookup.ray_proj_csr is not None
            and forward_lookup.ray_proj_transpose_csr is not None
        ):
            return forward_lookup.ray_proj_csr, forward_lookup.ray_proj_transpose_csr

        num_pixels = nx * nx
        det_grid = np.broadcast_to(np.arange(nx, dtype=np.intp)[:, np.newaxis], (nx, nx)).ravel()
        rows_proj = np.broadcast_to(det_grid[:, np.newaxis], (num_pixels, 4)).ravel()
        cols_proj = np.column_stack(
            (
                (forward_lookup.x0 * nx + forward_lookup.y0).ravel(),
                (forward_lookup.x0 * nx + forward_lookup.y1).ravel(),
                (forward_lookup.x1 * nx + forward_lookup.y0).ravel(),
                (forward_lookup.x1 * nx + forward_lookup.y1).ravel(),
            )
        ).ravel()

        if ray_att_2d is not None:
            w00_att = (forward_lookup.w00 * ray_att_2d).ravel()
            w01_att = (forward_lookup.w01 * ray_att_2d).ravel()
            w10_att = (forward_lookup.w10 * ray_att_2d).ravel()
            w11_att = (forward_lookup.w11 * ray_att_2d).ravel()
        else:
            w00_att = forward_lookup.w00.ravel()
            w01_att = forward_lookup.w01.ravel()
            w10_att = forward_lookup.w10.ravel()
            w11_att = forward_lookup.w11.ravel()

        vals_proj = np.column_stack((w00_att, w01_att, w10_att, w11_att)).ravel()
        proj_csr = csr_matrix((vals_proj, (rows_proj, cols_proj)), shape=(nx, num_pixels))
        proj_csr_transpose = proj_csr.T.tocsr()
        if ray_att_2d is None:
            forward_lookup.ray_proj_csr = proj_csr
            forward_lookup.ray_proj_transpose_csr = proj_csr_transpose
        return proj_csr, proj_csr_transpose

    @staticmethod
    def _apply_rotation_lookup_3d(
        volume_3d: NDArray[np.float64],
        lookup: _RotationLookup,
    ) -> NDArray[np.float64]:
        """
        Применяет предвычисленную 2D таблицу билинейной интерполяции сразу ко всем
        аксиальным срезам 3D массива формы `(nx, nx, ny)`.
        """
        nx_dim, _, ny_dim = volume_3d.shape
        fwd_csr, _ = TomographicProjector._ensure_rotation_csr(lookup, nx_dim)
        flat_vol = volume_3d.reshape(nx_dim * nx_dim, ny_dim)
        return np.asarray(fwd_csr @ flat_vol, dtype=np.float64).reshape(nx_dim, nx_dim, ny_dim)

    @staticmethod
    def _apply_rotation_transpose_3d(
        rotated_volume_3d: NDArray[np.float64],
        lookup: _RotationLookup,
        nx: int,
    ) -> NDArray[np.float64]:
        """
        Применяет строго сопряжённый (транспонированный) оператор билинейного поворота
        `R_theta^T` для перевода из лучевых координат детектора обратно в лабораторную
        систему координат `(nx, nx, ny)`.
        """
        ny = int(rotated_volume_3d.shape[2])
        _, tr_csr = TomographicProjector._ensure_rotation_csr(lookup, nx)
        flat_rot = rotated_volume_3d.reshape(nx * nx, ny)
        return np.asarray(tr_csr @ flat_rot, dtype=np.float64).reshape(nx, nx, ny)

    @classmethod
    def _prepare_attenuation_3d(
        cls,
        attenuation_map: NDArray[np.float64] | None,
        nx: int,
        ny: int,
        collapse_uniform_z: bool = True,
    ) -> NDArray[np.float64] | None:
        """
        Приводит карту коэффициентов линейного ослабления к 3D форме, совместимой с `(nx, nx, ny)`.

        При `collapse_uniform_z=True` однородные вдоль аксиальной оси Z карты хранятся
        в компактной форме `(nx, nx, 1)` для многократной экономии памяти и ускорения
        интерполяции за счёт стандартного NumPy broadcasting.
        """
        if attenuation_map is None:
            return None
        att_arr = np.asarray(attenuation_map, dtype=np.float64)
        if att_arr.ndim == 2:
            if att_arr.shape != (nx, nx):
                raise ValueError(
                    f"Неверная форма 2D карты затухания {att_arr.shape}, ожидалось {(nx, nx)}."
                )
            if collapse_uniform_z:
                return att_arr[:, :, np.newaxis]
            return np.repeat(att_arr[:, :, np.newaxis], ny, axis=2)
        if att_arr.ndim == 3:
            if att_arr.shape != (nx, nx, ny):
                raise ValueError(
                    f"Неверная форма 3D карты затухания {att_arr.shape}, ожидалось {(nx, nx, ny)}."
                )
            if collapse_uniform_z and ny > 1 and bool(np.all(att_arr[:, :, :1] == att_arr)):
                return att_arr[:, :, :1]
            return att_arr
        raise ValueError(
            f"Карта затухания должна быть 2D или 3D массивом, получено ndim={att_arr.ndim}."
        )

    @classmethod
    def compute_ray_attenuation_factors(
        cls,
        attenuation_3d: NDArray[np.float64],
        forward_lookup: _RotationLookup,
        voxel_size: float,
    ) -> NDArray[np.float64]:
        """
        Вычисляет кумулятивные коэффициенты пропускания гамма-квантов
        `exp(-integral(mu(s) ds))` вдоль лучей детектора в повёрнутой системе координат.

        Детектор расположен со стороны положительного направления лучевой оси (индекс `nx - 1`).
        Интеграл берётся по правилу средних точек от центра вокселя испускания до детектора.

        :param attenuation_3d: 3D карта линейных коэффициентов ослабления `(nx, nx, ny)` в `hepunits`.
        :param forward_lookup: Таблица поворота из лабораторных координат в лучевые.
        :param voxel_size: Физический шаг вокселя в `hepunits`.
        :return: Массив коэффициентов прохождения лучей формы `(nx, nx, ny)`.
        """
        mu_rotated = cls._apply_rotation_lookup_3d(attenuation_3d, forward_lookup)
        mu_rotated = np.clip(mu_rotated, 0.0, None)
        # Кумулятивный интеграл от детектора (nx - 1) вглубь фантома (к индексу 0)
        reverse_cumsum = np.cumsum(mu_rotated[:, ::-1, :], axis=1)[:, ::-1, :]
        optical_depth = (reverse_cumsum - 0.5 * mu_rotated) * float(voxel_size)
        return np.exp(-np.clip(optical_depth, 0.0, None))

    @classmethod
    def compute_voxel_attenuation_weights_for_angle(
        cls,
        attenuation_3d: NDArray[np.float64],
        forward_lookup: _RotationLookup,
        inverse_lookup: _RotationLookup,
        voxel_size: float,
    ) -> NDArray[np.float64]:
        """
        Вычисляет коэффициенты прохождения излучения `exp(-integral(mu ds))`
        от каждого вокселя до детектора на заданном угле в лабораторной системе координат.
        """
        ray_factors = cls.compute_ray_attenuation_factors(
            attenuation_3d=attenuation_3d,
            forward_lookup=forward_lookup,
            voxel_size=voxel_size,
        )
        voxel_weights = cls._apply_rotation_lookup_3d(ray_factors, inverse_lookup)
        return np.clip(voxel_weights, 0.0, 1.0)

    @classmethod
    def forward_project(
        cls,
        volume: NDArray[np.float64],
        angles: NDArray[np.float64] | Sequence[float],
        attenuation_map: NDArray[np.float64] | None = None,
        voxel_size: float = 4.2 * mm,
        psf_fwhm: float | None = None,
        apply_fov_mask: bool = True,
    ) -> NDArray[np.float64]:
        """
        Выполняет прямое томографическое проецирование 2D среза `(nx, nx)` или
        3D объёма `(nx, nx, ny)` на заданный набор углов с учётом ослабления
        гамма-квантов в среде `attenuation_map` и опционального размытия PSF.

        Оператор прямого проецирования имеет вид:
        `A = H_PSF * S_ray * W_att * R_theta * M_FOV`,
        где размытие коллиматорно-детекторного отклика `H_PSF` применяется на плоскости
        детектора после прохождения излучения через ослабляющую среду.

        :param volume: Входное распределение активности `(nx, nx)` или `(nx, nx, ny)`.
        :param angles: Массив углов проецирования в `hepunits`.
        :param attenuation_map: Опциональная карта линейного ослабления в `hepunits`.
        :param voxel_size: Размер вокселя в `hepunits`.
        :param psf_fwhm: Опциональная ширина гауссианы отклика коллиматора-детектора (FWHM) в `hepunits`.
        :param apply_fov_mask: Ограничить ли входной объём вписанным кругом поля зрения (FOV).
        :return: Стек проекций формы `(views, nx)` для 2D или `(views, nx, ny)` для 3D.
        """
        vol_arr = np.asarray(volume, dtype=np.float64)
        if vol_arr.ndim not in (2, 3):
            raise ValueError(
                f"Ожидался 2D или 3D массив объёма, получена форма: {vol_arr.shape}"
            )

        is_2d = vol_arr.ndim == 2
        vol_3d = vol_arr[:, :, np.newaxis].copy() if is_2d else vol_arr.copy()
        nx, ny_trans, ny = vol_3d.shape
        if nx != ny_trans:
            raise ValueError(
                f"Трансаксиальные оси объёма должны быть одинаковыми (nx == ny), получено: {vol_arr.shape}"
            )

        if apply_fov_mask:
            fov_mask = create_circular_fov_mask(nx)
            vol_3d *= fov_mask[:, :, np.newaxis]

        angle_seq = np.asarray(angles, dtype=np.float64).ravel()
        num_views = int(angle_seq.size)
        norm_angles = normalize_projection_angles(angle_seq, num_views)
        att_3d = cls._prepare_attenuation_3d(attenuation_map, nx, ny)

        projections = np.zeros((num_views, nx, ny), dtype=np.float64)
        use_compact_csr = (att_3d is None or att_3d.shape[2] == 1) and ny >= 8
        flat_vol = vol_3d.reshape(nx * nx, ny) if use_compact_csr else None

        for view_idx, angle in enumerate(norm_angles):
            fwd_lookup = cls._build_forward_rotation_lookup(nx, float(angle))
            if use_compact_csr and flat_vol is not None:
                ray_att_2d = (
                    cls.compute_ray_attenuation_factors(att_3d, fwd_lookup, voxel_size)[:, :, 0]
                    if att_3d is not None
                    else None
                )
                proj_csr, _ = cls.build_ray_projector_csr_pair(fwd_lookup, nx, ray_att_2d)
                projections[view_idx] = np.asarray(proj_csr @ flat_vol, dtype=np.float64)
            else:
                rot_vol = cls._apply_rotation_lookup_3d(vol_3d, fwd_lookup)
                if att_3d is not None:
                    ray_att = cls.compute_ray_attenuation_factors(att_3d, fwd_lookup, voxel_size)
                    rot_vol = rot_vol * ray_att
                projections[view_idx] = np.sum(rot_vol, axis=1)

        if psf_fwhm is not None and float(psf_fwhm) > 0.0:
            smooth_axes = (1,) if is_2d else (1, 2)
            projections = gaussian_smooth_volume(
                volume=projections,
                fwhm=float(psf_fwhm),
                voxel_size=voxel_size,
                axes=smooth_axes,
            )

        return projections[:, :, 0] if is_2d else projections

    @classmethod
    def backproject(
        cls,
        projections: NDArray[np.float64],
        angles: NDArray[np.float64] | Sequence[float] | None = None,
        attenuation_map: NDArray[np.float64] | None = None,
        voxel_size: float = 4.2 * mm,
        apply_fov_mask: bool = True,
        psf_fwhm: float | None = None,
    ) -> NDArray[np.float64]:
        """
        Выполняет строго сопряжённое обратное томографическое проецирование стека проекций
        `(views, nx)` или `(views, nx, ny)` в лабораторную систему координат
        `(nx, nx)` или `(nx, nx, ny)` с опциональным учётом весов затухания и PSF.

        Оператор обратного проецирования является точным эрмитово-сопряжённым оператором
        к `forward_project`:
        `A^T = M_FOV * R_theta^T * W_att * S_ray^T * H_PSF^T`,
        удовлетворяющим тождеству `<A x, y> == <x, A^T y>`.

        :param projections: Входной стек проекций `(views, nx)` или `(views, nx, ny)`.
        :param angles: Углы поворота детектора в `hepunits`.
        :param attenuation_map: Опциональная карта линейного ослабления в `hepunits`.
        :param voxel_size: Размер вокселя в `hepunits`.
        :param apply_fov_mask: Ограничить ли результат вписанным кругом реконструкции.
        :param psf_fwhm: Опциональная ширина гауссианы отклика коллиматора-детектора (FWHM) в `hepunits`.
        :return: Обратно спроецированный 2D срез `(nx, nx)` или 3D объём `(nx, nx, ny)`.
        """
        proj_arr = np.asarray(projections, dtype=np.float64)
        if proj_arr.ndim not in (2, 3):
            raise ValueError(
                f"Ожидался 2D или 3D стек проекций, получена форма: {proj_arr.shape}"
            )

        is_2d = proj_arr.ndim == 2
        proj_3d = proj_arr[:, :, np.newaxis].copy() if is_2d else proj_arr.copy()
        num_views, nx, ny = proj_3d.shape
        norm_angles = normalize_projection_angles(angles, num_views)
        att_3d = cls._prepare_attenuation_3d(attenuation_map, nx, ny)

        if psf_fwhm is not None and float(psf_fwhm) > 0.0:
            smooth_axes = (1,) if is_2d else (1, 2)
            proj_3d = gaussian_smooth_volume(
                volume=proj_3d,
                fwhm=float(psf_fwhm),
                voxel_size=voxel_size,
                axes=smooth_axes,
            )

        if att_3d is None or att_3d.shape[2] == 1:
            flat_volume = np.zeros((nx * nx, ny), dtype=np.float64)
            for view_idx, angle in enumerate(norm_angles):
                fwd_lookup = cls._build_forward_rotation_lookup(nx, float(angle))
                ray_att_2d = (
                    cls.compute_ray_attenuation_factors(att_3d, fwd_lookup, voxel_size)[:, :, 0]
                    if att_3d is not None
                    else None
                )
                _, proj_csr_transpose = cls.build_ray_projector_csr_pair(fwd_lookup, nx, ray_att_2d)
                flat_volume += np.asarray(proj_csr_transpose @ proj_3d[view_idx], dtype=np.float64)
            volume_3d = flat_volume.reshape(nx, nx, ny)
        else:
            volume_3d = np.zeros((nx, nx, ny), dtype=np.float64)
            for view_idx, angle in enumerate(norm_angles):
                fwd_lookup = cls._build_forward_rotation_lookup(nx, float(angle))
                proj_view = proj_3d[view_idx]
                rot_backproj = np.broadcast_to(proj_view[:, np.newaxis, :], (nx, nx, ny))
                ray_att = cls.compute_ray_attenuation_factors(att_3d, fwd_lookup, voxel_size)
                rot_backproj = rot_backproj * ray_att
                volume_3d += cls._apply_rotation_transpose_3d(rot_backproj, fwd_lookup, nx)

        if apply_fov_mask:
            fov_mask = create_circular_fov_mask(nx)
            volume_3d *= fov_mask[:, :, np.newaxis]

        return volume_3d[:, :, 0] if is_2d else volume_3d
