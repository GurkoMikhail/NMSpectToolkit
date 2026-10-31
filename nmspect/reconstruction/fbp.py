"""
Аналитический алгоритм томографической реконструкции ОФЭКТ методом фильтрованной
обратной проекции (Filtered Backprojection, FBP).

Поддерживает частотную аподизацию (окна Рама-Лак / ramp, Шеппа-Логана, Ханна,
Хэмминга, косинусное и Баттерворта), коррекцию затухания первого и высших порядков
по методу Чанга и 2D/3D гауссову постобработку.

Все физические величины передаются и обрабатываются строго в системе единиц `hepunits`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from hepunits import mm
from numpy.typing import NDArray

from nmspect.reconstruction.attenuation import (
    apply_chang_attenuation_correction,
    create_uniform_attenuation_map,
    estimate_attenuation_map_from_projections,
)
from nmspect.reconstruction.iterative import ReconstructionResult
from nmspect.reconstruction.projector import (
    TomographicProjector,
    create_circular_fov_mask,
    gaussian_smooth_volume,
    normalize_projection_angles,
)


class FBPReconstructor:
    """
    Реконструктор методом фильтрованной обратной проекции (Filtered Backprojection, FBP)
    с поддержкой частотных окон аподизации и коррекции затухания по методу Чанга.
    """

    SUPPORTED_FILTERS: tuple[str, ...] = (
        "ramp",
        "shepp-logan",
        "cosine",
        "hamming",
        "hann",
        "butterworth",
    )

    def __init__(
        self,
        filter_name: str = "ramp",
        cutoff_frequency: float = 1.0,
        butterworth_order: int = 4,
        voxel_size: float = 4.2 * mm,
        attenuation_map: NDArray[np.float64] | None = None,
        linear_attenuation: float | None = None,
        phantom_radius: float | None = None,
        chang_iterations: int = 1,
        post_smoothing_fwhm: float | None = None,
        clip_negative: bool = True,
        apply_fov_mask: bool = True,
    ) -> None:
        """
        Инициализирует реконструктор FBP.

        :param filter_name: Тип окна фильтрации (`'ramp'`, `'hann'`, `'hamming'`, `'shepp-logan'`, `'butterworth'`, `'cosine'`).
        :param cutoff_frequency: Относительная частота среза в долях частоты Найквиста (`0 < cutoff <= 1.0`).
        :param butterworth_order: Порядок фильтра Баттерворта (используется при `filter_name='butterworth'`).
        :param voxel_size: Физический размер пикселя/вокселя в единицах `hepunits`.
        :param attenuation_map: 2D или 3D карта линейных коэффициентов ослабления в `hepunits`.
        :param linear_attenuation: Однородный линейный коэффициент ослабления среды в `hepunits`.
        :param phantom_radius: Физический радиус цилиндрического фантома в `hepunits` (для однородной карты).
        :param chang_iterations: Число итераций метода Чанга (`1` — классический метод первого порядка).
        :param post_smoothing_fwhm: FWHM 3D/2D гауссова сглаживания после реконструкции в `hepunits`.
        :param clip_negative: Обнулять ли отрицательные значения в итоговом объёме.
        :param apply_fov_mask: Ограничивать ли область реконструкции круговым полем зрения.
        """
        normalized_filter = filter_name.strip().lower()
        if normalized_filter == "ram-lak":
            normalized_filter = "ramp"
        if normalized_filter == "hanning":
            normalized_filter = "hann"
        if normalized_filter not in self.SUPPORTED_FILTERS:
            raise ValueError(
                f"Неподдерживаемый фильтр '{filter_name}'. "
                f"Допустимые значения: {self.SUPPORTED_FILTERS}"
            )
        if not (0.0 < float(cutoff_frequency) <= 1.0):
            raise ValueError(
                "Частота среза cutoff_frequency должна лежать в интервале (0.0, 1.0]."
            )
        if float(voxel_size) <= 0.0:
            raise ValueError("Размер вокселя voxel_size должен быть строго положительным.")

        self.filter_name: str = normalized_filter
        self.cutoff_frequency: float = float(cutoff_frequency)
        self.butterworth_order: int = max(int(butterworth_order), 1)
        self.voxel_size: float = float(voxel_size)
        self.attenuation_map: NDArray[np.float64] | None = attenuation_map
        self.linear_attenuation: float | None = (
            float(linear_attenuation) if linear_attenuation is not None else None
        )
        self.phantom_radius: float | None = (
            float(phantom_radius) if phantom_radius is not None else None
        )
        self.chang_iterations: int = max(int(chang_iterations), 1)
        self.post_smoothing_fwhm: float | None = (
            float(post_smoothing_fwhm) if post_smoothing_fwhm is not None else None
        )
        self.clip_negative: bool = bool(clip_negative)
        self.apply_fov_mask: bool = bool(apply_fov_mask)

    def build_frequency_filter(self, padded_size: int) -> NDArray[np.float64]:
        """
        Формирует одномерную передаточную характеристику фильтра `H(f) = |f| * W(f)`
        в области пространственных частот БПФ.

        :param padded_size: Длина БПФ с учётом дополнения нулями (степень двойки).
        :return: Вещественный массив коэффициентов фильтра длины `padded_size`.
        """
        # Нормированная частота относительно частоты Найквиста: [0, 1]
        freqs = 2.0 * np.abs(np.fft.fftfreq(padded_size))
        cutoff = self.cutoff_frequency
        active = freqs <= cutoff

        window = np.zeros_like(freqs, dtype=np.float64)
        if self.filter_name == "ramp":
            window[active] = 1.0
        elif self.filter_name == "shepp-logan":
            arg = 0.5 * np.pi * freqs[active] / cutoff
            window[active] = np.where(
                arg == 0.0, 1.0, np.sin(arg) / np.where(arg == 0.0, 1.0, arg)
            )
        elif self.filter_name == "cosine":
            window[active] = np.cos(0.5 * np.pi * freqs[active] / cutoff)
        elif self.filter_name == "hamming":
            window[active] = 0.54 + 0.46 * np.cos(np.pi * freqs[active] / cutoff)
        elif self.filter_name == "hann":
            window[active] = 0.5 * (1.0 + np.cos(np.pi * freqs[active] / cutoff))
        elif self.filter_name == "butterworth":
            # Фильтр Баттерворта имеет плавный спад с частотой полуспада cutoff
            window = 1.0 / np.sqrt(1.0 + (freqs / cutoff) ** (2 * self.butterworth_order))

        return freqs * window

    def filter_projections(self, projections: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Выполняет фильтрацию стека проекций в частотной области вдоль трансаксиальной
        координаты детектора (ось 1).

        :param projections: Массив проекций формы `(views, nx)` или `(views, nx, ny)`.
        :return: Отфильтрованные проекции той же формы.
        """
        proj_arr = np.asarray(projections, dtype=np.float64)
        is_2d = proj_arr.ndim == 2
        proj_3d = proj_arr[:, :, np.newaxis] if is_2d else proj_arr
        num_views, nx, ny = proj_3d.shape

        # Дополнение нулями до ближайшей степени двойки >= 2 * nx для устранения циклической свёртки
        padded_size = max(64, int(2 ** np.ceil(np.log2(2.0 * nx))))
        freq_filter = self.build_frequency_filter(padded_size)

        padded = np.zeros((num_views, padded_size, ny), dtype=np.float64)
        padded[:, :nx, :] = proj_3d

        spectrum = np.fft.fft(padded, axis=1)
        filtered_padded = np.fft.ifft(
            spectrum * freq_filter[np.newaxis, :, np.newaxis], axis=1
        ).real
        filtered = filtered_padded[:, :nx, :]

        return filtered[:, :, 0] if is_2d else filtered

    def _resolve_attenuation_map(
        self,
        projections: NDArray[np.float64],
        angles: NDArray[np.float64],
        override_attenuation_map: NDArray[np.float64] | None,
        active_voxel_size: float,
    ) -> NDArray[np.float64] | None:
        """
        Определяет актуальную карту ослабления (явно переданную, сохранённую в экземпляре
        или автоматически построенную по `linear_attenuation`).
        """
        if override_attenuation_map is not None:
            return np.asarray(override_attenuation_map, dtype=np.float64)
        if self.attenuation_map is not None:
            return np.asarray(self.attenuation_map, dtype=np.float64)
        if self.linear_attenuation is not None and self.linear_attenuation > 0.0:
            nx = int(projections.shape[1])
            out_shape: tuple[int, ...] = (
                (nx, nx) if projections.ndim == 2 else (nx, nx, int(projections.shape[2]))
            )
            if self.phantom_radius is not None:
                return create_uniform_attenuation_map(
                    shape=out_shape,
                    linear_attenuation=self.linear_attenuation,
                    voxel_size=active_voxel_size,
                    phantom_radius=self.phantom_radius,
                )
            return estimate_attenuation_map_from_projections(
                projections=projections,
                angles=angles,
                linear_attenuation=self.linear_attenuation,
                voxel_size=active_voxel_size,
            )
        return None

    @staticmethod
    def _compute_backprojection_scale(norm_angles: NDArray[np.float64]) -> float:
        """
        Вычисляет нормировочный множитель квадратуры обратного преобразования Радона
        с учётом фактического углового шага `angular_step` и кратности углового обзора
        (`0.25 * angular_step` при полном двухстороннем обзоре `2*pi` и `0.5 * angular_step`
        при односторонней полуорбите `pi`).
        """
        num_views = int(norm_angles.size)
        if num_views <= 1:
            return float(np.pi / 2.0)

        sorted_angles = np.sort(np.mod(norm_angles, 2.0 * np.pi))
        diffs = np.diff(sorted_angles)
        positive_diffs = diffs[diffs > 1e-6]
        if positive_diffs.size == 0:
            return float(np.pi / (2.0 * float(num_views)))

        angular_step = float(np.median(positive_diffs))
        total_span = float(sorted_angles[-1] - sorted_angles[0]) + angular_step
        # При охвате > pi каждый луч измеряется с двух противоположных направлений (множитель 1/2 в интеграле Радона)
        if total_span > np.pi + 0.5 * angular_step:
            return 0.25 * angular_step
        return 0.5 * angular_step

    def reconstruct_with_metadata(
        self,
        projections: NDArray[np.float64],
        angles: NDArray[np.float64] | Sequence[float] | None = None,
        attenuation_map: NDArray[np.float64] | None = None,
        voxel_size: float | None = None,
    ) -> ReconstructionResult:
        """
        Выполняет полную реконструкцию FBP (с опциональной коррекцией затухания по Чангу)
        и возвращает структурированный объект `ReconstructionResult`.

        :param projections: Входные проекции `(views, nx)` или `(views, nx, ny)`.
        :param angles: Углы проекций в `hepunits`.
        :param attenuation_map: Опциональная карта линейного ослабления в `hepunits`.
        :param voxel_size: Опциональное переопределение физического размера вокселя в `hepunits`.
        :return: Объект `ReconstructionResult`.
        """
        proj_arr = np.asarray(projections, dtype=np.float64)
        if proj_arr.ndim not in (2, 3):
            raise ValueError(
                f"Ожидался 2D или 3D массив проекций, получена форма: {proj_arr.shape}"
            )

        num_views = int(proj_arr.shape[0])
        active_voxel_size = float(voxel_size) if voxel_size is not None else self.voxel_size
        norm_angles = normalize_projection_angles(angles, num_views)
        bp_scale = self._compute_backprojection_scale(norm_angles)

        # 1. Частотная фильтрация проекций и классическая обратная проекция
        filtered_proj = self.filter_projections(proj_arr)
        raw_backproj = TomographicProjector.backproject(
            projections=filtered_proj,
            angles=norm_angles,
            attenuation_map=None,
            voxel_size=active_voxel_size,
            apply_fov_mask=self.apply_fov_mask,
        )
        volume = raw_backproj * bp_scale

        # 2. Коррекция затухания по методу Чанга (1-го или итеративного порядка)
        resolved_att = self._resolve_attenuation_map(
            projections=proj_arr,
            angles=norm_angles,
            override_attenuation_map=attenuation_map,
            active_voxel_size=active_voxel_size,
        )
        correction_map: NDArray[np.float64] | None = None
        if resolved_att is not None:
            volume, correction_map = apply_chang_attenuation_correction(
                reconstructed_volume=volume,
                attenuation_map=resolved_att,
                angles=norm_angles,
                voxel_size=active_voxel_size,
            )
            # При chang_iterations > 1 выполняется итеративное уточнение Чанга по невязке проекций
            for _ in range(1, self.chang_iterations):
                current_est = np.clip(volume, 0.0, None) if self.clip_negative else volume
                reprojected = TomographicProjector.forward_project(
                    volume=current_est,
                    angles=norm_angles,
                    attenuation_map=resolved_att,
                    voxel_size=active_voxel_size,
                )
                residual_proj = proj_arr - reprojected
                filtered_res = self.filter_projections(residual_proj)
                res_backproj = TomographicProjector.backproject(
                    projections=filtered_res,
                    angles=norm_angles,
                    attenuation_map=None,
                    voxel_size=active_voxel_size,
                    apply_fov_mask=self.apply_fov_mask,
                ) * bp_scale
                safe_div = np.maximum(correction_map, 1e-3)
                delta = np.where(correction_map > 1e-3, res_backproj / safe_div, 0.0)
                volume = volume + delta

        if self.clip_negative:
            volume = np.clip(volume, 0.0, None)

        if self.post_smoothing_fwhm is not None and self.post_smoothing_fwhm > 0.0:
            volume = gaussian_smooth_volume(
                volume=volume,
                fwhm=self.post_smoothing_fwhm,
                voxel_size=active_voxel_size,
            )

        if self.apply_fov_mask:
            fov_mask = create_circular_fov_mask(int(proj_arr.shape[1]))
            if volume.ndim == 2:
                volume = volume * fov_mask
            else:
                volume = volume * fov_mask[:, :, np.newaxis]

        return ReconstructionResult(
            volume=volume,
            voxel_size=active_voxel_size,
            method="FBP",
            attenuation_corrected=resolved_att is not None,
            correction_map=correction_map,
            metadata={
                "filter_name": self.filter_name,
                "cutoff_frequency": self.cutoff_frequency,
                "chang_iterations": self.chang_iterations if resolved_att is not None else 0,
            },
        )

    def reconstruct(
        self,
        projections: NDArray[np.float64],
        angles: NDArray[np.float64] | Sequence[float] | None = None,
        attenuation_map: NDArray[np.float64] | None = None,
        voxel_size: float | None = None,
    ) -> NDArray[np.float64]:
        """
        Выполняет томографическую реконструкцию методом FBP и возвращает восстановленный
        2D срез `(nx, nx)` или 3D объём `(nx, nx, ny)`.

        :param projections: Стек проекций `(views, nx)` или `(views, nx, ny)`.
        :param angles: Углы поворота детектора в `hepunits`.
        :param attenuation_map: Опциональная карта линейного ослабления в `hepunits`.
        :param voxel_size: Опциональный физический размер вокселя в `hepunits`.
        :return: Восстановленное распределение активности.
        """
        return self.reconstruct_with_metadata(
            projections=projections,
            angles=angles,
            attenuation_map=attenuation_map,
            voxel_size=voxel_size,
        ).volume
