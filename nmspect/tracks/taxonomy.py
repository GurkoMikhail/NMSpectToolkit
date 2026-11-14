"""Настраиваемая топологическая таксономия треков, каноническая 10-классовая схема и спектрометрические компоненты."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from nmspect.spectrometry.models import PhotonComponent
from nmspect.tracks.models import TrackAnnotator, TrackBatch

TrackPredicate = Callable[[Mapping[str, NDArray[Any]]], NDArray[np.bool_]]
TaxonomyMode = Literal["strict_partition", "priority"]


class TrackTopology(IntEnum):
    """Каноническое дизъюнктное разбиение пространства зарегистрированных треков на 10 топологических групп."""

    GEOMETRIC_FULL = 1
    GEOMETRIC_ESCAPE = 2
    SEPTAL_FULL = 3
    SEPTAL_ESCAPE = 4
    PHANTOM_COMPTON_SINGLE = 5
    PHANTOM_INELASTIC_MULTI = 6
    PHANTOM_COHERENT_ONLY = 7
    COLLIMATOR_ONLY_SCATTER = 8
    PHANTOM_AND_COLLIMATOR = 9
    BACKSCATTER = 10


EXCLUSIVE_TOPOLOGY_NAMES: Final[tuple[str, ...]] = (
    "geometric_full",
    "geometric_escape",
    "septal_full",
    "septal_escape",
    "phantom_compton_single",
    "phantom_inelastic_multi",
    "phantom_coherent_only",
    "collimator_only_scatter",
    "phantom_and_collimator",
    "backscatter",
)

AUXILIARY_TOPOLOGY_NAMES: Final[tuple[str, ...]] = (
    "non_geometric",
    "phantom_pure_compton_ge2",
    "septal_single_full",
    "septal_multi_full",
    "collimator_scatter_no_back",
    "collimator_scatter_any",
    "backscatter_pure",
    "backscatter_mixed",
    "any_ext_scatter",
    "ext_scatter_with_septal",
    "ext_scatter_with_escape",
    "any_crystal_compton",
    "any_crystal_escape",
    "any_septal",
)

ALL_TOPOLOGY_MASK_NAMES: Final[tuple[str, ...]] = EXCLUSIVE_TOPOLOGY_NAMES + AUXILIARY_TOPOLOGY_NAMES

TOPOLOGY_KEY_BY_ENUM: Final[dict[TrackTopology, str]] = {
    TrackTopology.GEOMETRIC_FULL: "geometric_full",
    TrackTopology.GEOMETRIC_ESCAPE: "geometric_escape",
    TrackTopology.SEPTAL_FULL: "septal_full",
    TrackTopology.SEPTAL_ESCAPE: "septal_escape",
    TrackTopology.PHANTOM_COMPTON_SINGLE: "phantom_compton_single",
    TrackTopology.PHANTOM_INELASTIC_MULTI: "phantom_inelastic_multi",
    TrackTopology.PHANTOM_COHERENT_ONLY: "phantom_coherent_only",
    TrackTopology.COLLIMATOR_ONLY_SCATTER: "collimator_only_scatter",
    TrackTopology.PHANTOM_AND_COLLIMATOR: "phantom_and_collimator",
    TrackTopology.BACKSCATTER: "backscatter",
}

TOPOLOGY_LABELS_RU: Final[dict[str, str]] = {
    "geometric_full": "C1: Прямой геометрический (полное поглощение)",
    "geometric_escape": "C2: Прямой геометрический (вылет из кристалла)",
    "septal_full": "C3: Прямой пробой септы (полное поглощение)",
    "septal_escape": "C4: Прямой пробой септы (вылет из кристалла)",
    "phantom_compton_single": "C5: Однократный Комптон в фантоме",
    "phantom_inelastic_multi": "C6: Многократное неупругое рассеяние в фантоме",
    "phantom_coherent_only": "C7: Чисто упругое рассеяние Рэлея в фантоме",
    "collimator_only_scatter": "C8: Чистое рассеяние в коллиматоре",
    "phantom_and_collimator": "C9: Смешанное рассеяние (фантом + коллиматор)",
    "backscatter": "C10: Обратное рассеяние за кристаллом",
}

DEFAULT_TOPOLOGY_COLORS: Final[dict[str, str]] = {
    "geometric_full": "#1b9e77",
    "geometric_escape": "#66c2a5",
    "septal_full": "#d95f02",
    "septal_escape": "#fc8d62",
    "phantom_compton_single": "#377eb8",
    "phantom_inelastic_multi": "#6baed6",
    "phantom_coherent_only": "#9ecae1",
    "collimator_only_scatter": "#984ea3",
    "phantom_and_collimator": "#bc80bd",
    "backscatter": "#e41a1c",
}

COMPONENT_LABELS_RU: Final[dict[PhotonComponent, str]] = {
    PhotonComponent.GEOMETRIC: "Геометрическая компонента",
    PhotonComponent.SEPTAL_PENETRATION: "Пробой септ коллиматора",
    PhotonComponent.COLLIMATOR_SCATTER: "Рассеяние в коллиматоре",
    PhotonComponent.PHANTOM_SCATTER: "Рассеяние в фантоме",
    PhotonComponent.BACKSCATTER: "Задетекторное обратное рассеяние",
}

TOPOLOGY_TO_COMPONENT_MAP: Final[dict[TrackTopology, PhotonComponent]] = {
    TrackTopology.GEOMETRIC_FULL: PhotonComponent.GEOMETRIC,
    TrackTopology.GEOMETRIC_ESCAPE: PhotonComponent.GEOMETRIC,
    TrackTopology.SEPTAL_FULL: PhotonComponent.SEPTAL_PENETRATION,
    TrackTopology.SEPTAL_ESCAPE: PhotonComponent.SEPTAL_PENETRATION,
    TrackTopology.PHANTOM_COMPTON_SINGLE: PhotonComponent.PHANTOM_SCATTER,
    TrackTopology.PHANTOM_INELASTIC_MULTI: PhotonComponent.PHANTOM_SCATTER,
    TrackTopology.PHANTOM_COHERENT_ONLY: PhotonComponent.PHANTOM_SCATTER,
    TrackTopology.COLLIMATOR_ONLY_SCATTER: PhotonComponent.COLLIMATOR_SCATTER,
    TrackTopology.PHANTOM_AND_COLLIMATOR: PhotonComponent.PHANTOM_SCATTER,
    TrackTopology.BACKSCATTER: PhotonComponent.BACKSCATTER,
}

REQUIRED_TAXONOMY_CELL_KEYS: Final[tuple[str, ...]] = (
    "phantom_compton",
    "phantom_coherent",
    "collimator_compton",
    "collimator_coherent",
    "backscatter_count",
    "septal_crossings",
    "crystal_compton",
    "crystal_photoelectric",
)

FALLBACK_COLOR_CYCLE: Final[tuple[str, ...]] = (
    "#1b9e77",
    "#d95f02",
    "#377eb8",
    "#984ea3",
    "#e41a1c",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#00897b",
    "#5e35b1",
)


@dataclass(frozen=True)
class FieldCondition:
    """Декларативное векторное условие отбора треков по одному скалярному полю `tracks.cell_data`.

    Все заданные ограничения (`equals`, `not_equals`, `in_values`, `min_value`, `max_value`) объединяются
    логическим И (`&`). Значения физических величин указываются в базовых единицах `hepunits`.
    """

    field_name: str
    equals: int | float | None = None
    not_equals: int | float | None = None
    in_values: tuple[int | float, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    include_min: bool = True
    include_max: bool = True

    def __post_init__(self) -> None:
        """Проверяет корректность имени поля и непротиворечивость числовых границ."""
        if not self.field_name.strip():
            raise ValueError("Имя поля field_name в FieldCondition не может быть пустым.")
        has_constraint = (
            self.equals is not None
            or self.not_equals is not None
            or self.in_values is not None
            or self.min_value is not None
            or self.max_value is not None
        )
        if not has_constraint:
            raise ValueError(
                f"В FieldCondition для поля '{self.field_name}' должно быть задано хотя бы одно ограничение."
            )
        if self.in_values is not None and len(self.in_values) == 0:
            raise ValueError(f"Кортеж in_values для поля '{self.field_name}' не может быть пустым.")
        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ValueError(
                f"Некорректный диапазон в FieldCondition('{self.field_name}'): "
                f"min_value ({self.min_value}) > max_value ({self.max_value})."
            )

    def evaluate(self, cell_data: Mapping[str, NDArray[Any]], num_tracks: int) -> NDArray[np.bool_]:
        """Вычисляет булеву маску формы `(num_tracks,)` для пакета треков."""
        if self.field_name not in cell_data:
            raise KeyError(
                f"В tracks.cell_data отсутствует поле '{self.field_name}', требуемое правилом таксономии."
            )
        if num_tracks == 0:
            return np.empty(0, dtype=np.bool_)

        values = np.asarray(cell_data[self.field_name])
        if values.ndim != 1 or values.size != num_tracks:
            raise ValueError(
                f"Поле '{self.field_name}' должно быть одномерным массивом длины {num_tracks}, "
                f"получено: {values.shape}."
            )

        result_mask: NDArray[np.bool_] = np.ones(num_tracks, dtype=np.bool_)
        if self.equals is not None:
            result_mask &= values == self.equals
        if self.not_equals is not None:
            result_mask &= values != self.not_equals
        if self.in_values is not None:
            result_mask &= np.isin(values, np.asarray(self.in_values))
        if self.min_value is not None:
            result_mask &= (values >= self.min_value) if self.include_min else (values > self.min_value)
        if self.max_value is not None:
            result_mask &= (values <= self.max_value) if self.include_max else (values < self.max_value)
        return result_mask

    def to_dict(self) -> dict[str, Any]:
        """Сериализует условие в словарь, совместимый с JSON/YAML."""
        payload: dict[str, Any] = {"field_name": self.field_name}
        if self.equals is not None:
            payload["equals"] = self.equals
        if self.not_equals is not None:
            payload["not_equals"] = self.not_equals
        if self.in_values is not None:
            payload["in_values"] = list(self.in_values)
        if self.min_value is not None:
            payload["min_value"] = self.min_value
            payload["include_min"] = self.include_min
        if self.max_value is not None:
            payload["max_value"] = self.max_value
            payload["include_max"] = self.include_max
        return payload

    @classmethod
    def from_dict(cls, raw_data: Mapping[str, Any]) -> FieldCondition:
        """Восстанавливает экземпляр `FieldCondition` из словаря."""
        raw_allowed = raw_data.get("in_values")
        return cls(
            field_name=str(raw_data["field_name"]),
            equals=raw_data.get("equals"),
            not_equals=raw_data.get("not_equals"),
            in_values=tuple(raw_allowed) if raw_allowed is not None else None,
            min_value=float(raw_data["min_value"]) if raw_data.get("min_value") is not None else None,
            max_value=float(raw_data["max_value"]) if raw_data.get("max_value") is not None else None,
            include_min=bool(raw_data.get("include_min", True)),
            include_max=bool(raw_data.get("include_max", True)),
        )


def _evaluate_rule_criteria(
    rule_name: str,
    conditions: tuple[FieldCondition, ...],
    predicate: TrackPredicate | None,
    cell_data: Mapping[str, NDArray[Any]],
    num_tracks: int,
) -> NDArray[np.bool_]:
    """Вычисляет конъюнкцию декларативных условий `FieldCondition` и пользовательского векторного предиката."""
    if num_tracks == 0:
        for cond_item in conditions:
            cond_item.evaluate(cell_data, 0)
        return np.empty(0, dtype=np.bool_)

    combined_mask: NDArray[np.bool_] = np.ones(num_tracks, dtype=np.bool_)
    for cond_item in conditions:
        combined_mask &= cond_item.evaluate(cell_data, num_tracks)

    if predicate is not None:
        pred_raw = np.asarray(predicate(cell_data), dtype=np.bool_)
        if pred_raw.ndim != 1 or pred_raw.size != num_tracks:
            raise ValueError(
                f"Пользовательский предикат правила '{rule_name}' вернул массив формы {pred_raw.shape} "
                f"вместо ожидаемой ({num_tracks},)."
            )
        combined_mask &= pred_raw

    return combined_mask


@dataclass(frozen=True)
class TrackClassRule:
    """Правило выделения основного взаимоисключающего класса треков в рамках `TaxonomyScheme`.

    Attributes:
        name: Программное имя класса (ключ словарей `exclusive_masks`, сводок и таблиц).
        label: Человекочитаемая подпись класса на русском языке для отчётов и легенд графиков.
        photon_component: Соответствующая укрупнённая спектрометрическая компонента `PhotonComponent` (`0..4`).
        conditions: Кортеж декларативных условий `FieldCondition` (объединяются по `AND`).
        predicate: Опциональная векторная функция `Callable[[cell_data], bool_mask]` для составных выражений.
        code: Опциональный целочисленный код `1..255` для `cell_data['topology_class']` (если `None` — по порядку).
        color: HEX-цвет для визуализации на спектрах и профилях ФРТ.
        is_primary_signal: Признак принадлежности к полезному первичному сигналу (эталон калибровки DEW/TEW и DOI).
    """

    name: str
    label: str
    photon_component: PhotonComponent
    conditions: tuple[FieldCondition, ...] = ()
    predicate: TrackPredicate | None = None
    code: int | None = None
    color: str = "#377eb8"
    is_primary_signal: bool = False

    def __post_init__(self) -> None:
        """Валидирует идентификатор, код `uint8` и наличие подписи."""
        if not self.name.strip():
            raise ValueError("Имя класса name в TrackClassRule не может быть пустым.")
        if not self.label.strip():
            raise ValueError(f"Подпись label в TrackClassRule('{self.name}') не может быть пустой.")
        if self.code is not None and (self.code < 1 or self.code > 255):
            raise ValueError(
                f"Код класса TrackClassRule('{self.name}') должен лежать в диапазоне [1, 255], получено: {self.code}."
            )

    @property
    def has_criteria(self) -> bool:
        """Возвращает `True`, если у правила задано хотя бы одно декларативное условие или предикат."""
        return len(self.conditions) > 0 or self.predicate is not None

    def evaluate(self, cell_data: Mapping[str, NDArray[Any]], num_tracks: int) -> NDArray[np.bool_]:
        """Вычисляет булеву маску соответствия правилу для `num_tracks` треков."""
        return _evaluate_rule_criteria(
            rule_name=self.name,
            conditions=self.conditions,
            predicate=self.predicate,
            cell_data=cell_data,
            num_tracks=num_tracks,
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализует декларативное правило в словарь (требует `predicate is None`)."""
        if self.predicate is not None:
            raise ValueError(
                f"Правило '{self.name}' использует Python-функцию predicate и не может быть сериализовано в JSON."
            )
        return {
            "name": self.name,
            "label": self.label,
            "photon_component": self.photon_component.name,
            "conditions": [cond_item.to_dict() for cond_item in self.conditions],
            "code": self.code,
            "color": self.color,
            "is_primary_signal": self.is_primary_signal,
        }

    @classmethod
    def from_dict(cls, raw_data: Mapping[str, Any]) -> TrackClassRule:
        """Восстанавливает декларативное правило `TrackClassRule` из словаря."""
        comp_raw = raw_data["photon_component"]
        resolved_comp = (
            PhotonComponent[comp_raw] if isinstance(comp_raw, str) else PhotonComponent(int(comp_raw))
        )
        raw_conds = raw_data.get("conditions", ())
        return cls(
            name=str(raw_data["name"]),
            label=str(raw_data["label"]),
            photon_component=resolved_comp,
            conditions=tuple(FieldCondition.from_dict(item) for item in raw_conds),
            code=int(raw_data["code"]) if raw_data.get("code") is not None else None,
            color=str(raw_data.get("color", "#377eb8")),
            is_primary_signal=bool(raw_data.get("is_primary_signal", False)),
        )


@dataclass(frozen=True)
class AuxiliaryMaskRule:
    """Правило вычисления дополнительного (не обязательно дизъюнктного) диагностического среза треков."""

    name: str
    label: str = ""
    conditions: tuple[FieldCondition, ...] = ()
    predicate: TrackPredicate | None = None

    def __post_init__(self) -> None:
        """Проверяет непустоту имени и наличие условий отбора."""
        if not self.name.strip():
            raise ValueError("Имя дополнительного среза AuxiliaryMaskRule не может быть пустым.")
        if len(self.conditions) == 0 and self.predicate is None:
            raise ValueError(
                f"В AuxiliaryMaskRule('{self.name}') необходимо задать conditions или predicate."
            )

    def evaluate(self, cell_data: Mapping[str, NDArray[Any]], num_tracks: int) -> NDArray[np.bool_]:
        """Вычисляет булеву маску диагностического среза."""
        return _evaluate_rule_criteria(
            rule_name=self.name,
            conditions=self.conditions,
            predicate=self.predicate,
            cell_data=cell_data,
            num_tracks=num_tracks,
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализует декларативное вспомогательное правило в словарь."""
        if self.predicate is not None:
            raise ValueError(
                f"Дополнительный срез '{self.name}' использует predicate и не может быть сериализован в JSON."
            )
        return {
            "name": self.name,
            "label": self.label,
            "conditions": [cond_item.to_dict() for cond_item in self.conditions],
        }

    @classmethod
    def from_dict(cls, raw_data: Mapping[str, Any]) -> AuxiliaryMaskRule:
        """Восстанавливает `AuxiliaryMaskRule` из словаря."""
        return cls(
            name=str(raw_data["name"]),
            label=str(raw_data.get("label", "")),
            conditions=tuple(FieldCondition.from_dict(item) for item in raw_data.get("conditions", ())),
        )


@dataclass(frozen=True)
class TrackClassification:
    """Результат классификации пакета треков на взаимоисключающие классы и 5 спектрометрических компонент.

    Attributes:
        topology_codes: Целочисленные коды классов (`1..255`) формы `(num_tracks,)`, `uint8`.
        component_codes: Укрупнённые коды `PhotonComponent` (`0..4`) формы `(num_tracks,)`, `uint8`.
        exclusive_masks: Словарь попарно непересекающихся булевых масок (`sum == num_tracks`).
        auxiliary_masks: Словарь дополнительных диагностических булевых масок.
        scheme_name: Имя применённой схемы таксономии `TaxonomyScheme`.
        primary_mask: Булева маска целевого первичного сигнала (`is_primary_signal`).
        labels: Словарь подписей основных классов `name -> label`.
        colors: Словарь HEX-цветов основных классов `name -> color`.
    """

    topology_codes: NDArray[np.uint8]
    component_codes: NDArray[np.uint8]
    exclusive_masks: dict[str, NDArray[np.bool_]]
    auxiliary_masks: dict[str, NDArray[np.bool_]]
    scheme_name: str = "default_10class"
    primary_mask: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, dtype=np.bool_))
    labels: dict[str, str] = field(default_factory=lambda: dict(TOPOLOGY_LABELS_RU))
    colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TOPOLOGY_COLORS))

    @property
    def all_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Возвращает объединённый словарь всех булевых масок (дизъюнктных и вспомогательных)."""
        merged: dict[str, NDArray[np.bool_]] = dict(self.exclusive_masks)
        merged.update(self.auxiliary_masks)
        return merged


@dataclass(frozen=True)
class TaxonomyScheme:
    """Конфигурируемая схема классификации треков с поддержкой строгого разбиения и приоритетного каскада.

    В режиме `mode='strict_partition'` все правила в `rules` обязаны быть попарно непересекающимися и в сумме
    покрывать 100% треков пакета (любой пропуск или пересечение вызывает информативный `RuntimeError`).
    В режиме `mode='priority'` правила применяются по порядку объявления к ещё не классифицированным трекам,
    а оставшиеся треки направляются в `fallback_rule` (если он задан).
    """

    name: str
    rules: tuple[TrackClassRule, ...]
    mode: TaxonomyMode = "strict_partition"
    fallback_rule: TrackClassRule | None = None
    auxiliary_rules: tuple[AuxiliaryMaskRule, ...] = ()
    required_cell_keys: tuple[str, ...] = ()
    auto_component_auxiliaries: bool = True

    def __post_init__(self) -> None:
        """Проверяет уникальность имён и кодов правил, а также согласованность режима разбиения."""
        if not self.name.strip():
            raise ValueError("Имя схемы таксономии TaxonomyScheme.name не может быть пустым.")
        if self.mode not in ("strict_partition", "priority"):
            raise ValueError(f"Неподдерживаемый режим таксономии mode='{self.mode}'.")
        if len(self.rules) == 0 and self.fallback_rule is None:
            raise ValueError(f"Схема таксономии '{self.name}' должна содержать хотя бы одно правило.")
        if self.mode == "strict_partition" and self.fallback_rule is not None:
            raise ValueError(
                "Параметр fallback_rule допустим только в каскадном режиме mode='priority', "
                "так как в mode='strict_partition' все правила проверяются на независимую непересекаемость."
            )

        for rule_item in self.rules:
            if not rule_item.has_criteria:
                raise ValueError(
                    f"Основное правило '{rule_item.name}' в схеме '{self.name}' должно иметь "
                    f"хотя бы одно условие conditions или функцию predicate."
                )

        effective = self.effective_rules
        seen_names: set[str] = set()
        used_codes: set[int] = set()
        for rule_item in effective:
            if rule_item.name in seen_names:
                raise ValueError(f"Дублирующееся имя класса '{rule_item.name}' в схеме '{self.name}'.")
            seen_names.add(rule_item.name)
            if rule_item.code is not None:
                if rule_item.code in used_codes:
                    raise ValueError(
                        f"Дублирующийся целочисленный код code={rule_item.code} в схеме '{self.name}'."
                    )
                used_codes.add(rule_item.code)

        for aux_item in self.auxiliary_rules:
            if aux_item.name in seen_names:
                raise ValueError(
                    f"Имя вспомогательного среза '{aux_item.name}' конфликтует с существующим именем в '{self.name}'."
                )
            seen_names.add(aux_item.name)

        if len(effective) > 255:
            raise ValueError(f"Число классов ({len(effective)}) превышает ёмкость типа uint8 (255).")

    @property
    def effective_rules(self) -> tuple[TrackClassRule, ...]:
        """Возвращает полный упорядоченный кортеж основных правил (включая `fallback_rule`, если он задан)."""
        if self.fallback_rule is not None:
            return (*self.rules, self.fallback_rule)
        return self.rules

    @property
    def exclusive_names(self) -> tuple[str, ...]:
        """Имена всех взаимоисключающих классов схемы в порядке их определения."""
        return tuple(rule_item.name for rule_item in self.effective_rules)

    @property
    def resolved_codes(self) -> dict[str, int]:
        """Отображение имени класса в уникальный код `1..255`."""
        reserved = {rule_item.code for rule_item in self.effective_rules if rule_item.code is not None}
        next_free = 1
        mapping: dict[str, int] = {}
        for rule_item in self.effective_rules:
            if rule_item.code is not None:
                mapping[rule_item.name] = rule_item.code
            else:
                while next_free in reserved:
                    next_free += 1
                if next_free > 255:
                    raise ValueError(f"Не удалось выделить свободный код uint8 для правила '{rule_item.name}'.")
                mapping[rule_item.name] = next_free
                reserved.add(next_free)
                next_free += 1
        return mapping

    @property
    def labels_map(self) -> dict[str, str]:
        """Словарь человекочитаемых подписей основных классов `name -> label`."""
        return {rule_item.name: rule_item.label for rule_item in self.effective_rules}

    @property
    def colors_map(self) -> dict[str, str]:
        """Словарь HEX-цветов основных классов `name -> color`."""
        result: dict[str, str] = {}
        for idx, rule_item in enumerate(self.effective_rules):
            resolved_col = rule_item.color.strip()
            result[rule_item.name] = (
                resolved_col if resolved_col else FALLBACK_COLOR_CYCLE[idx % len(FALLBACK_COLOR_CYCLE)]
            )
        return result

    @property
    def primary_class_names(self) -> tuple[str, ...]:
        """Кортеж имён классов, формирующих эталонный первичный сигнал (`is_primary_signal`)."""
        explicit_primary = tuple(
            rule_item.name for rule_item in self.effective_rules if rule_item.is_primary_signal
        )
        if explicit_primary:
            return explicit_primary
        return tuple(
            rule_item.name
            for rule_item in self.effective_rules
            if rule_item.photon_component == PhotonComponent.GEOMETRIC
        )

    @property
    def auto_auxiliary_names(self) -> tuple[str, ...]:
        """Имена стандартных диагностических срезов, автоматически добавляемых при `auto_component_auxiliaries=True`."""
        if not self.auto_component_auxiliaries:
            return ()
        occupied = set(self.exclusive_names) | {aux_r.name for aux_r in self.auxiliary_rules}
        standard_candidates = (
            "non_geometric",
            "any_ext_scatter",
            "any_septal",
            "collimator_scatter_no_back",
            "phantom_compton_all",
        )
        return tuple(cand_name for cand_name in standard_candidates if cand_name not in occupied)

    @property
    def auxiliary_names(self) -> tuple[str, ...]:
        """Полный кортеж имён вспомогательных масок схемы."""
        return (*(aux_r.name for aux_r in self.auxiliary_rules), *self.auto_auxiliary_names)

    @property
    def all_mask_names(self) -> tuple[str, ...]:
        """Полный кортеж имён всех масок (`exclusive_names + auxiliary_names`)."""
        return (*self.exclusive_names, *self.auxiliary_names)

    def map_codes_to_components(self, topology_codes: NDArray[np.integer[Any]]) -> NDArray[np.uint8]:
        """Векторно отображает массив кодов классов данной схемы в коды `PhotonComponent` (`0..4`)."""
        codes_arr = np.asarray(topology_codes, dtype=np.int32)
        if codes_arr.size == 0:
            return np.empty(0, dtype=np.uint8)

        code_map = self.resolved_codes
        valid_codes = np.array(list(code_map.values()), dtype=np.int32)
        if not bool(np.all(np.isin(codes_arr, valid_codes))):
            raise ValueError(
                f"Массив topology_codes содержит значения, не принадлежащие схеме '{self.name}' "
                f"(допустимые коды: {sorted(valid_codes.tolist())})."
            )

        lookup = np.zeros(256, dtype=np.uint8)
        for rule_item in self.effective_rules:
            cls_code = code_map[rule_item.name]
            lookup[cls_code] = np.uint8(int(rule_item.photon_component))
        return lookup[codes_arr]

    def with_added_auxiliary(self, *extra_rules: AuxiliaryMaskRule) -> TaxonomyScheme:
        """Создаёт новую схему `TaxonomyScheme` с добавленными пользовательскими вспомогательными срезами."""
        return TaxonomyScheme(
            name=self.name,
            rules=self.rules,
            mode=self.mode,
            fallback_rule=self.fallback_rule,
            auxiliary_rules=(*self.auxiliary_rules, *extra_rules),
            required_cell_keys=self.required_cell_keys,
            auto_component_auxiliaries=self.auto_component_auxiliaries,
        )

    def with_replaced_rule(self, target_name: str, new_rule: TrackClassRule) -> TaxonomyScheme:
        """Создаёт копию схемы с заменой одного основного правила `target_name` на `new_rule`."""
        replaced = False
        updated_rules: list[TrackClassRule] = []
        for rule_item in self.rules:
            if rule_item.name == target_name:
                updated_rules.append(new_rule)
                replaced = True
            else:
                updated_rules.append(rule_item)

        updated_fallback = self.fallback_rule
        if not replaced and self.fallback_rule is not None and self.fallback_rule.name == target_name:
            updated_fallback = new_rule
            replaced = True

        if not replaced:
            raise KeyError(f"В схеме '{self.name}' отсутствует основное правило с именем '{target_name}'.")

        return TaxonomyScheme(
            name=self.name,
            rules=tuple(updated_rules),
            mode=self.mode,
            fallback_rule=updated_fallback,
            auxiliary_rules=self.auxiliary_rules,
            required_cell_keys=self.required_cell_keys,
            auto_component_auxiliaries=self.auto_component_auxiliaries,
        )

    def _build_auto_auxiliaries(
        self,
        cell_data: Mapping[str, NDArray[Any]],
        component_codes: NDArray[np.uint8],
        primary_mask: NDArray[np.bool_],
    ) -> dict[str, NDArray[np.bool_]]:
        """Формирует стандартные сводные маски для совместимости потокового анализатора с любой кастомной схемой."""
        auto_names = self.auto_auxiliary_names
        if not auto_names:
            return {}

        has_phantom_fields = "phantom_compton" in cell_data and "phantom_coherent" in cell_data
        has_coll_fields = "collimator_compton" in cell_data and "collimator_coherent" in cell_data
        has_back_field = "backscatter_count" in cell_data

        generated: dict[str, NDArray[np.bool_]] = {}
        for auto_key in auto_names:
            if auto_key == "non_geometric":
                generated[auto_key] = ~primary_mask
            elif auto_key == "any_septal":
                if "septal_crossings" in cell_data:
                    generated[auto_key] = np.asarray(cell_data["septal_crossings"]) >= 1
                else:
                    generated[auto_key] = component_codes == np.uint8(int(PhotonComponent.SEPTAL_PENETRATION))
            elif auto_key == "any_ext_scatter":
                if has_phantom_fields and has_coll_fields and has_back_field:
                    generated[auto_key] = (
                        (np.asarray(cell_data["phantom_compton"]) >= 1)
                        | (np.asarray(cell_data["phantom_coherent"]) >= 1)
                        | (np.asarray(cell_data["collimator_compton"]) >= 1)
                        | (np.asarray(cell_data["collimator_coherent"]) >= 1)
                        | (np.asarray(cell_data["backscatter_count"]) >= 1)
                    )
                else:
                    generated[auto_key] = np.isin(
                        component_codes,
                        (
                            int(PhotonComponent.PHANTOM_SCATTER),
                            int(PhotonComponent.COLLIMATOR_SCATTER),
                            int(PhotonComponent.BACKSCATTER),
                        ),
                    )
            elif auto_key == "collimator_scatter_no_back":
                if has_coll_fields and has_back_field:
                    has_coll = (np.asarray(cell_data["collimator_compton"]) >= 1) | (
                        np.asarray(cell_data["collimator_coherent"]) >= 1
                    )
                    generated[auto_key] = has_coll & (np.asarray(cell_data["backscatter_count"]) == 0)
                else:
                    generated[auto_key] = component_codes == np.uint8(int(PhotonComponent.COLLIMATOR_SCATTER))
            elif auto_key == "phantom_compton_all":
                if "phantom_compton" in cell_data and has_coll_fields and has_back_field:
                    has_coll = (np.asarray(cell_data["collimator_compton"]) >= 1) | (
                        np.asarray(cell_data["collimator_coherent"]) >= 1
                    )
                    has_back = np.asarray(cell_data["backscatter_count"]) >= 1
                    generated[auto_key] = (np.asarray(cell_data["phantom_compton"]) >= 1) & (~has_coll) & (~has_back)
                else:
                    generated[auto_key] = component_codes == np.uint8(int(PhotonComponent.PHANTOM_SCATTER))
        return generated

    def classify(self, tracks: TrackBatch) -> TrackClassification:
        """Классифицирует все треки пакета согласно правилам схемы и проверяет полноту разбиения."""
        for req_key in self.required_cell_keys:
            if req_key not in tracks.cell_data:
                raise KeyError(
                    f"В tracks.cell_data отсутствует обязательный ключ '{req_key}' "
                    f"для схемы таксономии '{self.name}'."
                )

        num_tracks = tracks.num_tracks
        labels_dict = self.labels_map
        colors_dict = self.colors_map
        if num_tracks == 0:
            for rule_item in self.effective_rules:
                rule_item.evaluate(tracks.cell_data, 0)
            for aux_item in self.auxiliary_rules:
                aux_item.evaluate(tracks.cell_data, 0)
            empty_bool = np.empty(0, dtype=np.bool_)
            return TrackClassification(
                topology_codes=np.empty(0, dtype=np.uint8),
                component_codes=np.empty(0, dtype=np.uint8),
                exclusive_masks={cls_name: empty_bool.copy() for cls_name in self.exclusive_names},
                auxiliary_masks={aux_name: empty_bool.copy() for aux_name in self.auxiliary_names},
                scheme_name=self.name,
                primary_mask=empty_bool.copy(),
                labels=labels_dict,
                colors=colors_dict,
            )

        code_map = self.resolved_codes
        exclusive_masks: dict[str, NDArray[np.bool_]] = {}
        topology_codes = np.zeros(num_tracks, dtype=np.uint8)

        if self.mode == "strict_partition":
            exclusive_sum = np.zeros(num_tracks, dtype=np.int32)
            for rule_item in self.rules:
                mask_arr = rule_item.evaluate(tracks.cell_data, num_tracks)
                exclusive_masks[rule_item.name] = mask_arr
                exclusive_sum += mask_arr.astype(np.int32)
                topology_codes = np.where(mask_arr, np.uint8(code_map[rule_item.name]), topology_codes)

            if not bool(np.all(exclusive_sum == 1)):
                unassigned_cnt = int(np.count_nonzero(exclusive_sum == 0))
                overlap_cnt = int(np.count_nonzero(exclusive_sum > 1))
                conflict_detail = ""
                if overlap_cnt > 0:
                    rule_names_seq = tuple(exclusive_masks.keys())
                    for idx_first in range(len(rule_names_seq)):
                        first_name = rule_names_seq[idx_first]
                        for idx_second in range(idx_first + 1, len(rule_names_seq)):
                            second_name = rule_names_seq[idx_second]
                            if bool(np.any(exclusive_masks[first_name] & exclusive_masks[second_name])):
                                conflict_detail = f" Пересекаются правила '{first_name}' и '{second_name}'."
                                break
                        if conflict_detail:
                            break
                raise RuntimeError(
                    f"Нарушен инвариант строгого разбиения в схеме '{self.name}': "
                    f"не классифицировано треков — {unassigned_cnt}, попало в несколько классов — {overlap_cnt}."
                    f"{conflict_detail}"
                )
        else:
            unassigned: NDArray[np.bool_] = np.ones(num_tracks, dtype=np.bool_)
            for rule_item in self.rules:
                raw_match = rule_item.evaluate(tracks.cell_data, num_tracks)
                assigned_mask = unassigned & raw_match
                exclusive_masks[rule_item.name] = assigned_mask
                topology_codes = np.where(assigned_mask, np.uint8(code_map[rule_item.name]), topology_codes)
                unassigned &= ~assigned_mask

            if self.fallback_rule is not None:
                fb_match = (
                    unassigned & self.fallback_rule.evaluate(tracks.cell_data, num_tracks)
                    if self.fallback_rule.has_criteria
                    else unassigned.copy()
                )
                exclusive_masks[self.fallback_rule.name] = fb_match
                topology_codes = np.where(fb_match, np.uint8(code_map[self.fallback_rule.name]), topology_codes)
                unassigned &= ~fb_match

            remaining_cnt = int(np.count_nonzero(unassigned))
            if remaining_cnt > 0:
                raise RuntimeError(
                    f"В каскадной схеме '{self.name}' остались не классифицированными {remaining_cnt} треков. "
                    f"Добавьте замыкающее правило fallback_rule или расширьте условия."
                )

        component_codes = self.map_codes_to_components(topology_codes)

        primary_mask = np.zeros(num_tracks, dtype=np.bool_)
        for pri_name in self.primary_class_names:
            primary_mask |= exclusive_masks[pri_name]

        auxiliary_masks: dict[str, NDArray[np.bool_]] = {}
        for aux_rule in self.auxiliary_rules:
            auxiliary_masks[aux_rule.name] = aux_rule.evaluate(tracks.cell_data, num_tracks)

        if self.auto_component_auxiliaries:
            auxiliary_masks.update(self._build_auto_auxiliaries(tracks.cell_data, component_codes, primary_mask))

        return TrackClassification(
            topology_codes=topology_codes,
            component_codes=component_codes,
            exclusive_masks=exclusive_masks,
            auxiliary_masks=auxiliary_masks,
            scheme_name=self.name,
            primary_mask=primary_mask,
            labels=labels_dict,
            colors=colors_dict,
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализует декларативную схему таксономии в JSON-совместимый словарь."""
        return {
            "name": self.name,
            "mode": self.mode,
            "rules": [rule_item.to_dict() for rule_item in self.rules],
            "fallback_rule": self.fallback_rule.to_dict() if self.fallback_rule is not None else None,
            "auxiliary_rules": [aux_item.to_dict() for aux_item in self.auxiliary_rules],
            "required_cell_keys": list(self.required_cell_keys),
            "auto_component_auxiliaries": self.auto_component_auxiliaries,
        }

    @classmethod
    def from_dict(cls, raw_data: Mapping[str, Any]) -> TaxonomyScheme:
        """Восстанавливает декларативную схему `TaxonomyScheme` из словаря."""
        raw_mode = str(raw_data.get("mode", "strict_partition"))
        if raw_mode not in ("strict_partition", "priority"):
            raise ValueError(f"Недопустимое значение mode='{raw_mode}' при чтении TaxonomyScheme.")
        resolved_mode: TaxonomyMode = "strict_partition" if raw_mode == "strict_partition" else "priority"
        fb_raw = raw_data.get("fallback_rule")
        return cls(
            name=str(raw_data["name"]),
            mode=resolved_mode,
            rules=tuple(TrackClassRule.from_dict(item) for item in raw_data.get("rules", ())),
            fallback_rule=TrackClassRule.from_dict(fb_raw) if fb_raw is not None else None,
            auxiliary_rules=tuple(AuxiliaryMaskRule.from_dict(item) for item in raw_data.get("auxiliary_rules", ())),
            required_cell_keys=tuple(str(key_s) for key_s in raw_data.get("required_cell_keys", ())),
            auto_component_auxiliaries=bool(raw_data.get("auto_component_auxiliaries", True)),
        )


def _has_phantom_scatter(cell_data: Mapping[str, NDArray[Any]]) -> NDArray[np.bool_]:
    return (np.asarray(cell_data["phantom_compton"]) >= 1) | (np.asarray(cell_data["phantom_coherent"]) >= 1)


def _has_collimator_scatter(cell_data: Mapping[str, NDArray[Any]]) -> NDArray[np.bool_]:
    return (np.asarray(cell_data["collimator_compton"]) >= 1) | (np.asarray(cell_data["collimator_coherent"]) >= 1)


def _has_backscatter(cell_data: Mapping[str, NDArray[Any]]) -> NDArray[np.bool_]:
    return np.asarray(cell_data["backscatter_count"]) >= 1


def _any_ext_scatter(cell_data: Mapping[str, NDArray[Any]]) -> NDArray[np.bool_]:
    return _has_phantom_scatter(cell_data) | _has_collimator_scatter(cell_data) | _has_backscatter(cell_data)


def _build_default_10class_scheme() -> TaxonomyScheme:
    """Создаёт каноническую строго взаимоисключающую 10-классовую таксономию треков ОФЭКТ."""
    rules_10 = (
        TrackClassRule(
            name="geometric_full",
            label=TOPOLOGY_LABELS_RU["geometric_full"],
            photon_component=PhotonComponent.GEOMETRIC,
            code=int(TrackTopology.GEOMETRIC_FULL),
            color=DEFAULT_TOPOLOGY_COLORS["geometric_full"],
            is_primary_signal=True,
            predicate=lambda cd: (
                (~_any_ext_scatter(cd))
                & (np.asarray(cd["septal_crossings"]) == 0)
                & (np.asarray(cd["crystal_photoelectric"]) >= 1)
            ),
        ),
        TrackClassRule(
            name="geometric_escape",
            label=TOPOLOGY_LABELS_RU["geometric_escape"],
            photon_component=PhotonComponent.GEOMETRIC,
            code=int(TrackTopology.GEOMETRIC_ESCAPE),
            color=DEFAULT_TOPOLOGY_COLORS["geometric_escape"],
            predicate=lambda cd: (
                (~_any_ext_scatter(cd))
                & (np.asarray(cd["septal_crossings"]) == 0)
                & (np.asarray(cd["crystal_photoelectric"]) == 0)
            ),
        ),
        TrackClassRule(
            name="septal_full",
            label=TOPOLOGY_LABELS_RU["septal_full"],
            photon_component=PhotonComponent.SEPTAL_PENETRATION,
            code=int(TrackTopology.SEPTAL_FULL),
            color=DEFAULT_TOPOLOGY_COLORS["septal_full"],
            predicate=lambda cd: (
                (~_any_ext_scatter(cd))
                & (np.asarray(cd["septal_crossings"]) >= 1)
                & (np.asarray(cd["crystal_photoelectric"]) >= 1)
            ),
        ),
        TrackClassRule(
            name="septal_escape",
            label=TOPOLOGY_LABELS_RU["septal_escape"],
            photon_component=PhotonComponent.SEPTAL_PENETRATION,
            code=int(TrackTopology.SEPTAL_ESCAPE),
            color=DEFAULT_TOPOLOGY_COLORS["septal_escape"],
            predicate=lambda cd: (
                (~_any_ext_scatter(cd))
                & (np.asarray(cd["septal_crossings"]) >= 1)
                & (np.asarray(cd["crystal_photoelectric"]) == 0)
            ),
        ),
        TrackClassRule(
            name="phantom_compton_single",
            label=TOPOLOGY_LABELS_RU["phantom_compton_single"],
            photon_component=PhotonComponent.PHANTOM_SCATTER,
            code=int(TrackTopology.PHANTOM_COMPTON_SINGLE),
            color=DEFAULT_TOPOLOGY_COLORS["phantom_compton_single"],
            predicate=lambda cd: (
                (np.asarray(cd["phantom_compton"]) == 1)
                & (np.asarray(cd["phantom_coherent"]) == 0)
                & (~_has_collimator_scatter(cd))
                & (~_has_backscatter(cd))
            ),
        ),
        TrackClassRule(
            name="phantom_inelastic_multi",
            label=TOPOLOGY_LABELS_RU["phantom_inelastic_multi"],
            photon_component=PhotonComponent.PHANTOM_SCATTER,
            code=int(TrackTopology.PHANTOM_INELASTIC_MULTI),
            color=DEFAULT_TOPOLOGY_COLORS["phantom_inelastic_multi"],
            predicate=lambda cd: (
                (np.asarray(cd["phantom_compton"]) >= 1)
                & (
                    (
                        np.asarray(cd["phantom_compton"], dtype=np.int32)
                        + np.asarray(cd["phantom_coherent"], dtype=np.int32)
                    )
                    >= 2
                )
                & (~_has_collimator_scatter(cd))
                & (~_has_backscatter(cd))
            ),
        ),
        TrackClassRule(
            name="phantom_coherent_only",
            label=TOPOLOGY_LABELS_RU["phantom_coherent_only"],
            photon_component=PhotonComponent.PHANTOM_SCATTER,
            code=int(TrackTopology.PHANTOM_COHERENT_ONLY),
            color=DEFAULT_TOPOLOGY_COLORS["phantom_coherent_only"],
            predicate=lambda cd: (
                (np.asarray(cd["phantom_coherent"]) >= 1)
                & (np.asarray(cd["phantom_compton"]) == 0)
                & (~_has_collimator_scatter(cd))
                & (~_has_backscatter(cd))
            ),
        ),
        TrackClassRule(
            name="collimator_only_scatter",
            label=TOPOLOGY_LABELS_RU["collimator_only_scatter"],
            photon_component=PhotonComponent.COLLIMATOR_SCATTER,
            code=int(TrackTopology.COLLIMATOR_ONLY_SCATTER),
            color=DEFAULT_TOPOLOGY_COLORS["collimator_only_scatter"],
            predicate=lambda cd: (
                (~_has_phantom_scatter(cd)) & _has_collimator_scatter(cd) & (~_has_backscatter(cd))
            ),
        ),
        TrackClassRule(
            name="phantom_and_collimator",
            label=TOPOLOGY_LABELS_RU["phantom_and_collimator"],
            photon_component=PhotonComponent.PHANTOM_SCATTER,
            code=int(TrackTopology.PHANTOM_AND_COLLIMATOR),
            color=DEFAULT_TOPOLOGY_COLORS["phantom_and_collimator"],
            predicate=lambda cd: _has_phantom_scatter(cd) & _has_collimator_scatter(cd) & (~_has_backscatter(cd)),
        ),
        TrackClassRule(
            name="backscatter",
            label=TOPOLOGY_LABELS_RU["backscatter"],
            photon_component=PhotonComponent.BACKSCATTER,
            code=int(TrackTopology.BACKSCATTER),
            color=DEFAULT_TOPOLOGY_COLORS["backscatter"],
            predicate=_has_backscatter,
        ),
    )

    aux_14 = (
        AuxiliaryMaskRule(
            name="non_geometric",
            label="Суммарный негеометрический фон (~C1)",
            predicate=lambda cd: ~(
                (~_any_ext_scatter(cd))
                & (np.asarray(cd["septal_crossings"]) == 0)
                & (np.asarray(cd["crystal_photoelectric"]) >= 1)
            ),
        ),
        AuxiliaryMaskRule(
            name="phantom_pure_compton_ge2",
            label="Многократный Комптон >= 2 в фантоме",
            predicate=lambda cd: (
                (np.asarray(cd["phantom_compton"]) >= 2)
                & (~_has_collimator_scatter(cd))
                & (~_has_backscatter(cd))
            ),
        ),
        AuxiliaryMaskRule(
            name="septal_single_full",
            label="Пробой ровно 1 септы (полное поглощение)",
            predicate=lambda cd: (
                (~_any_ext_scatter(cd))
                & (np.asarray(cd["septal_crossings"]) == 1)
                & (np.asarray(cd["crystal_photoelectric"]) >= 1)
            ),
        ),
        AuxiliaryMaskRule(
            name="septal_multi_full",
            label="Пробой >= 2 септ (полное поглощение)",
            predicate=lambda cd: (
                (~_any_ext_scatter(cd))
                & (np.asarray(cd["septal_crossings"]) >= 2)
                & (np.asarray(cd["crystal_photoelectric"]) >= 1)
            ),
        ),
        AuxiliaryMaskRule(
            name="collimator_scatter_no_back",
            label="Рассеяние в коллиматоре без задетекторного возврата",
            predicate=lambda cd: _has_collimator_scatter(cd) & (~_has_backscatter(cd)),
        ),
        AuxiliaryMaskRule(
            name="collimator_scatter_any",
            label="Любое рассеяние в коллиматоре",
            predicate=_has_collimator_scatter,
        ),
        AuxiliaryMaskRule(
            name="backscatter_pure",
            label="Чистое обратное рассеяние за кристаллом",
            predicate=lambda cd: (
                _has_backscatter(cd) & (~_has_phantom_scatter(cd)) & (~_has_collimator_scatter(cd))
            ),
        ),
        AuxiliaryMaskRule(
            name="backscatter_mixed",
            label="Смешанное обратное рассеяние",
            predicate=lambda cd: _has_backscatter(cd) & (_has_phantom_scatter(cd) | _has_collimator_scatter(cd)),
        ),
        AuxiliaryMaskRule(
            name="any_ext_scatter",
            label="Любое внешнее рассеяние вне кристалла",
            predicate=_any_ext_scatter,
        ),
        AuxiliaryMaskRule(
            name="ext_scatter_with_septal",
            label="Внешнее рассеяние с пересечением септ",
            predicate=lambda cd: _any_ext_scatter(cd) & (np.asarray(cd["septal_crossings"]) >= 1),
        ),
        AuxiliaryMaskRule(
            name="ext_scatter_with_escape",
            label="Внешнее рассеяние с вылетом из кристалла",
            predicate=lambda cd: _any_ext_scatter(cd) & (np.asarray(cd["crystal_photoelectric"]) == 0),
        ),
        AuxiliaryMaskRule(
            name="any_crystal_compton",
            label="Наличие Комптона внутри кристалла",
            conditions=(FieldCondition("crystal_compton", min_value=1),),
        ),
        AuxiliaryMaskRule(
            name="any_crystal_escape",
            label="Неполное поглощение (вылет кванта из кристалла)",
            conditions=(FieldCondition("crystal_photoelectric", equals=0),),
        ),
        AuxiliaryMaskRule(
            name="any_septal",
            label="Любое пересечение септ коллиматора",
            conditions=(FieldCondition("septal_crossings", min_value=1),),
        ),
    )

    return TaxonomyScheme(
        name="default_10class",
        rules=rules_10,
        mode="strict_partition",
        auxiliary_rules=aux_14,
        required_cell_keys=REQUIRED_TAXONOMY_CELL_KEYS,
        auto_component_auxiliaries=False,
    )


DEFAULT_TRACK_TAXONOMY: Final[TaxonomyScheme] = _build_default_10class_scheme()


def map_topology_to_photon_component(
    topology_codes: NDArray[np.integer[Any]],
    scheme: TaxonomyScheme | None = None,
) -> NDArray[np.uint8]:
    """Векторно преобразует массив кодов классов таксономии в коды `PhotonComponent` (`0..4`)."""
    target_scheme = DEFAULT_TRACK_TAXONOMY if scheme is None else scheme
    return target_scheme.map_codes_to_components(topology_codes)


def classify_tracks(
    tracks: TrackBatch,
    scheme: TaxonomyScheme | None = None,
) -> TrackClassification:
    """Классифицирует пакет треков по заданной схеме `TaxonomyScheme` (по умолчанию — 10 классов `DEFAULT_TRACK_TAXONOMY`)."""
    target_scheme = DEFAULT_TRACK_TAXONOMY if scheme is None else scheme
    return target_scheme.classify(tracks)


class TopologyClassAnnotator(TrackAnnotator):
    """Записывает в `tracks.cell_data` поля `'topology_class'` (`uint8`) и `'photon_component'` (`0..4`)."""

    def __init__(self, scheme: TaxonomyScheme | None = None) -> None:
        """Инициализирует аннотатор выбранной схемой таксономии (по умолчанию — `DEFAULT_TRACK_TAXONOMY`)."""
        self.scheme: TaxonomyScheme = DEFAULT_TRACK_TAXONOMY if scheme is None else scheme

    def annotate(self, tracks: TrackBatch, metadata: dict[str, Any] | None = None) -> None:
        """Классифицирует каждый трек пакета и сохраняет коды классов в `tracks.cell_data`."""
        del metadata
        classification = self.scheme.classify(tracks)
        tracks.cell_data["topology_class"] = classification.topology_codes
        tracks.cell_data["photon_component"] = classification.component_codes
