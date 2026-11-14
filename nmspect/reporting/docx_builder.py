"""Сборщик академических научно-технических отчётов в формате Microsoft Word OOXML (.docx)."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.sax.saxutils import escape

from hepunits import mm
from PIL import Image, ImageOps

from nmspect.reporting.omml import inline_math, latex_to_omml

WORD_NS: Final[str] = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MATH_NS: Final[str] = "http://schemas.openxmlformats.org/officeDocument/2006/math"
REL_NS: Final[str] = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
DRAWING_WP_NS: Final[str] = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
DRAWING_MAIN_NS: Final[str] = "http://schemas.openxmlformats.org/drawingml/2006/main"
DRAWING_PIC_NS: Final[str] = "http://schemas.openxmlformats.org/drawingml/2006/picture"

DXA_PER_MILLIMETER: Final[float] = 56.6929133858
EMU_PER_MILLIMETER: Final[float] = 36000.0

RICH_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(<m:oMath>.*?</m:oMath>"
    r"|\$[^$\n]+\$"
    r"|`[^`]+`"
    r"|\*\*[^*]+\*\*"
    r"|\*[^*]+\*"
    r"|\^\{[^}]+\}"
    r"|_\{[^}]+\})",
    re.DOTALL,
)


@dataclass(frozen=True)
class PageGeometryConfig:
    """Геометрия печатного листа документа в базовых единицах `hepunits` (`mm`, по умолчанию ГОСТ А4)."""

    page_width: float = 210.0 * mm
    page_height: float = 297.0 * mm
    margin_left: float = 25.0 * mm
    margin_right: float = 15.0 * mm
    margin_top: float = 20.0 * mm
    margin_bottom: float = 20.0 * mm
    header_offset: float = 12.5 * mm
    footer_offset: float = 12.5 * mm

    @property
    def printable_width(self) -> float:
        """Возвращает полезную ширину области набора текста в базовых единицах `hepunits` (`mm`)."""
        return max(40.0 * mm, self.page_width - self.margin_left - self.margin_right)


def _flatten_image_onto_white(source_image: Image.Image) -> Image.Image:
    """Безопасно сводит прозрачные каналы (`RGBA`/`LA`/`P`) на чистый белый холст `(255, 255, 255)`."""
    has_alpha = (
        source_image.mode in ("RGBA", "LA")
        or (source_image.mode == "P" and "transparency" in source_image.info)
    )
    if not has_alpha:
        return source_image.convert("RGB")

    rgba_img = source_image.convert("RGBA")
    white_canvas = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_canvas, rgba_img)
    return composited.convert("RGB")


def render_rich_runs(
    rich_text: str,
    *,
    default_bold: bool = False,
    default_italic: bool = False,
    font_family: str = "Times New Roman",
    font_half_points: int | None = None,
    color_hex: str | None = None,
) -> str:
    """Преобразует размеченную строку (`**bold**`, `*italic*`, `` `code` ``, `$math$`, `^{sup}`, `_{sub}`) в XML Word."""

    def _build_run_props(
        is_bold: bool,
        is_italic: bool,
        family_name: str,
        vertical_align: str | None = None,
    ) -> str:
        prop_parts = [
            f'<w:rFonts w:ascii="{escape(family_name)}" w:hAnsi="{escape(family_name)}" w:cs="{escape(family_name)}"/>'
        ]
        if is_bold:
            prop_parts.append("<w:b/><w:bCs/>")
        if is_italic:
            prop_parts.append("<w:i/><w:iCs/>")
        if font_half_points is not None:
            prop_parts.append(f'<w:sz w:val="{font_half_points}"/><w:szCs w:val="{font_half_points}"/>')
        if color_hex is not None:
            prop_parts.append(f'<w:color w:val="{escape(color_hex)}"/>')
        if vertical_align is not None:
            prop_parts.append(f'<w:vertAlign w:val="{vertical_align}"/>')
        return f"<w:rPr>{''.join(prop_parts)}</w:rPr>"

    xml_chunks: list[str] = []
    last_pos = 0
    for match_obj in RICH_TOKEN_PATTERN.finditer(rich_text):
        if match_obj.start() > last_pos:
            plain_part = rich_text[last_pos : match_obj.start()]
            run_pr = _build_run_props(default_bold, default_italic, font_family)
            xml_chunks.append(f'<w:r>{run_pr}<w:t xml:space="preserve">{escape(plain_part)}</w:t></w:r>')

        token_str = match_obj.group(0)
        if token_str.startswith("<m:oMath>"):
            xml_chunks.append(token_str)
        elif token_str.startswith("$") and token_str.endswith("$"):
            latex_inner = token_str[1:-1]
            xml_chunks.append(inline_math(latex_to_omml(latex_inner)))
        elif token_str.startswith("`") and token_str.endswith("`"):
            code_inner = token_str[1:-1]
            code_pr = _build_run_props(default_bold, False, "Consolas")
            xml_chunks.append(f'<w:r>{code_pr}<w:t xml:space="preserve">{escape(code_inner)}</w:t></w:r>')
        elif token_str.startswith("**") and token_str.endswith("**"):
            bold_inner = token_str[2:-2]
            xml_chunks.append(
                render_rich_runs(
                    bold_inner,
                    default_bold=True,
                    default_italic=default_italic,
                    font_family=font_family,
                    font_half_points=font_half_points,
                    color_hex=color_hex,
                )
            )
        elif token_str.startswith("*") and token_str.endswith("*"):
            italic_inner = token_str[1:-1]
            xml_chunks.append(
                render_rich_runs(
                    italic_inner,
                    default_bold=default_bold,
                    default_italic=True,
                    font_family=font_family,
                    font_half_points=font_half_points,
                    color_hex=color_hex,
                )
            )
        elif token_str.startswith("^{") and token_str.endswith("}"):
            sup_inner = token_str[2:-1]
            sup_pr = _build_run_props(default_bold, default_italic, font_family, "superscript")
            xml_chunks.append(f'<w:r>{sup_pr}<w:t xml:space="preserve">{escape(sup_inner)}</w:t></w:r>')
        elif token_str.startswith("_{") and token_str.endswith("}"):
            sub_inner = token_str[2:-1]
            sub_pr = _build_run_props(default_bold, default_italic, font_family, "subscript")
            xml_chunks.append(f'<w:r>{sub_pr}<w:t xml:space="preserve">{escape(sub_inner)}</w:t></w:r>')

        last_pos = match_obj.end()

    if last_pos < len(rich_text):
        tail_part = rich_text[last_pos:]
        tail_pr = _build_run_props(default_bold, default_italic, font_family)
        xml_chunks.append(f'<w:r>{tail_pr}<w:t xml:space="preserve">{escape(tail_part)}</w:t></w:r>')

    return "".join(xml_chunks)


class ScientificDocxBuilder:
    """Конструктор строгих научно-технических отчётов OOXML `.docx` без внешних офисных зависимостей."""

    def __init__(
        self,
        geometry: PageGeometryConfig | None = None,
        base_font_family: str = "Times New Roman",
        base_font_half_points: int = 24,
    ) -> None:
        """Инициализирует построитель документа с заданной геометрией страницы и гарнитурой."""
        self.geometry: PageGeometryConfig = geometry if geometry is not None else PageGeometryConfig()
        self.base_font_family: str = base_font_family
        self.base_font_half_points: int = int(base_font_half_points)
        self._body_blocks: list[str] = []
        self._media_entries: list[tuple[str, bytes]] = []
        self._drawing_counter: int = 1

    @property
    def printable_twips(self) -> int:
        """Возвращает ширину полосы набора в двадцатых долях пункта (`dxa`)."""
        return round(float(self.geometry.printable_width / mm) * DXA_PER_MILLIMETER)

    def add_raw_xml(self, xml_block: str) -> None:
        """Добавляет готовый валидный XML-фрагмент напрямую в тело документа `<w:body>`."""
        self._body_blocks.append(xml_block)

    def add_page_break(self) -> None:
        """Добавляет принудительный разрыв страницы."""
        self._body_blocks.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def add_paragraph(
        self,
        text: str,
        *,
        align: str = "both",
        first_line_indent: float = 12.5 * mm,
        left_indent: float = 0.0 * mm,
        space_before: int = 0,
        space_after: int = 100,
        line_spacing: int = 276,
        bold: bool = False,
        italic: bool = False,
        font_half_points: int | None = None,
        color_hex: str | None = None,
        keep_next: bool = False,
    ) -> None:
        """Добавляет текстовый абзац с поддержкой встроенных формул `$...$`, `<m:oMath>` и стилевой разметки."""
        keep_tags = "<w:keepNext/><w:keepLines/>" if keep_next else ""
        left_twips = max(0, round(float(left_indent / mm) * DXA_PER_MILLIMETER))
        first_twips = max(0, round(float(first_line_indent / mm) * DXA_PER_MILLIMETER))
        ind_attr = f'w:left="{left_twips}"'
        if first_twips > 0:
            ind_attr += f' w:firstLine="{first_twips}"'

        para_pr = (
            f"<w:pPr>{keep_tags}"
            f'<w:spacing w:before="{space_before}" w:after="{space_after}" '
            f'w:line="{line_spacing}" w:lineRule="auto"/>'
            f"<w:ind {ind_attr}/>"
            f'<w:jc w:val="{escape(align)}"/>'
            "</w:pPr>"
        )
        runs_xml = render_rich_runs(
            text,
            default_bold=bold,
            default_italic=italic,
            font_family=self.base_font_family,
            font_half_points=font_half_points,
            color_hex=color_hex,
        )
        self._body_blocks.append(f"<w:p>{para_pr}{runs_xml}</w:p>")

    def add_bullet(self, text: str, *, marker_prefix: str = "•") -> None:
        """Добавляет элемент маркированного или нумерованного перечня с висячим отступом."""
        para_pr = (
            "<w:pPr>"
            '<w:spacing w:before="20" w:after="70" w:line="268" w:lineRule="auto"/>'
            '<w:ind w:left="709" w:hanging="354"/>'
            '<w:jc w:val="both"/>'
            "</w:pPr>"
        )
        prefix_xml = render_rich_runs(
            f"{marker_prefix}\t",
            default_bold=marker_prefix != "•",
            font_family=self.base_font_family,
        )
        body_xml = render_rich_runs(text, font_family=self.base_font_family)
        self._body_blocks.append(f"<w:p>{para_pr}{prefix_xml}{body_xml}</w:p>")

    def add_heading(self, title_text: str, level: int = 1, *, page_break_before: bool = False) -> None:
        """Добавляет структурный заголовок `Heading1`..`Heading3`, включаемый в автооглавление и закладки PDF."""
        if page_break_before:
            self.add_page_break()
        clamped_level = max(1, min(3, int(level)))
        style_id = f"Heading{clamped_level}"
        size_lookup = {1: 28, 2: 25, 3: 24}
        before_lookup = {1: 260, 2: 200, 3: 150}
        after_lookup = {1: 120, 2: 90, 3: 70}

        para_pr = (
            f'<w:pPr><w:pStyle w:val="{style_id}"/>'
            "<w:keepNext/><w:keepLines/>"
            f'<w:spacing w:before="{before_lookup[clamped_level]}" w:after="{after_lookup[clamped_level]}" '
            'w:line="260" w:lineRule="auto"/>'
            '<w:ind w:left="0" w:firstLine="0"/>'
            '<w:jc w:val="left"/>'
            f'<w:outlineLvl w:val="{clamped_level - 1}"/>'
            "</w:pPr>"
        )
        runs_xml = render_rich_runs(
            title_text,
            default_bold=True,
            font_family=self.base_font_family,
            font_half_points=size_lookup[clamped_level],
        )
        self._body_blocks.append(f"<w:p>{para_pr}{runs_xml}</w:p>")

    def add_toc_field(self, heading_title: str = "Оглавление") -> None:
        """Вставляет стандартное поле оглавления Word (`TOC \\o \"1-3\" \\h \\z \\u`)."""
        self.add_paragraph(
            heading_title,
            align="center",
            first_line_indent=0.0 * mm,
            space_before=160,
            space_after=140,
            bold=True,
            font_half_points=28,
            keep_next=True,
        )
        toc_block = (
            "<w:p>"
            '<w:pPr><w:spacing w:before="0" w:after="80"/></w:pPr>'
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r><w:rPr><w:rFonts w:ascii="{escape(self.base_font_family)}" '
            f'w:hAnsi="{escape(self.base_font_family)}"/><w:i/></w:rPr>'
            "<w:t>Автоматическое оглавление (обновляется при открытии или экспорте в PDF)</w:t></w:r>"
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            "</w:p>"
        )
        self._body_blocks.append(toc_block)

    def add_equation(self, omml_inner: str, equation_number: int | str | None = None) -> None:
        """Добавляет выключную математическую формулу OMML по центру строки и опциональный номер справа."""
        total_twips = self.printable_twips
        num_width_twips = 800 if equation_number is not None else 0
        math_width_twips = total_twips - num_width_twips

        grid_num_xml = ""
        num_cell_xml = ""
        if equation_number is not None:
            grid_num_xml = f'<w:gridCol w:w="{num_width_twips}"/>'
            num_runs = render_rich_runs(f"({equation_number})", font_family=self.base_font_family)
            num_cell_xml = (
                f'<w:tc><w:tcPr><w:tcW w:w="{num_width_twips}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
                '<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:ind w:firstLine="0"/><w:jc w:val="right"/></w:pPr>'
                f"{num_runs}</w:p></w:tc>"
            )

        table_xml = (
            "<w:tbl>"
            "<w:tblPr>"
            f'<w:tblW w:w="{total_twips}" w:type="dxa"/>'
            '<w:jc w:val="center"/>'
            "<w:tblBorders>"
            '<w:top w:val="none"/><w:left w:val="none"/><w:bottom w:val="none"/>'
            '<w:right w:val="none"/><w:insideH w:val="none"/><w:insideV w:val="none"/>'
            "</w:tblBorders>"
            '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/>'
            '<w:left w:w="40" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
            "</w:tblPr>"
            f'<w:tblGrid><w:gridCol w:w="{math_width_twips}"/>{grid_num_xml}</w:tblGrid>'
            "<w:tr><w:trPr><w:cantSplit/></w:trPr>"
            f'<w:tc><w:tcPr><w:tcW w:w="{math_width_twips}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
            '<w:p><w:pPr><w:spacing w:before="40" w:after="40"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
            '<m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>'
            f"<m:oMath>{omml_inner}</m:oMath></m:oMathPara>"
            "</w:p></w:tc>"
            f"{num_cell_xml}"
            "</w:tr></w:tbl>"
        )
        self._body_blocks.append(table_xml)

    def add_latex_equation(self, latex_expression: str, equation_number: int | str | None = None) -> None:
        """Транслирует выражение LaTeX в нативный OMML и вставляет выключную формулу."""
        self.add_equation(latex_to_omml(latex_expression), equation_number=equation_number)

    def add_table(
        self,
        caption: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        column_weights: Sequence[float] | None = None,
        alignments: Sequence[str] | None = None,
        *,
        summary_rows: Sequence[int] = (),
        font_half_points: int = 19,
    ) -> None:
        """Добавляет академическую таблицу с повторяющейся шапкой и явной подсветкой итоговых строк `summary_rows`."""
        num_cols = len(headers)
        if num_cols == 0:
            raise ValueError("Список заголовков таблицы headers не может быть пустым.")

        resolved_weights = (
            tuple(float(wt) for wt in column_weights)
            if column_weights is not None
            else tuple(1.0 for _ in range(num_cols))
        )
        if len(resolved_weights) != num_cols or sum(resolved_weights) <= 0.0:
            raise ValueError("Длина column_weights должна совпадать с числом столбцов, а сумма быть положительной.")

        resolved_alignments = (
            tuple(alignments)
            if alignments is not None
            else ("left", *("center" for _ in range(num_cols - 1)))
        )
        if len(resolved_alignments) != num_cols:
            raise ValueError("Длина alignments должна совпадать с числом столбцов таблицы.")

        num_rows = len(rows)
        normalized_summary_indices = {
            (idx_val if idx_val >= 0 else num_rows + idx_val) for idx_val in summary_rows
        }

        if caption:
            self.add_paragraph(
                caption,
                align="left",
                first_line_indent=0.0 * mm,
                space_before=160,
                space_after=60,
                italic=True,
                font_half_points=22,
                keep_next=True,
            )

        total_twips = self.printable_twips
        weight_sum = sum(resolved_weights)
        col_twips = [round(total_twips * (wt / weight_sum)) for wt in resolved_weights]
        col_twips[-1] += total_twips - sum(col_twips)

        grid_cols_xml = "".join(f'<w:gridCol w:w="{width_val}"/>' for width_val in col_twips)

        header_cells_xml: list[str] = []
        for hdr_text, width_val in zip(headers, col_twips, strict=True):
            hdr_runs = render_rich_runs(
                hdr_text,
                default_bold=True,
                font_family=self.base_font_family,
                font_half_points=font_half_points,
            )
            header_cells_xml.append(
                "<w:tc><w:tcPr>"
                f'<w:tcW w:w="{width_val}" w:type="dxa"/>'
                '<w:shd w:val="clear" w:color="auto" w:fill="EAEFF5"/>'
                '<w:vAlign w:val="center"/>'
                "</w:tcPr>"
                '<w:p><w:pPr><w:spacing w:before="30" w:after="30" w:line="230" w:lineRule="auto"/>'
                '<w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
                f"{hdr_runs}</w:p></w:tc>"
            )
        header_row_xml = f"<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>{''.join(header_cells_xml)}</w:tr>"

        body_rows_xml: list[str] = []
        for row_idx, row_values in enumerate(rows):
            if len(row_values) != num_cols:
                raise ValueError(
                    f"Строка таблицы #{row_idx} содержит {len(row_values)} ячеек, ожидалось {num_cols}."
                )
            is_summary = row_idx in normalized_summary_indices
            shade_xml = '<w:shd w:val="clear" w:color="auto" w:fill="F5F7FA"/>' if is_summary else ""
            row_cells_xml: list[str] = []
            for cell_text, width_val, col_align in zip(row_values, col_twips, resolved_alignments, strict=True):
                sub_lines = str(cell_text).split("\n")
                cell_paras: list[str] = []
                for line_idx, line_str in enumerate(sub_lines):
                    cell_runs = render_rich_runs(
                        line_str,
                        default_bold=is_summary,
                        font_family=self.base_font_family,
                        font_half_points=font_half_points,
                    )
                    sp_before = "24" if line_idx == 0 else "0"
                    sp_after = "24" if line_idx == len(sub_lines) - 1 else "10"
                    cell_paras.append(
                        f'<w:p><w:pPr><w:spacing w:before="{sp_before}" w:after="{sp_after}" '
                        f'w:line="225" w:lineRule="auto"/><w:ind w:firstLine="0"/>'
                        f'<w:jc w:val="{escape(col_align)}"/></w:pPr>{cell_runs}</w:p>'
                    )
                row_cells_xml.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{width_val}" w:type="dxa"/>{shade_xml}'
                    f'<w:vAlign w:val="center"/></w:tcPr>{"".join(cell_paras)}</w:tc>'
                )
            body_rows_xml.append(f"<w:tr><w:trPr><w:cantSplit/></w:trPr>{''.join(row_cells_xml)}</w:tr>")

        table_xml = (
            "<w:tbl>"
            "<w:tblPr>"
            f'<w:tblW w:w="{total_twips}" w:type="dxa"/>'
            '<w:jc w:val="center"/>'
            "<w:tblBorders>"
            '<w:top w:val="single" w:sz="6" w:space="0" w:color="222222"/>'
            '<w:left w:val="single" w:sz="4" w:space="0" w:color="555555"/>'
            '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="222222"/>'
            '<w:right w:val="single" w:sz="4" w:space="0" w:color="555555"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
            '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
            "</w:tblBorders>"
            '<w:tblCellMar><w:top w:w="30" w:type="dxa"/><w:bottom w:w="30" w:type="dxa"/>'
            '<w:left w:w="65" w:type="dxa"/><w:right w:w="65" w:type="dxa"/></w:tblCellMar>'
            "</w:tblPr>"
            f"<w:tblGrid>{grid_cols_xml}</w:tblGrid>"
            f"{header_row_xml}{''.join(body_rows_xml)}"
            "</w:tbl>"
        )
        self._body_blocks.append(table_xml)
        self.add_paragraph(
            "",
            first_line_indent=0.0 * mm,
            space_before=0,
            space_after=60,
            line_spacing=120,
        )

    def add_figure(
        self,
        image_path: str | Path,
        caption: str,
        *,
        max_width: float = 169.0 * mm,
        max_height: float = 198.0 * mm,
        crop_whitespace: bool = True,
    ) -> None:
        """Встраивает растровую иллюстрацию с безопасным сведением прозрачности и подрисуночной подписью."""
        resolved_img = Path(image_path)
        if not resolved_img.is_file():
            raise FileNotFoundError(f"Файл иллюстрации не найден: {resolved_img}")
        if max_width <= 0.0 or max_height <= 0.0:
            raise ValueError("Максимальные размеры иллюстрации max_width и max_height должны быть положительными.")

        with Image.open(resolved_img) as opened_img:
            rgb_img = _flatten_image_onto_white(opened_img)
            if crop_whitespace:
                inverted = ImageOps.invert(rgb_img)
                bbox = inverted.getbbox()
                if bbox is not None:
                    crop_margin = 16
                    crop_left = max(0, bbox[0] - crop_margin)
                    crop_upper = max(0, bbox[1] - crop_margin)
                    crop_right = min(rgb_img.width, bbox[2] + crop_margin)
                    crop_lower = min(rgb_img.height, bbox[3] + crop_margin)
                    rgb_img = rgb_img.crop((crop_left, crop_upper, crop_right, crop_lower))
            px_width, px_height = rgb_img.size
            png_buffer = io.BytesIO()
            rgb_img.save(png_buffer, format="PNG", optimize=True)
            png_bytes = png_buffer.getvalue()

        aspect_ratio = float(px_height) / float(max(px_width, 1))
        target_width = float(max_width / mm)
        target_height = target_width * aspect_ratio
        max_height_val = float(max_height / mm)
        if target_height > max_height_val:
            target_height = max_height_val
            target_width = target_height / max(aspect_ratio, 1e-6)

        extent_cx = round(target_width * EMU_PER_MILLIMETER)
        extent_cy = round(target_height * EMU_PER_MILLIMETER)

        media_idx = len(self._media_entries) + 1
        rel_id = f"rIdImg{media_idx}"
        media_filename = f"image{media_idx}.png"
        self._media_entries.append((media_filename, png_bytes))

        drawing_id = self._drawing_counter
        self._drawing_counter += 1

        drawing_block = (
            "<w:p>"
            '<w:pPr><w:keepNext/><w:spacing w:before="140" w:after="50"/>'
            '<w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
            "<w:r><w:drawing>"
            '<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{extent_cx}" cy="{extent_cy}"/>'
            f'<wp:docPr id="{drawing_id}" name="Figure {drawing_id}"/>'
            "<wp:cNvGraphicFramePr>"
            '<a:graphicFrameLocks noChangeAspect="1"/>'
            "</wp:cNvGraphicFramePr>"
            "<a:graphic>"
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            "<pic:pic>"
            "<pic:nvPicPr>"
            f'<pic:cNvPr id="{drawing_id}" name="{media_filename}"/>'
            "<pic:cNvPicPr/>"
            "</pic:nvPicPr>"
            "<pic:blipFill>"
            f'<a:blip r:embed="{rel_id}"/>'
            "<a:stretch><a:fillRect/></a:stretch>"
            "</pic:blipFill>"
            "<pic:spPr>"
            "<a:xfrm>"
            '<a:off x="0" y="0"/>'
            f'<a:ext cx="{extent_cx}" cy="{extent_cy}"/>'
            "</a:xfrm>"
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            "</pic:spPr>"
            "</pic:pic>"
            "</a:graphicData>"
            "</a:graphic>"
            "</wp:inline>"
            "</w:drawing></w:r></w:p>"
        )
        self._body_blocks.append(drawing_block)

        if caption:
            self.add_paragraph(
                caption,
                align="center",
                first_line_indent=0.0 * mm,
                space_before=20,
                space_after=150,
                line_spacing=250,
                italic=True,
                font_half_points=21,
            )

    def save_docx(self, output_path: str | Path) -> Path:
        """Собирает и записывает валидный ZIP-архив OOXML `.docx` на диск."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        geom = self.geometry
        pg_w = round(float(geom.page_width / mm) * DXA_PER_MILLIMETER)
        pg_h = round(float(geom.page_height / mm) * DXA_PER_MILLIMETER)
        mar_top = round(float(geom.margin_top / mm) * DXA_PER_MILLIMETER)
        mar_right = round(float(geom.margin_right / mm) * DXA_PER_MILLIMETER)
        mar_bottom = round(float(geom.margin_bottom / mm) * DXA_PER_MILLIMETER)
        mar_left = round(float(geom.margin_left / mm) * DXA_PER_MILLIMETER)
        hdr_off = round(float(geom.header_offset / mm) * DXA_PER_MILLIMETER)
        ftr_off = round(float(geom.footer_offset / mm) * DXA_PER_MILLIMETER)

        sect_pr = (
            "<w:sectPr>"
            '<w:footerReference w:type="default" r:id="rIdFooter1"/>'
            f'<w:pgSz w:w="{pg_w}" w:h="{pg_h}"/>'
            f'<w:pgMar w:top="{mar_top}" w:right="{mar_right}" w:bottom="{mar_bottom}" w:left="{mar_left}" '
            f'w:header="{hdr_off}" w:footer="{ftr_off}" w:gutter="0"/>'
            "<w:titlePg/>"
            "</w:sectPr>"
        )

        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{WORD_NS}" xmlns:m="{MATH_NS}" xmlns:r="{REL_NS}" '
            f'xmlns:wp="{DRAWING_WP_NS}" xmlns:a="{DRAWING_MAIN_NS}" xmlns:pic="{DRAWING_PIC_NS}">'
            f"<w:body>{''.join(self._body_blocks)}{sect_pr}</w:body></w:document>"
        )

        content_types_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="png" ContentType="image/png"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            '<Override PartName="/word/settings.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
            '<Override PartName="/word/footer1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
            "</Types>"
        )

        root_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        )

        image_rels_xml = "".join(
            f'<Relationship Id="rIdImg{idx_rel}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{media_name}"/>'
            for idx_rel, (media_name, _) in enumerate(self._media_entries, start=1)
        )
        doc_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rIdStyles" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            '<Relationship Id="rIdSettings" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" '
            'Target="settings.xml"/>'
            '<Relationship Id="rIdFooter1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
            'Target="footer1.xml"/>'
            f"{image_rels_xml}</Relationships>"
        )

        fam_esc = escape(self.base_font_family)
        sz_str = str(self.base_font_half_points)
        styles_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:styles xmlns:w="{WORD_NS}">'
            "<w:docDefaults>"
            "<w:rPrDefault><w:rPr>"
            f'<w:rFonts w:ascii="{fam_esc}" w:hAnsi="{fam_esc}" w:cs="{fam_esc}"/>'
            f'<w:sz w:val="{sz_str}"/><w:szCs w:val="{sz_str}"/>'
            '<w:lang w:val="ru-RU" w:eastAsia="en-US" w:bidi="ar-SA"/>'
            "</w:rPr></w:rPrDefault>"
            "<w:pPrDefault><w:pPr>"
            '<w:spacing w:after="100" w:line="276" w:lineRule="auto"/>'
            "</w:pPr></w:pPrDefault>"
            "</w:docDefaults>"
            '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
            '<w:name w:val="Normal"/>'
            "</w:style>"
            '<w:style w:type="paragraph" w:styleId="Heading1">'
            '<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>'
            '<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="260" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>'
            '<w:rPr><w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>'
            "</w:style>"
            '<w:style w:type="paragraph" w:styleId="Heading2">'
            '<w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>'
            '<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="200" w:after="90"/><w:outlineLvl w:val="1"/></w:pPr>'
            '<w:rPr><w:b/><w:bCs/><w:sz w:val="25"/><w:szCs w:val="25"/></w:rPr>'
            "</w:style>"
            '<w:style w:type="paragraph" w:styleId="Heading3">'
            '<w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>'
            '<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="150" w:after="70"/><w:outlineLvl w:val="2"/></w:pPr>'
            '<w:rPr><w:b/><w:bCs/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
            "</w:style>"
            '<w:style w:type="paragraph" w:styleId="TOC1">'
            '<w:name w:val="toc 1"/><w:basedOn w:val="Normal"/>'
            '<w:pPr><w:spacing w:before="60" w:after="40"/><w:ind w:left="0" w:firstLine="0"/></w:pPr>'
            '<w:rPr><w:b/><w:sz w:val="23"/></w:rPr>'
            "</w:style>"
            '<w:style w:type="paragraph" w:styleId="TOC2">'
            '<w:name w:val="toc 2"/><w:basedOn w:val="Normal"/>'
            '<w:pPr><w:spacing w:before="20" w:after="20"/><w:ind w:left="340" w:firstLine="0"/></w:pPr>'
            '<w:rPr><w:sz w:val="22"/></w:rPr>'
            "</w:style>"
            '<w:style w:type="paragraph" w:styleId="TOC3">'
            '<w:name w:val="toc 3"/><w:basedOn w:val="Normal"/>'
            '<w:pPr><w:spacing w:before="10" w:after="10"/><w:ind w:left="680" w:firstLine="0"/></w:pPr>'
            '<w:rPr><w:sz w:val="21"/></w:rPr>'
            "</w:style>"
            "</w:styles>"
        )

        settings_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:settings xmlns:w="{WORD_NS}" xmlns:m="{MATH_NS}">'
            '<w:updateFields w:val="true"/>'
            "<m:mathPr>"
            '<m:mathFont m:val="Cambria Math"/>'
            '<m:brkBin m:val="before"/>'
            '<m:brkBinSub m:val="--"/>'
            '<m:smallFrac m:val="0"/>'
            "<m:dispDef/>"
            '<m:lMargin m:val="0"/>'
            '<m:rMargin m:val="0"/>'
            '<m:defJc m:val="centerGroup"/>'
            '<m:wrapIndent m:val="1440"/>'
            '<m:intLim m:val="subSup"/>'
            '<m:naryLim m:val="undOvr"/>'
            "</m:mathPr>"
            "</w:settings>"
        )

        footer_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:ftr xmlns:w="{WORD_NS}">'
            '<w:p><w:pPr><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r><w:rPr><w:rFonts w:ascii="{fam_esc}" w:hAnsi="{fam_esc}"/>'
            '<w:sz w:val="21"/><w:color w:val="444444"/></w:rPr><w:t xml:space="preserve">— </w:t></w:r>'
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            '<w:r><w:rPr><w:sz w:val="21"/><w:color w:val="444444"/></w:rPr><w:t>1</w:t></w:r>'
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            f'<w:r><w:rPr><w:rFonts w:ascii="{fam_esc}" w:hAnsi="{fam_esc}"/>'
            '<w:sz w:val="21"/><w:color w:val="444444"/></w:rPr><w:t xml:space="preserve"> —</w:t></w:r>'
            "</w:p></w:ftr>"
        )

        with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types_xml)
            archive.writestr("_rels/.rels", root_rels_xml)
            archive.writestr("word/_rels/document.xml.rels", doc_rels_xml)
            archive.writestr("word/document.xml", document_xml)
            archive.writestr("word/styles.xml", styles_xml)
            archive.writestr("word/settings.xml", settings_xml)
            archive.writestr("word/footer1.xml", footer_xml)
            for media_name, media_payload in self._media_entries:
                archive.writestr(f"word/media/{media_name}", media_payload)

        return out_file


__all__ = [
    "PageGeometryConfig",
    "ScientificDocxBuilder",
    "render_rich_runs",
]
