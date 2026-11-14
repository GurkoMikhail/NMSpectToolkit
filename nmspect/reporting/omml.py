"""Генератор узлов Office Math Markup Language (OMML) и транслятор формул LaTeX в нативный XML Microsoft Word."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final
from xml.sax.saxutils import escape

LATEX_SYMBOL_MAP: Final[dict[str, str]] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "ϕ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "ne": "≠",
    "neq": "≠",
    "approx": "≈",
    "sim": "∼",
    "simeq": "≃",
    "equiv": "≡",
    "pm": "±",
    "mp": "∓",
    "cdot": "⋅",
    "times": "×",
    "div": "÷",
    "ast": "∗",
    "star": "⋆",
    "circ": "∘",
    "bullet": "•",
    "oplus": "⊕",
    "otimes": "⊗",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "cup": "∪",
    "cap": "∩",
    "sqcup": "⊔",
    "emptyset": "∅",
    "varnothing": "∅",
    "forall": "∀",
    "exists": "∃",
    "nabla": "∇",
    "partial": "∂",
    "infty": "∞",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "Rightarrow": "⇒",
    "Leftrightarrow": "⇔",
    "mapsto": "↦",
    "perp": "⊥",
    "parallel": "∥",
    "angle": "∠",
    "degree": "°",
    "circ_deg": "°",
    "dots": "…",
    "ldots": "…",
    "cdots": "⋯",
    "prime": "′",
    "langle": "⟨",
    "rangle": "⟩",
    "mid": "∣",
}

LATEX_FUNCTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "ln",
        "log",
        "exp",
        "min",
        "max",
        "sup",
        "inf",
        "arg",
        "det",
        "dim",
        "lim",
    }
)

LATEX_NARY_MAP: Final[dict[str, str]] = {
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "oint": "∮",
    "bigcup": "⋃",
    "bigcap": "⋂",
    "bigsqcup": "⨆",
    "bigoplus": "⨁",
}

UPPERCASE_GREEK_SET: Final[frozenset[str]] = frozenset(
    {"Γ", "Δ", "Θ", "Λ", "Ξ", "Π", "Σ", "Υ", "Φ", "Ψ", "Ω"}
)


def math_run(text: str, *, italic: bool = True, bold: bool = False) -> str:
    """Создаёт математический прогон `<m:r>` со шрифтом `Cambria Math`."""
    if bold and italic:
        style_code = "bi"
    elif bold:
        style_code = "b"
    elif italic:
        style_code = "i"
    else:
        style_code = "p"
    return (
        f'<m:r><m:rPr><m:sty m:val="{style_code}"/></m:rPr>'
        '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="Cambria Math"/></w:rPr>'
        f"<m:t>{escape(text)}</m:t></m:r>"
    )


def math_operator(text: str) -> str:
    """Создаёт прямой математический оператор, число или знак препинания."""
    return math_run(text, italic=False, bold=False)


def math_text(text: str) -> str:
    """Создаёт прямой текстовый фрагмент внутри формулы (`<m:nor/>`, сохраняющий пробелы)."""
    return (
        '<m:r><m:rPr><m:nor/><m:sty m:val="p"/></m:rPr>'
        '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="Cambria Math"/></w:rPr>'
        f'<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'
    )


def math_vector(symbol: str) -> str:
    """Создаёт полужирный прямой символ вектора или матрицы."""
    return math_run(symbol, italic=False, bold=True)


def math_sub(base_xml: str, sub_xml: str) -> str:
    """Создаёт узел нижнего индекса `<m:sSub>`."""
    return f"<m:sSub><m:e>{base_xml}</m:e><m:sub>{sub_xml}</m:sub></m:sSub>"


def math_sup(base_xml: str, sup_xml: str) -> str:
    """Создаёт узел верхнего индекса `<m:sSup>`."""
    return f"<m:sSup><m:e>{base_xml}</m:e><m:sup>{sup_xml}</m:sup></m:sSup>"


def math_subsup(base_xml: str, sub_xml: str, sup_xml: str) -> str:
    """Создаёт узел одновременного нижнего и верхнего индекса `<m:sSubSup>`."""
    return f"<m:sSubSup><m:e>{base_xml}</m:e><m:sub>{sub_xml}</m:sub><m:sup>{sup_xml}</m:sup></m:sSubSup>"


def math_fraction(numerator_xml: str, denominator_xml: str) -> str:
    """Создаёт вертикальную дробь `<m:f>`."""
    return f"<m:f><m:num>{numerator_xml}</m:num><m:den>{denominator_xml}</m:den></m:f>"


def math_radical(argument_xml: str, degree_xml: str | None = None) -> str:
    """Создаёт квадратный корень или радикал произвольной степени `<m:rad>`."""
    if degree_xml is None:
        return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{argument_xml}</m:e></m:rad>'
    return f"<m:rad><m:deg>{degree_xml}</m:deg><m:e>{argument_xml}</m:e></m:rad>"


def math_delimiter(inner_xml: str, begin_char: str = "(", end_char: str = ")") -> str:
    """Создаёт масштабируемые скобки или модульные черты `<m:d>`."""
    return (
        f'<m:d><m:dPr><m:begChr m:val="{escape(begin_char)}"/><m:endChr m:val="{escape(end_char)}"/></m:dPr>'
        f"<m:e>{inner_xml}</m:e></m:d>"
    )


def math_nary(
    body_xml: str,
    *,
    operator_char: str = "∑",
    sub_xml: str | None = None,
    sup_xml: str | None = None,
) -> str:
    """Создаёт крупный оператор суммы, интеграла или объединения `<m:nary>`."""
    sub_hide = '<m:subHide m:val="1"/>' if sub_xml is None else ""
    sup_hide = '<m:supHide m:val="1"/>' if sup_xml is None else ""
    limit_mode = "subSup" if operator_char in ("∫", "∮") else "undOvr"
    return (
        f'<m:nary><m:naryPr><m:chr m:val="{escape(operator_char)}"/>'
        f'<m:limLoc m:val="{limit_mode}"/>{sub_hide}{sup_hide}</m:naryPr>'
        f"<m:sub>{sub_xml or ''}</m:sub><m:sup>{sup_xml or ''}</m:sup>"
        f"<m:e>{body_xml}</m:e></m:nary>"
    )


def math_accent(inner_xml: str, accent_char: str = "\u0302") -> str:
    """Создаёт диакритический знак над выражением (`^`, `~`, стрелка вектора)."""
    return f'<m:acc><m:accPr><m:chr m:val="{accent_char}"/></m:accPr><m:e>{inner_xml}</m:e></m:acc>'


def math_overline(inner_xml: str) -> str:
    """Создаёт горизонтальную черту усреднения сверху `<m:bar>`."""
    return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{inner_xml}</m:e></m:bar>'


def math_column_vector(entries: Sequence[str], begin_char: str = "(", end_char: str = ")") -> str:
    """Создаёт вектор-столбец в масштабируемых скобках `<m:d><m:m>...</m:m></m:d>`."""
    rows_xml = "".join(f"<m:mr><m:e>{entry}</m:e></m:mr>" for entry in entries)
    matrix_xml = (
        '<m:m><m:mPr><m:mcs><m:mc><m:mcPr><m:count m:val="1"/><m:mcJc m:val="center"/>'
        f"</m:mcPr></m:mc></m:mcs></m:mPr>{rows_xml}</m:m>"
    )
    return math_delimiter(matrix_xml, begin_char=begin_char, end_char=end_char)


def math_equation_array(lines: Sequence[str]) -> str:
    """Создаёт многострочный выровненный математический блок `<m:eqArr>`."""
    items_xml = "".join(f"<m:e>{line_item}</m:e>" for line_item in lines)
    return f"<m:eqArr>{items_xml}</m:eqArr>"


def inline_math(omml_inner: str) -> str:
    """Оборачивает внутренний XML узлов OMML в контейнер `<m:oMath>` для вставки в абзац."""
    return f"<m:oMath>{omml_inner}</m:oMath>"


class _LatexStreamParser:
    """Рекурсивный парсер математического подмножества LaTeX в узлы OMML."""

    def __init__(self, source: str) -> None:
        self.source: str = source
        self.length: int = len(source)
        self.cursor: int = 0

    def _skip_spaces(self) -> None:
        while self.cursor < self.length and self.source[self.cursor].isspace():
            self.cursor += 1

    def _peek(self) -> str:
        if self.cursor >= self.length:
            return ""
        return self.source[self.cursor]

    def _read_command_name(self) -> str:
        start_pos = self.cursor
        if self.cursor < self.length and not self.source[self.cursor].isalpha():
            char_tok = self.source[self.cursor]
            self.cursor += 1
            return char_tok
        while self.cursor < self.length and self.source[self.cursor].isalpha():
            self.cursor += 1
        return self.source[start_pos : self.cursor]

    def _read_raw_braced_text(self) -> str:
        self._skip_spaces()
        if self._peek() != "{":
            if self.cursor < self.length:
                single_char = self.source[self.cursor]
                self.cursor += 1
                return single_char
            return ""
        self.cursor += 1
        depth = 1
        chunks: list[str] = []
        while self.cursor < self.length and depth > 0:
            curr_ch = self.source[self.cursor]
            self.cursor += 1
            if curr_ch == "{":
                depth += 1
                chunks.append(curr_ch)
            elif curr_ch == "}":
                depth -= 1
                if depth > 0:
                    chunks.append(curr_ch)
            else:
                chunks.append(curr_ch)
        return "".join(chunks)

    def _read_optional_bracket_xml(self) -> str | None:
        self._skip_spaces()
        if self._peek() != "[":
            return None
        self.cursor += 1
        inner_xml = self.parse_sequence(stop_tokens=("]",))
        if self._peek() == "]":
            self.cursor += 1
        return inner_xml

    def _read_required_group_xml(self) -> str:
        self._skip_spaces()
        if self._peek() == "{":
            self.cursor += 1
            inner_xml = self.parse_sequence(stop_tokens=("}",))
            if self._peek() == "}":
                self.cursor += 1
            return inner_xml
        atom_xml = self._parse_primary_atom()
        return atom_xml if atom_xml is not None else ""

    def _read_delimiter_token(self) -> str:
        self._skip_spaces()
        if self.cursor >= self.length:
            return ""
        curr_ch = self.source[self.cursor]
        if curr_ch == "\\":
            self.cursor += 1
            cmd_name = self._read_command_name()
            delim_lookup = {
                "{": "{",
                "}": "}",
                "|": "‖",
                "Vert": "‖",
                "vert": "|",
                "langle": "⟨",
                "rangle": "⟩",
                "lfloor": "⌊",
                "rfloor": "⌋",
                "lceil": "⌈",
                "rceil": "⌉",
            }
            return delim_lookup.get(cmd_name, "")
        self.cursor += 1
        return "" if curr_ch == "." else curr_ch

    def _peek_right_delimiter(self) -> bool:
        saved_pos = self.cursor
        self._skip_spaces()
        if self.source.startswith(r"\right", self.cursor):
            after_pos = self.cursor + 6
            if after_pos >= self.length or not self.source[after_pos].isalpha():
                self.cursor = saved_pos
                return True
        self.cursor = saved_pos
        return False

    def parse_sequence(self, stop_tokens: tuple[str, ...] = (), stop_on_right: bool = False) -> str:
        """Разбирает последовательность математических термов до указанного ограничителя."""
        nodes: list[str] = []
        while self.cursor < self.length:
            self._skip_spaces()
            if self.cursor >= self.length:
                break
            if stop_on_right and self._peek_right_delimiter():
                break
            curr_ch = self._peek()
            if curr_ch in stop_tokens:
                break
            term_xml = self._parse_scripted_term(stop_tokens=stop_tokens, stop_on_right=stop_on_right)
            if term_xml:
                nodes.append(term_xml)
        return "".join(nodes)

    def _parse_scripted_term(self, stop_tokens: tuple[str, ...], stop_on_right: bool) -> str:
        base_info = self._parse_base_or_nary()
        if base_info is None:
            return ""

        kind, base_val = base_info
        sub_xml: str | None = None
        sup_xml: str | None = None

        while self.cursor < self.length:
            self._skip_spaces()
            marker = self._peek()
            if marker == "_" and sub_xml is None:
                self.cursor += 1
                sub_xml = self._read_required_group_xml()
            elif marker == "^" and sup_xml is None:
                self.cursor += 1
                sup_xml = self._read_required_group_xml()
            elif marker == "'" and sup_xml is None:
                primes: list[str] = []
                while self._peek() == "'":
                    self.cursor += 1
                    primes.append("′")
                sup_xml = math_operator("".join(primes))
            else:
                break

        if kind == "nary":
            body_xml = self._parse_nary_operand(stop_tokens=stop_tokens, stop_on_right=stop_on_right)
            return math_nary(body_xml, operator_char=base_val, sub_xml=sub_xml, sup_xml=sup_xml)

        if sub_xml is not None and sup_xml is not None:
            return math_subsup(base_val, sub_xml, sup_xml)
        if sub_xml is not None:
            return math_sub(base_val, sub_xml)
        if sup_xml is not None:
            return math_sup(base_val, sup_xml)
        return base_val

    def _parse_nary_operand(self, stop_tokens: tuple[str, ...], stop_on_right: bool) -> str:
        """Читает подынтегральное или суммируемое выражение для крупного оператора `<m:nary>`."""
        self._skip_spaces()
        if self.cursor >= self.length:
            return math_operator("")
        if stop_on_right and self._peek_right_delimiter():
            return math_operator("")
        if self._peek() in stop_tokens:
            return math_operator("")
        return self._parse_scripted_term(stop_tokens=stop_tokens, stop_on_right=stop_on_right)

    def _parse_base_or_nary(self) -> tuple[str, str] | None:
        self._skip_spaces()
        if self.cursor >= self.length:
            return None
        if self.source[self.cursor] == "\\":
            saved_pos = self.cursor
            self.cursor += 1
            cmd_name = self._read_command_name()
            if cmd_name in LATEX_NARY_MAP:
                return ("nary", LATEX_NARY_MAP[cmd_name])
            self.cursor = saved_pos
        atom_xml = self._parse_primary_atom()
        if atom_xml is None:
            return None
        return ("atom", atom_xml)

    def _parse_primary_atom(self) -> str | None:
        self._skip_spaces()
        if self.cursor >= self.length:
            return None

        curr_ch = self.source[self.cursor]
        if curr_ch == "{":
            self.cursor += 1
            inner_xml = self.parse_sequence(stop_tokens=("}",))
            if self._peek() == "}":
                self.cursor += 1
            return inner_xml

        if curr_ch == "\\":
            self.cursor += 1
            cmd_name = self._read_command_name()
            return self._handle_command(cmd_name)

        if curr_ch.isdigit():
            start_pos = self.cursor
            while self.cursor < self.length and (
                self.source[self.cursor].isdigit() or self.source[self.cursor] in (".", ",")
            ):
                if (
                    self.source[self.cursor] in (".", ",")
                    and (self.cursor + 1 >= self.length or not self.source[self.cursor + 1].isdigit())
                ):
                    break
                self.cursor += 1
            return math_operator(self.source[start_pos : self.cursor])

        if curr_ch.isalpha():
            self.cursor += 1
            is_italic = curr_ch not in UPPERCASE_GREEK_SET
            return math_run(curr_ch, italic=is_italic, bold=False)

        self.cursor += 1
        if curr_ch == "-":
            return math_operator("−")
        return math_operator(curr_ch)

    def _handle_command(self, cmd_name: str) -> str:
        if cmd_name in ("frac", "dfrac", "tfrac"):
            num_xml = self._read_required_group_xml()
            den_xml = self._read_required_group_xml()
            return math_fraction(num_xml, den_xml)

        if cmd_name == "sqrt":
            deg_xml = self._read_optional_bracket_xml()
            arg_xml = self._read_required_group_xml()
            return math_radical(arg_xml, degree_xml=deg_xml)

        if cmd_name in ("text", "textrm", "mathrm", "operatorname"):
            raw_txt = self._read_raw_braced_text()
            return math_text(raw_txt)

        if cmd_name in ("mathbf", "boldsymbol", "bm"):
            raw_txt = self._read_raw_braced_text()
            if raw_txt.startswith("\\"):
                sym_key = raw_txt.lstrip("\\").strip()
                resolved_sym = LATEX_SYMBOL_MAP.get(sym_key, sym_key)
                return math_vector(resolved_sym)
            return math_vector(raw_txt)

        if cmd_name == "vec":
            inner_xml = self._read_required_group_xml()
            return math_accent(inner_xml, accent_char="\u20d7")

        if cmd_name in ("hat", "widehat"):
            inner_xml = self._read_required_group_xml()
            return math_accent(inner_xml, accent_char="\u0302")

        if cmd_name in ("tilde", "widetilde"):
            inner_xml = self._read_required_group_xml()
            return math_accent(inner_xml, accent_char="\u0303")

        if cmd_name in ("bar", "overline"):
            inner_xml = self._read_required_group_xml()
            return math_overline(inner_xml)

        if cmd_name == "left":
            beg_chr = self._read_delimiter_token()
            inner_xml = self.parse_sequence(stop_on_right=True)
            end_chr = ""
            self._skip_spaces()
            if self.source.startswith(r"\right", self.cursor):
                self.cursor += 6
                end_chr = self._read_delimiter_token()
            return math_delimiter(inner_xml, begin_char=beg_chr, end_char=end_chr)

        if cmd_name in (",", ";", ":", "!", "quad", "qquad", " "):
            space_str = "  " if cmd_name in ("quad", "qquad") else " "
            return math_text(space_str)

        if cmd_name in ("{", "}", "%", "_", "&", "#"):
            return math_operator(cmd_name)

        if cmd_name in ("|", "Vert"):
            return math_operator("‖")

        if cmd_name in LATEX_FUNCTION_NAMES:
            return math_operator(cmd_name)

        if cmd_name in LATEX_SYMBOL_MAP:
            sym_val = LATEX_SYMBOL_MAP[cmd_name]
            if len(sym_val) == 1 and sym_val.isalpha():
                is_italic = sym_val not in UPPERCASE_GREEK_SET
                return math_run(sym_val, italic=is_italic, bold=False)
            return math_operator(sym_val)

        return math_text(cmd_name)


def latex_to_omml(expression: str) -> str:
    """Транслирует математическое выражение LaTeX во внутренний XML узлов OMML (`Cambria Math`)."""
    cleaned = expression.strip()
    if cleaned.startswith("$$") and cleaned.endswith("$$") and len(cleaned) >= 4:
        cleaned = cleaned[2:-2].strip()
    elif cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    if not cleaned:
        return math_operator("")
    parser = _LatexStreamParser(cleaned)
    return parser.parse_sequence()


__all__ = [
    "LATEX_FUNCTION_NAMES",
    "LATEX_NARY_MAP",
    "LATEX_SYMBOL_MAP",
    "inline_math",
    "latex_to_omml",
    "math_accent",
    "math_column_vector",
    "math_delimiter",
    "math_equation_array",
    "math_fraction",
    "math_nary",
    "math_operator",
    "math_overline",
    "math_radical",
    "math_run",
    "math_sub",
    "math_subsup",
    "math_sup",
    "math_text",
    "math_vector",
]
