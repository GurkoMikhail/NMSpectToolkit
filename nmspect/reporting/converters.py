"""Конвертер документов `.docx` в форматы `.pdf` и `.doc` с обновлением полей оглавления (MS Word COM / LibreOffice)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

POWERSHELL_WORD_CONVERT_SCRIPT: Final[str] = r"""
$ErrorActionPreference = 'Stop'
$docxPath = $env:NMSPECT_DOCX_INPUT
$pdfPath  = $env:NMSPECT_PDF_OUTPUT
$docPath  = $env:NMSPECT_DOC_OUTPUT

$wordApp = New-Object -ComObject Word.Application
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
try {
    $docObj = $wordApp.Documents.Open($docxPath)
    try {
        foreach ($toc in $docObj.TablesOfContents) {
            $toc.Update()
        }
        $docObj.Fields.Update() | Out-Null
        $docObj.Save()
        if (-not [string]::IsNullOrEmpty($pdfPath)) {
            $docObj.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
        }
        if (-not [string]::IsNullOrEmpty($docPath)) {
            $docObj.SaveAs([ref]$docPath, [ref]0)
        }
    } finally {
        $docObj.Close(0)
    }
} finally {
    $wordApp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null
}
"""


@dataclass(frozen=True)
class DocumentConversionResult:
    """Результат пакетного преобразования исходного `.docx` в производные форматы `.pdf` и `.doc`."""

    docx_path: Path
    pdf_path: Path | None
    legacy_doc_path: Path | None
    backend_used: str | None
    succeeded: bool
    message: str


class DocumentFormatConverter:
    """Преобразует файлы `.docx` в `.pdf` и бинарный `.doc` через установленный в системе офисный движок."""

    def __init__(self, timeout_duration: float = 180.0) -> None:
        """Инициализирует конвертер с предельным временем ожидания внешнего процесса."""
        if timeout_duration <= 0.0:
            raise ValueError(f"Таймаут конвертации должен быть положительным: {timeout_duration}.")
        self.timeout_duration: float = float(timeout_duration)

    @staticmethod
    def _find_libreoffice_binary() -> str | None:
        """Ищет исполняемый файл LibreOffice (`soffice`) в `PATH` и стандартных путях установки."""
        for cmd_name in ("soffice", "libreoffice"):
            found_bin = shutil.which(cmd_name)
            if found_bin is not None:
                return found_bin
        if sys.platform == "win32":
            standard_win_paths = (
                Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
                Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
            )
            for candidate in standard_win_paths:
                if candidate.is_file():
                    return str(candidate)
        return None

    @staticmethod
    def _is_word_com_available() -> bool:
        """Проверяет наличие зарегистрированного COM-класса `Word.Application` в ОС Windows."""
        if sys.platform != "win32":
            return False
        ps_exe = shutil.which("powershell") or shutil.which("pwsh")
        if ps_exe is None:
            return False
        probe_cmd = [
            ps_exe,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "if ($null -ne [Type]::GetTypeFromProgID('Word.Application')) { exit 0 } else { exit 1 }",
        ]
        completed = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        return completed.returncode == 0

    def available_backends(self) -> tuple[str, ...]:
        """Возвращает упорядоченный кортеж доступных в текущем окружении конвертеров (`'word_com'`, `'libreoffice'`)."""
        detected: list[str] = []
        if self._is_word_com_available():
            detected.append("word_com")
        if self._find_libreoffice_binary() is not None:
            detected.append("libreoffice")
        return tuple(detected)

    def convert_all(
        self,
        docx_path: str | Path,
        *,
        pdf_path: str | Path | None = None,
        legacy_doc_path: str | Path | None = None,
        strict: bool = False,
    ) -> DocumentConversionResult:
        """Обновляет поля `.docx` и экспортирует `.pdf` и/или `.doc` при наличии доступного бэкенда."""
        src_docx = Path(docx_path).resolve()
        if not src_docx.is_file():
            raise FileNotFoundError(f"Исходный документ .docx не найден: {src_docx}")

        target_pdf = Path(pdf_path).resolve() if pdf_path is not None else None
        target_doc = Path(legacy_doc_path).resolve() if legacy_doc_path is not None else None
        if target_pdf is None and target_doc is None:
            return DocumentConversionResult(
                docx_path=src_docx,
                pdf_path=None,
                legacy_doc_path=None,
                backend_used=None,
                succeeded=True,
                message="Дополнительные форматы экспорта не запрашивались.",
            )

        if target_pdf is not None:
            target_pdf.parent.mkdir(parents=True, exist_ok=True)
        if target_doc is not None:
            target_doc.parent.mkdir(parents=True, exist_ok=True)

        backends = self.available_backends()
        if not backends:
            msg = (
                "В окружении не обнаружены ни Microsoft Word (COM), ни LibreOffice (soffice); "
                "сохранён только нативный файл .docx."
            )
            if strict:
                raise RuntimeError(msg)
            return DocumentConversionResult(
                docx_path=src_docx,
                pdf_path=None,
                legacy_doc_path=None,
                backend_used=None,
                succeeded=False,
                message=msg,
            )

        if "word_com" in backends:
            return self._convert_via_word_com(src_docx, target_pdf, target_doc, strict=strict)
        return self._convert_via_libreoffice(src_docx, target_pdf, target_doc, strict=strict)

    def export_pdf(
        self,
        docx_path: str | Path,
        pdf_path: str | Path | None = None,
        *,
        strict: bool = False,
    ) -> DocumentConversionResult:
        """Экспортирует `.docx` в формат `.pdf` с автоматическим пересчётом оглавления и закладок."""
        src_docx = Path(docx_path).resolve()
        resolved_pdf = Path(pdf_path).resolve() if pdf_path is not None else src_docx.with_suffix(".pdf")
        return self.convert_all(src_docx, pdf_path=resolved_pdf, legacy_doc_path=None, strict=strict)

    def export_legacy_doc(
        self,
        docx_path: str | Path,
        legacy_doc_path: str | Path | None = None,
        *,
        strict: bool = False,
    ) -> DocumentConversionResult:
        """Экспортирует `.docx` в классический бинарный формат Word 97–2003 (`.doc`)."""
        src_docx = Path(docx_path).resolve()
        resolved_doc = (
            Path(legacy_doc_path).resolve() if legacy_doc_path is not None else src_docx.with_suffix(".doc")
        )
        return self.convert_all(src_docx, pdf_path=None, legacy_doc_path=resolved_doc, strict=strict)

    def _convert_via_word_com(
        self,
        src_docx: Path,
        target_pdf: Path | None,
        target_doc: Path | None,
        *,
        strict: bool,
    ) -> DocumentConversionResult:
        ps_exe = shutil.which("powershell") or shutil.which("pwsh")
        if ps_exe is None:
            if strict:
                raise RuntimeError("Исполняемый файл PowerShell не найден для вызова Word COM.")
            return DocumentConversionResult(
                docx_path=src_docx,
                pdf_path=None,
                legacy_doc_path=None,
                backend_used=None,
                succeeded=False,
                message="PowerShell недоступен.",
            )

        child_env = dict(os.environ)
        child_env["NMSPECT_DOCX_INPUT"] = str(src_docx)
        child_env["NMSPECT_PDF_OUTPUT"] = str(target_pdf) if target_pdf is not None else ""
        child_env["NMSPECT_DOC_OUTPUT"] = str(target_doc) if target_doc is not None else ""

        completed = subprocess.run(
            [ps_exe, "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_WORD_CONVERT_SCRIPT],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=self.timeout_duration,
            check=False,
        )
        if completed.returncode != 0:
            err_msg = f"Ошибка конвертации через Microsoft Word COM: {completed.stderr.strip()}"
            if strict:
                raise RuntimeError(err_msg)
            return DocumentConversionResult(
                docx_path=src_docx,
                pdf_path=None,
                legacy_doc_path=None,
                backend_used="word_com",
                succeeded=False,
                message=err_msg,
            )

        return DocumentConversionResult(
            docx_path=src_docx,
            pdf_path=target_pdf if target_pdf is not None and target_pdf.is_file() else None,
            legacy_doc_path=target_doc if target_doc is not None and target_doc.is_file() else None,
            backend_used="word_com",
            succeeded=True,
            message="Документ успешно обновлён и экспортирован через Microsoft Word COM.",
        )

    def _convert_via_libreoffice(
        self,
        src_docx: Path,
        target_pdf: Path | None,
        target_doc: Path | None,
        *,
        strict: bool,
    ) -> DocumentConversionResult:
        soffice_bin = self._find_libreoffice_binary()
        if soffice_bin is None:
            if strict:
                raise RuntimeError("Исполняемый файл LibreOffice не найден.")
            return DocumentConversionResult(
                docx_path=src_docx,
                pdf_path=None,
                legacy_doc_path=None,
                backend_used=None,
                succeeded=False,
                message="LibreOffice не найден.",
            )

        produced_pdf: Path | None = None
        produced_doc: Path | None = None

        for fmt_code, final_target in (("pdf", target_pdf), ("doc", target_doc)):
            if final_target is None:
                continue
            out_dir = final_target.parent
            cmd_args = [
                soffice_bin,
                "--headless",
                "--convert-to",
                fmt_code,
                "--outdir",
                str(out_dir),
                str(src_docx),
            ]
            completed = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=self.timeout_duration,
                check=False,
            )
            if completed.returncode != 0:
                err_msg = f"Ошибка конвертации LibreOffice ({fmt_code}): {completed.stderr.strip()}"
                if strict:
                    raise RuntimeError(err_msg)
                return DocumentConversionResult(
                    docx_path=src_docx,
                    pdf_path=produced_pdf,
                    legacy_doc_path=produced_doc,
                    backend_used="libreoffice",
                    succeeded=False,
                    message=err_msg,
                )
            default_out = out_dir / f"{src_docx.stem}.{fmt_code}"
            if default_out.is_file() and default_out.resolve() != final_target.resolve():
                default_out.replace(final_target)
            if fmt_code == "pdf" and final_target.is_file():
                produced_pdf = final_target
            elif fmt_code == "doc" and final_target.is_file():
                produced_doc = final_target

        return DocumentConversionResult(
            docx_path=src_docx,
            pdf_path=produced_pdf,
            legacy_doc_path=produced_doc,
            backend_used="libreoffice",
            succeeded=True,
            message="Документ успешно экспортирован через LibreOffice.",
        )


__all__ = [
    "DocumentConversionResult",
    "DocumentFormatConverter",
]
