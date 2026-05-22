"""
agents/report/tools/pdf_renderer.py
──────────────────────────────────────
Optional PDF renderer using WeasyPrint.

Converts a Markdown report string → HTML → PDF.
Guarded behind a lazy import — if WeasyPrint is not installed the module
still loads and render_pdf() raises a clear ImportError rather than
crashing the entire pipeline at import time.

Install: pip install ".[pdf]"   (requires WeasyPrint system deps)
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

_WEASYPRINT_AVAILABLE: bool | None = None  # None = not yet checked


def is_available() -> bool:
    """Return True if WeasyPrint is installed and usable."""
    global _WEASYPRINT_AVAILABLE
    if _WEASYPRINT_AVAILABLE is None:
        try:
            import weasyprint  # noqa: F401
            _WEASYPRINT_AVAILABLE = True
        except ImportError:
            _WEASYPRINT_AVAILABLE = False
    return _WEASYPRINT_AVAILABLE


def render_pdf(markdown_content: str, output_path: str) -> str:
    """
    Convert a Markdown string to a PDF file via WeasyPrint.

    Args:
        markdown_content: Full Markdown string (as produced by md_formatter).
        output_path:      Absolute path where the PDF should be written.

    Returns:
        The output_path string (for chaining).

    Raises:
        ImportError: WeasyPrint not installed.
        RuntimeError: Conversion failed.
    """
    if not is_available():
        raise ImportError(
            "WeasyPrint is not installed. "
            "Install it with: pip install 'multi-agent-pipeline[pdf]'"
        )

    try:
        import weasyprint
    except ImportError as e:
        raise ImportError(f"WeasyPrint import failed: {e}") from e

    html = _markdown_to_html(markdown_content)

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        weasyprint.HTML(string=html).write_pdf(output_path)
        size_kb = Path(output_path).stat().st_size // 1024
        logger.info("pdf_rendered", output_path=output_path, size_kb=size_kb)
        return output_path
    except Exception as e:
        raise RuntimeError(f"PDF rendering failed: {e}") from e


def _markdown_to_html(markdown: str) -> str:
    """
    Convert Markdown to HTML with embedded CSS for PDF rendering.

    Uses the `markdown` library (already a project dependency).
    Falls back to a minimal regex-based conversion if unavailable.
    """
    try:
        import markdown as md_lib
        body = md_lib.markdown(
            markdown,
            extensions=["tables", "fenced_code", "nl2br"],
        )
    except ImportError:
        body = _basic_markdown_to_html(markdown)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Pipeline Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 12pt;
    line-height: 1.6;
    color: #1a1a1a;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
  }}
  h1 {{ font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ font-size: 16pt; border-bottom: 1px solid #ccc; margin-top: 32px; }}
  h3 {{ font-size: 13pt; margin-top: 20px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 10pt;
    margin: 16px 0;
    page-break-inside: avoid;
  }}
  th {{ background: #f0f0f0; font-weight: 600; text-align: left; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  code {{
    font-family: 'Courier New', monospace;
    font-size: 10pt;
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
  }}
  pre {{
    background: #f5f5f5;
    padding: 12px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 9pt;
    page-break-inside: avoid;
  }}
  blockquote {{
    border-left: 4px solid #e0a000;
    background: #fffbf0;
    padding: 8px 16px;
    margin: 16px 0;
    border-radius: 0 4px 4px 0;
  }}
  @page {{ margin: 2cm; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _basic_markdown_to_html(markdown: str) -> str:
    """
    Minimal regex Markdown → HTML fallback.
    Handles headings, bold, code, and paragraphs only.
    Used only when the markdown library is unavailable.
    """
    lines = markdown.split("\n")
    html_lines = []
    in_code_block = False

    for line in lines:
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                html_lines.append("<pre><code>")
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(line)
            continue

        if line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- ") or line.startswith("* "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.startswith("|"):
            html_lines.append(f"<tr><td>{line.replace('|', '</td><td>').strip()}</td></tr>")
        elif line.strip():
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
            html_lines.append(f"<p>{line}</p>")
        else:
            html_lines.append("")

    return "\n".join(html_lines)
