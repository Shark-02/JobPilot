from __future__ import annotations

import re
from dataclasses import dataclass

from app.db.models import DocumentFormat

PARSER_VERSION = "v1"


@dataclass(frozen=True)
class ParsedSection:
    heading: str | None
    level: int
    section_path: list[str]
    ordinal: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class ParsedDocument:
    normalized_text: str
    sections: list[ParsedSection]


class DocumentParser:
    def parse(self, content: str, format: DocumentFormat) -> ParsedDocument:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            raise ValueError("Document content cannot be empty")
        if format is DocumentFormat.MARKDOWN:
            return self._parse_markdown(normalized)
        if format is DocumentFormat.LATEX:
            return self._parse_latex(normalized)
        return ParsedDocument(normalized, [ParsedSection(None, 0, [], 0, 0, len(normalized))])

    def _parse_markdown(self, text: str) -> ParsedDocument:
        headings = list(re.finditer(r"(?m)^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", text))
        if not headings:
            return ParsedDocument(text, [ParsedSection(None, 0, [], 0, 0, len(text))])
        levels: list[str | None] = [None] * 7
        sections: list[ParsedSection] = []
        preamble_end = headings[0].start()
        if text[:preamble_end].strip():
            sections.append(ParsedSection(None, 0, [], 0, 0, preamble_end))
        for heading_index, match in enumerate(headings):
            ordinal = len(sections)
            level = len(match.group(1))
            heading = match.group(2).strip()
            levels[level] = heading
            for index in range(level + 1, 7):
                levels[index] = None
            end = headings[heading_index + 1].start() if heading_index + 1 < len(headings) else len(text)
            sections.append(
                ParsedSection(heading, level, [item for item in levels[1 : level + 1] if item], ordinal, match.start(), end)
            )
        return ParsedDocument(text, sections)

    def _parse_latex(self, text: str) -> ParsedDocument:
        cleaned = re.sub(r"(?m)^\s*%.*$", "", text)
        levels = {"section": "#", "subsection": "##", "subsubsection": "###"}
        cleaned = re.sub(
            r"\\(section|subsection|subsubsection)\*?\{([^{}]+)\}",
            lambda match: f"{levels[match.group(1)]} {match.group(2)}",
            cleaned,
        )
        cleaned = re.sub(r"\\(?:textbf|textit|emph|href)\{([^{}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\(?:item|newline|par)\b", "\n", cleaned)
        cleaned = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?\{?", "", cleaned)
        cleaned = re.sub(r"[{}]", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return self._parse_markdown(cleaned)
