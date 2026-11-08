from app.db.models import DocumentFormat
from app.infra.token_count import TokenCounter
from app.knowledge.chunker import Chunker
from app.knowledge.parser import DocumentParser


def test_markdown_parser_preserves_section_paths() -> None:
    parsed = DocumentParser().parse("# 项目\n简介\n\n## 技术\n`event_id` 幂等", DocumentFormat.MARKDOWN)
    assert [item.section_path for item in parsed.sections] == [["项目"], ["项目", "技术"]]


def test_markdown_parser_retains_preamble_as_root_section() -> None:
    parsed = DocumentParser().parse("张三\n邮箱：a@example.com\n\n# 经历\n项目内容", DocumentFormat.MARKDOWN)
    assert parsed.sections[0].level == 0
    assert parsed.normalized_text[parsed.sections[0].start_offset : parsed.sections[0].end_offset].startswith("张三")
    assert parsed.sections[1].section_path == ["经历"]


def test_latex_parser_recognizes_headings() -> None:
    parsed = DocumentParser().parse("\\section{经历}\n内容\n\\subsection{支付}\n细节", DocumentFormat.LATEX)
    assert [item.section_path for item in parsed.sections] == [["经历"], ["经历", "支付"]]


def test_chunker_keeps_technical_terms_in_retrieval_text() -> None:
    parsed = DocumentParser().parse("# Project\n使用 event_id 完成幂等处理。", DocumentFormat.MARKDOWN)
    chunks = Chunker(TokenCounter()).chunk(parsed, "Resume")
    assert chunks[0].content.endswith("幂等处理。")
    assert "event_id" in chunks[0].retrieval_text


def test_chunker_never_exceeds_hard_token_limit_for_unbroken_text() -> None:
    counter = TokenCounter()
    parser = DocumentParser()
    parsed = parser.parse("x" * 12_000, DocumentFormat.TEXT)
    chunks = Chunker(counter, max_tokens=100).chunk(parsed, "Source")
    assert chunks and all(item.token_count <= 100 for item in chunks)
