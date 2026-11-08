from __future__ import annotations

import re
from dataclasses import dataclass

from app.infra.token_count import TokenCounter
from app.knowledge.parser import ParsedDocument


CHUNKER_VERSION = "v2"


@dataclass(frozen=True)
class ChunkDraft:
    section_ordinal: int
    ordinal: int
    content: str
    retrieval_text: str
    start_offset: int
    end_offset: int
    token_count: int


@dataclass(frozen=True)
class _Span:
    """
    A continuous span inside document.normalized_text.

    start/end are absolute offsets in normalized_text.
    The represented text must always be obtainable with:

        normalized_text[start:end]

    We deliberately keep offsets instead of reconstructing strings and
    searching them back with str.find(), because whitespace normalization
    would otherwise make source locations unreliable.
    """

    start: int
    end: int


class Chunker:
    def __init__(
        self,
        counter: TokenCounter,
        target_tokens: int = 600,
        min_tokens: int = 180,
        max_tokens: int = 1000,
    ) -> None:
        # A caller that narrows the hard ceiling should not have to restate the
        # two soft defaults.  Clamp them coherently; the hard limit remains
        # authoritative for every emitted chunk.
        target_tokens = min(target_tokens, max_tokens)
        min_tokens = min(min_tokens, target_tokens)
        if min_tokens <= 0:
            raise ValueError("min_tokens must be positive")
        if target_tokens < min_tokens:
            raise ValueError("target_tokens must be >= min_tokens")

        self.counter = counter
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def chunk(
        self,
        document: ParsedDocument,
        title: str,
    ) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []

        for section in document.sections:
            spans = self._chunk_section(
                document.normalized_text,
                section.start_offset,
                section.end_offset,
            )

            prefix = " > ".join(
                part
                for part in [title, *section.section_path]
                if part
            )

            for span in spans:
                content = document.normalized_text[span.start : span.end]

                # _trim_span() should already guarantee this, but keep the
                # invariant explicit.
                if not content.strip():
                    continue

                token_count = self.counter.count(content)

                drafts.append(
                    ChunkDraft(
                        section_ordinal=section.ordinal,
                        ordinal=len(drafts),
                        content=content,
                        retrieval_text=f"{prefix}\n{content}",
                        start_offset=span.start,
                        end_offset=span.end,
                        token_count=token_count,
                    )
                )

        return drafts

    def _chunk_section(
        self,
        text: str,
        section_start: int,
        section_end: int,
    ) -> list[_Span]:
        section_span = self._trim_span(text, section_start, section_end)

        if section_span is None:
            return []

        section_tokens = self._count_span(text, section_span)

        # A semantically complete section below the hard maximum is kept
        # intact, even when it is smaller than min_tokens.
        #
        # min_tokens is a soft lower bound inside a section, not a reason to
        # merge unrelated sections.
        if section_tokens <= self.max_tokens:
            return [section_span]

        blocks = self._paragraph_spans(
            text,
            section_span.start,
            section_span.end,
        )

        atomic_spans: list[_Span] = []

        for block in blocks:
            if self._count_span(text, block) <= self.max_tokens:
                atomic_spans.append(block)
            else:
                atomic_spans.extend(
                    self._hard_split_span(text, block)
                )

        if not atomic_spans:
            return []

        return self._pack_spans(text, atomic_spans)

    def _paragraph_spans(
        self,
        text: str,
        start: int,
        end: int,
    ) -> list[_Span]:
        """
        Split a section on blank lines while preserving exact source offsets.

        We do not use re.split() here because re.split() loses the positions of
        separators and later reconstructing paragraphs with "\\n\\n" changes
        the source text.
        """

        segment = text[start:end]

        # Two or more line breaks, optionally containing horizontal whitespace
        # on the blank line(s), are considered paragraph boundaries.
        separator = re.compile(r"\n[ \t]*\n+")

        spans: list[_Span] = []
        cursor = 0

        for match in separator.finditer(segment):
            relative_start = cursor
            relative_end = match.start()

            span = self._trim_span(
                text,
                start + relative_start,
                start + relative_end,
            )

            if span is not None:
                spans.append(span)

            cursor = match.end()

        tail = self._trim_span(
            text,
            start + cursor,
            end,
        )

        if tail is not None:
            spans.append(tail)

        return spans

    def _pack_spans(
        self,
        text: str,
        spans: list[_Span],
    ) -> list[_Span]:
        """
        Pack atomic spans into chunks.

        target_tokens is a SOFT target.
        max_tokens is the HARD limit.

        Important rule:

        If the current chunk is still below min_tokens, do not flush it merely
        because adding the next block would exceed target_tokens. Let it grow
        past target_tokens as long as it remains <= max_tokens.
        """

        output: list[_Span] = []

        current_start = spans[0].start
        current_end = spans[0].end

        for span in spans[1:]:
            current = _Span(current_start, current_end)
            combined = _Span(current_start, span.end)

            current_tokens = self._count_span(text, current)
            combined_tokens = self._count_span(text, combined)

            # Still under the preferred target: keep accumulating.
            if combined_tokens <= self.target_tokens:
                current_end = span.end
                continue

            # target_tokens is only a soft target.
            #
            # If the current chunk is still too small, keep the next block as
            # long as doing so does not violate max_tokens.
            if (
                current_tokens < self.min_tokens
                and combined_tokens <= self.max_tokens
            ):
                current_end = span.end
                continue

            # Current chunk is already large enough to stand alone.
            if current_tokens >= self.min_tokens:
                output.append(current)
                current_start = span.start
                current_end = span.end
                continue

            # Rare fallback:
            #
            # current < min_tokens but adding the next atomic block would exceed
            # the hard maximum. The atomic blocks themselves are already <=
            # max_tokens, so we have no legal way to combine them.
            output.append(current)
            current_start = span.start
            current_end = span.end

        tail = _Span(current_start, current_end)

        # Avoid an avoidable tiny tail. Prefer merging it back into the previous
        # chunk when the merged chunk remains inside max_tokens.
        if output and self._count_span(text, tail) < self.min_tokens:
            previous = output[-1]
            merged = _Span(previous.start, tail.end)

            if self._count_span(text, merged) <= self.max_tokens:
                output[-1] = merged
                return output

        output.append(tail)

        return output

    def _hard_split_span(
        self,
        text: str,
        span: _Span,
    ) -> list[_Span]:
        """
        Split one atomic block that itself exceeds max_tokens.

        Prefer sentence/newline boundaries first. If a single sentence is still
        too large, fall back to an exact character-range split.

        All returned pieces point directly into normalized_text, so offsets stay
        valid.
        """

        sentence_spans = self._sentence_spans(
            text,
            span.start,
            span.end,
        )

        pieces: list[_Span] = []

        for sentence in sentence_spans:
            if self._count_span(text, sentence) <= self.max_tokens:
                pieces.append(sentence)
            else:
                pieces.extend(
                    self._split_span_by_token_limit(
                        text,
                        sentence,
                        self.max_tokens,
                    )
                )

        if not pieces:
            return self._split_span_by_token_limit(
                text,
                span,
                self.max_tokens,
            )

        return self._pack_hard_split_pieces(text, pieces)

    def _sentence_spans(
        self,
        text: str,
        start: int,
        end: int,
    ) -> list[_Span]:
        """
        Produce approximate sentence spans without rewriting text.

        Sentence boundaries:
        - newline
        - punctuation followed by whitespace
        """

        segment = text[start:end]

        boundary = re.compile(
            r"(?<=[。！？.!?])(?:[ \t]+|\n+)|\n+"
        )

        spans: list[_Span] = []
        cursor = 0

        for match in boundary.finditer(segment):
            span = self._trim_span(
                text,
                start + cursor,
                start + match.start(),
            )

            if span is not None:
                spans.append(span)

            cursor = match.end()

        tail = self._trim_span(
            text,
            start + cursor,
            end,
        )

        if tail is not None:
            spans.append(tail)

        return spans

    def _pack_hard_split_pieces(
        self,
        text: str,
        pieces: list[_Span],
    ) -> list[_Span]:
        """
        Re-pack sentence-sized pieces up to max_tokens.

        This avoids creating one tiny chunk per sentence after a hard split.
        """

        if not pieces:
            return []

        output: list[_Span] = []

        current_start = pieces[0].start
        current_end = pieces[0].end

        for piece in pieces[1:]:
            combined = _Span(current_start, piece.end)

            if self._count_span(text, combined) <= self.max_tokens:
                current_end = piece.end
                continue

            output.append(
                _Span(current_start, current_end)
            )

            current_start = piece.start
            current_end = piece.end

        output.append(
            _Span(current_start, current_end)
        )

        return output

    def _split_span_by_token_limit(
        self,
        text: str,
        span: _Span,
        max_tokens: int,
    ) -> list[_Span]:
        """
        Exact fallback splitter.

        It uses offsets directly rather than TokenCounter.split(), because the
        latter returns strings and would force us to search reconstructed text
        back inside the original document.

        Binary search finds the longest prefix whose token count fits within the
        hard limit.
        """

        output: list[_Span] = []

        cursor = span.start

        while cursor < span.end:
            remaining = self._trim_span(
                text,
                cursor,
                span.end,
            )

            if remaining is None:
                break

            cursor = remaining.start

            if self._count_span(text, remaining) <= max_tokens:
                output.append(remaining)
                break

            low = cursor + 1
            high = span.end
            best = cursor + 1

            while low <= high:
                mid = (low + high) // 2

                candidate = _Span(cursor, mid)

                if self._count_span(text, candidate) <= max_tokens:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1

            split_end = self._prefer_natural_boundary(
                text,
                cursor,
                best,
            )

            # Safety against pathological boundary selection.
            if split_end <= cursor:
                split_end = best

            piece = self._trim_span(
                text,
                cursor,
                split_end,
            )

            if piece is None:
                # Must always make progress.
                cursor = max(split_end, cursor + 1)
                continue

            output.append(piece)
            cursor = split_end

        return output

    def _prefer_natural_boundary(
        self,
        text: str,
        start: int,
        proposed_end: int,
    ) -> int:
        """
        When possible, move a hard character split backwards to nearby
        whitespace/punctuation.

        Do not move too far backwards, otherwise we would create unnecessary
        tiny chunks.
        """

        if proposed_end <= start:
            return proposed_end

        length = proposed_end - start

        # Search only the final ~20% of the candidate span.
        search_floor = start + max(1, int(length * 0.8))

        boundary_chars = {
            "\n",
            " ",
            "\t",
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            "；",
            ";",
            "，",
            ",",
        }

        for index in range(proposed_end - 1, search_floor - 1, -1):
            if text[index] in boundary_chars:
                return index + 1

        return proposed_end

    def _count_span(
        self,
        text: str,
        span: _Span,
    ) -> int:
        return self.counter.count(
            text[span.start : span.end]
        )

    @staticmethod
    def _trim_span(
        text: str,
        start: int,
        end: int,
    ) -> _Span | None:
        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1].isspace():
            end -= 1

        if start >= end:
            return None

        return _Span(start, end)
