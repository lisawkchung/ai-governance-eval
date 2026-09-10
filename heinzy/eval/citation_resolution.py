"""
Deterministic citation-to-retrieved-evidence resolution (Judge v1 support).

pre:  a list of raw citation strings already extracted from a response's text
      (e.g. via heinzy.generation.grounding.extract_citations -- extraction
      only; this module never reuses that module's *matching* logic, see
      below), and the exact list of chunks retrieved FOR THAT SPECIFIC
      RESPONSE (never the full handbook/corpus).
post: one CitationResolution per raw citation, classifying it RESOLVED /
      UNRESOLVED / AMBIGUOUS against that retrieved set only.
invariant: exact-match only. A citation resolves if and only if, after the
      canonical normalization below, it equals the normalized form of some
      retrieved chunk's section_path exactly. No substring containment, no
      fuzzy/semantic similarity, no embeddings, and no lookup against
      section identifiers that were not actually retrieved for this
      response. This is a deliberately different (stricter) contract than
      heinzy.generation.grounding.is_supported/unsupported_citations, which
      does loose bidirectional substring matching for the runtime grounding
      check -- that checker is NOT reused here as Citation Validity (see
      judge_rubric_v1.md section 7's documented failure risks: fuzzy
      substring matching, hierarchical section ambiguity, numeric sibling
      collisions, numeric/page-like parsing ambiguity, body-text-as-citation
      -- an exact-match-only design is immune to all of these by
      construction, since it never does prefix/substring/numeric-prefix
      comparison of any kind).
invariant: never guesses among ambiguous candidates. If canonical
      normalization causes two or more DISTINCT retrieved section_path
      strings to collide onto the same normalized key, every citation
      matching that key is AMBIGUOUS, not resolved to either candidate.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

RESOLVED = "RESOLVED"
UNRESOLVED = "UNRESOLVED"
AMBIGUOUS = "AMBIGUOUS"
_VALID_STATUSES = (RESOLVED, UNRESOLVED, AMBIGUOUS)

_WS_RE = re.compile(r"\s+")
_QUOTE_TRANSLATION = str.maketrans(
    {"“": '"', "”": '"', "‘": "'", "’": "'"}
)
_EDGE_PUNCT = " \t\n\r.,;:-\"'"


def normalize_citation_label(label: str | None) -> str:
    """Canonical, auditable normalization: unicode-fold, straighten quotes,
    strip surrounding quotes/punctuation, collapse whitespace, lowercase.

    Deliberately minimal and exact -- no stemming, no punctuation removal
    from the *interior* of the label (so "8.1" and "8.1.2" never collide),
    no fuzzy/semantic step of any kind.
    """
    if not label:
        return ""
    s = unicodedata.normalize("NFKC", label)
    s = s.translate(_QUOTE_TRANSLATION)
    s = s.strip(_EDGE_PUNCT)
    s = _WS_RE.sub(" ", s)
    return s.strip().lower()


@dataclass(frozen=True)
class RetrievedSectionChunk:
    """Minimal shape this module needs from a retrieval chunk snapshot."""

    section_path: str | None
    doc_id: str | None
    text: str
    rank: int


@dataclass(frozen=True)
class CitationResolution:
    raw_citation: str
    normalized_citation: str
    resolution_status: str  # RESOLVED | UNRESOLVED | AMBIGUOUS
    resolved_section_id: str | None
    resolved_section_path: str | None
    resolved_doc_id: str | None
    resolved_retrieved_text: str | None
    ambiguous_candidates: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.resolution_status not in _VALID_STATUSES:
            raise ValueError(
                f"resolution_status must be one of {_VALID_STATUSES}, "
                f"got {self.resolution_status!r}"
            )


def resolve_citations(
    raw_citations: list[str],
    retrieved_chunks: list[RetrievedSectionChunk],
) -> list[CitationResolution]:
    """Resolve each raw citation against ONLY the chunks retrieved for this
    response. Never touches any corpus/handbook-wide section list.
    """
    # normalized section_path -> distinct raw section_path strings that
    # normalize to it, in first-seen order. More than one distinct raw
    # value under the same normalized key is exactly the ambiguity case.
    by_normalized: dict[str, list[str]] = {}
    for chunk in retrieved_chunks:
        if not chunk.section_path:
            continue
        norm = normalize_citation_label(chunk.section_path)
        if not norm:
            continue
        bucket = by_normalized.setdefault(norm, [])
        if chunk.section_path not in bucket:
            bucket.append(chunk.section_path)

    out: list[CitationResolution] = []
    for raw in raw_citations:
        norm = normalize_citation_label(raw)
        candidates = by_normalized.get(norm, [])

        if not candidates:
            out.append(
                CitationResolution(
                    raw_citation=raw,
                    normalized_citation=norm,
                    resolution_status=UNRESOLVED,
                    resolved_section_id=None,
                    resolved_section_path=None,
                    resolved_doc_id=None,
                    resolved_retrieved_text=None,
                    ambiguous_candidates=(),
                )
            )
            continue

        if len(candidates) > 1:
            out.append(
                CitationResolution(
                    raw_citation=raw,
                    normalized_citation=norm,
                    resolution_status=AMBIGUOUS,
                    resolved_section_id=None,
                    resolved_section_path=None,
                    resolved_doc_id=None,
                    resolved_retrieved_text=None,
                    ambiguous_candidates=tuple(candidates),
                )
            )
            continue

        section_path = candidates[0]
        matching = sorted(
            (c for c in retrieved_chunks if c.section_path == section_path),
            key=lambda c: c.rank,
        )
        resolved_text = "\n\n".join(c.text for c in matching) if matching else None
        resolved_doc_id = matching[0].doc_id if matching else None
        out.append(
            CitationResolution(
                raw_citation=raw,
                normalized_citation=norm,
                resolution_status=RESOLVED,
                # This corpus exposes a single heading-style identifier per
                # chunk (section_path) rather than a separate short numeric
                # section_id distinct from the heading -- both fields are
                # therefore the same string here; kept as two fields to
                # match the documented contract for a corpus that does
                # distinguish them.
                resolved_section_id=section_path,
                resolved_section_path=section_path,
                resolved_doc_id=resolved_doc_id,
                resolved_retrieved_text=resolved_text,
                ambiguous_candidates=(),
            )
        )
    return out
