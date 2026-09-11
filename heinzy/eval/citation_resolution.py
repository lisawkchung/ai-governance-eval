"""
Deterministic, hierarchy-aware citation-to-retrieved-evidence resolution
(Judge v1 support).

pre:  a list of raw citation strings already extracted from a response's text
      (e.g. via heinzy.generation.grounding.extract_citations -- extraction
      only; this module never reuses that module's *matching* logic, see
      below), and the exact list of chunks retrieved FOR THAT SPECIFIC
      RESPONSE (never the full handbook/corpus).
post: one CitationResolution per raw citation, classifying it RESOLVED /
      UNRESOLVED / AMBIGUOUS against a deterministic LOGICAL SECTION INDEX
      built from that retrieved set only (see build_logical_section_index).

invariant: exact-match only, at the logical-section level. A citation
      resolves if and only if, after canonical normalization, it equals
      either (a) the full "number. title" heading of exactly one logical
      section in the index, or (b) -- only as a fallback when (a) finds
      nothing -- the bare TITLE (number stripped) of exactly one logical
      section. No substring containment, no fuzzy/semantic similarity, no
      embeddings, and no lookup against section identifiers that were not
      actually retrieved for this response. This is a deliberately
      different (stricter) contract than
      heinzy.generation.grounding.is_supported/unsupported_citations, which
      does loose bidirectional substring matching for the runtime grounding
      check -- that checker is NOT reused here as Citation Validity.

invariant: never guesses among ambiguous candidates. If two or more DISTINCT
      logical sections collide onto the same normalized key (whether via
      the full-heading pass or the title-only pass), every citation
      matching that key is AMBIGUOUS, not resolved to either candidate.
      "Distinct" is judged by the merge identity rule immediately below --
      two retrieved chunks from the SAME document and the SAME canonical
      section are the SAME logical section (merged, not ambiguous); two
      chunks that share a canonical section label but come from DIFFERENT
      documents remain genuinely distinct candidates (may still be
      AMBIGUOUS).

invariant: same-document/same-section retrieval chunks merge; nothing is
      silently dropped. Real retrieval can return more than one chunk for
      the same logical section (e.g. two adjacent passages both retrieved
      under "9. Internship Requirement" from the same document). Such
      nodes -- identified by an exact (doc_id, parsed-number-or-normalized-
      heading) match, never semantic/fuzzy similarity -- are merged into
      ONE logical section node whose own_text is the concatenation of
      EVERY contributing chunk's text, in deterministic rank order (never
      just the longer chunk kept and the rest discarded). See
      _merge_same_document_section_nodes.

invariant: synthetic retrieval-plumbing placeholders are excluded. A
      section_path that is an exact canonical match for a known synthetic
      placeholder value (currently: "unresolved") never becomes a logical
      section node or citation target -- it is retrieval-pipeline metadata
      (e.g. an unresolved front-matter/cover chunk), not a real handbook
      section. This is a narrow allowlist match, never a broad "exclude
      every unnumbered section" heuristic -- see
      _SYNTHETIC_SECTION_PATH_VALUES.

invariant: hierarchy scope is one-directional and exact. A citation to
      section N covers N and every logical section whose dotted number is a
      STRICT descendant of N (i.e. N's own tuple is a strict prefix of the
      descendant's tuple) -- and nothing else. A citation to a child never
      covers its parent or siblings; a citation to a parent covers its
      descendants regardless of which retrieved chunk they came from
      (a separately-retrieved child chunk IS in a cited parent's scope).
      This is computed purely from parsed dotted section numbers -- never
      from embeddings, semantic similarity, or fuzzy title matching.

invariant: a retrieved chunk may be a "parent" chunk whose raw text embeds
      further logical subsections (e.g. a chunk whose section_path is
      "8. Number of Units per Semester" but whose text contains "8.1 ...",
      "8.2 ...", "8.2.1. Requirements", "8.2.2. Unit Increase Timeline" as
      literal heading lines). build_logical_section_index() deterministically
      detects such embedded headings -- a leading dotted number confirmed at
      the START of a line is the only anchor, so ordinary numbered body
      prose mid-sentence is never mistaken for one -- and indexes them as
      their own logical sections, each with its own text slice, no separate
      retrieval chunk required per subsection. A heading whose title and
      first body sentence share that same physical source line with no
      blank-line separator (a real formatting artifact in this corpus) is
      still indexed: see _extract_embedded_heading_title's same-line
      run-on case, which locates the title/body boundary via an exact,
      narrow, immediate word-repetition signal -- never a general
      sentence-boundary guess and never tied to any specific section
      number.
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


# --------------------------------------------------------------------------- #
# Heading parsing: "8.2.1. Requirements" -> (number_tuple=(8,2,1),
# number_str="8.2.1", title="Requirements")
# --------------------------------------------------------------------------- #
# A leading dotted-numeric prefix (1-5 levels, matching the handbook's actual
# depth) followed by optional '.', whitespace, then a capitalized title.
_HEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+(.+)$")

# Embedded-heading DETECTION inside a chunk's raw text. A candidate heading
# must ALWAYS start at the beginning of a line (optionally preceded by a
# bullet/markdown marker) -- this anchor is what keeps ordinary numbered
# body prose ("a 3.5 CQPA", "90-700 Heinz Journal", "126 core units", a
# mid-sentence "8.1" reference, etc.) from ever being mistaken for a
# heading: none of that text starts a line with a bare dotted number.
_HEADING_LINE_START_RE = re.compile(
    r"^[ \t]*[-*#]{0,3}[ \t]*\*{0,2}(\d+(?:\.\d+){0,4})\.?[ \t]+", re.MULTILINE,
)

# Once a heading number is confirmed at a line start, its TITLE is
# extracted by trying two patterns, in order:
#
#   1. Clean case (the common one): the title simply runs to the end of the
#      physical line, bounded so a run-on line can never be swallowed whole.
#   2. Same-line run-on case: this corpus's source formatting sometimes
#      joins a subsection's heading and its own first body sentence onto
#      ONE physical line with no blank-line separator AND no punctuation
#      between them at all -- so there is no line-end and no punctuation
#      boundary to anchor on. The only structural signal actually present
#      in that situation is that the title's own final word is immediately
#      restated as the body's first word (a real, observed artifact of how
#      this corpus's heading text was assembled). This pattern is
#      deliberately narrow: it fires ONLY on an immediate, exact,
#      back-to-back word repetition right after a line-start heading
#      number -- never a general sentence-boundary or fuzzy-title guess,
#      and it is not tied to any specific section number or wording, so it
#      applies uniformly to any subsection with this same formatting
#      artifact, not just one hard-coded case.
_CLEAN_TITLE_LINE_RE = re.compile(r"([A-Z][^\n]{0,99}?)\*{0,2}[ \t]*$")
_RUNON_TITLE_RE = re.compile(r"([A-Z][^\n]{0,149}?)[ \t](\w+)[ \t]+\2\b")


def _extract_embedded_heading_title(text: str, title_start: int) -> str | None:
    """Given the text offset right after a confirmed line-start heading
    number, return the heading's title, or None if neither the clean
    end-of-line pattern nor the same-line run-on pattern matches there
    (i.e. this line-start number is not treated as a heading at all)."""
    rest = text[title_start:]
    line_end = rest.find("\n")
    line = rest if line_end == -1 else rest[:line_end]

    m_clean = _CLEAN_TITLE_LINE_RE.match(line)
    if m_clean:
        return m_clean.group(1).strip()

    m_runon = _RUNON_TITLE_RE.match(rest)
    if m_runon:
        return f"{m_runon.group(1).strip()} {m_runon.group(2)}".strip()

    return None


def _parse_number_tuple(number_str: str) -> tuple[int, ...]:
    return tuple(int(p) for p in number_str.split("."))


def _parse_heading(raw_heading: str) -> tuple[str | None, str]:
    """(number_str or None, title). title is the whole string when no
    leading dotted number is present."""
    if not raw_heading:
        return None, ""
    m = _HEADING_NUMBER_RE.match(raw_heading.strip())
    if not m:
        return None, raw_heading.strip()
    return m.group(1), m.group(2).strip()


def _is_strict_descendant(candidate: tuple[int, ...] | None, ancestor: tuple[int, ...] | None) -> bool:
    if candidate is None or ancestor is None:
        return False
    if len(candidate) <= len(ancestor):
        return False
    return candidate[: len(ancestor)] == ancestor


# --------------------------------------------------------------------------- #
# Logical section index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LogicalSection:
    """One node in the deterministic logical-section index: either a
    retrieved chunk's own top-level section, or a subsection heading
    detected embedded inside a chunk's text."""

    number_str: str | None
    number_tuple: tuple[int, ...] | None
    title: str
    raw_heading: str
    own_text: str
    doc_id: str | None
    rank: int


# Synthetic retrieval-plumbing placeholders that are NOT real logical
# sections and must never become a citation target -- narrowest possible
# rule: an exact canonical (case/whitespace/quote-normalized) match against
# this explicit allowlist of known placeholder values, never a broad
# "exclude anything unnumbered" rule. "unresolved" is the confirmed value
# observed in real pilot retrieval output (e.g. q010: a front-matter/cover
# chunk whose section structure could not be resolved by the retrieval
# pipeline, stamped section_path="unresolved"). Extend this set only when a
# new confirmed placeholder value is observed -- never widen it to a
# pattern/heuristic.
_SYNTHETIC_SECTION_PATH_VALUES = frozenset({"unresolved"})


def _is_synthetic_section_path(section_path: str | None) -> bool:
    return normalize_citation_label(section_path) in _SYNTHETIC_SECTION_PATH_VALUES


def _merge_same_document_section_nodes(nodes: list[LogicalSection]) -> list[LogicalSection]:
    """Identity rule: same doc_id + same canonical logical section (parsed
    number, or normalized raw heading when unparseable) -> ONE logical
    section node. Different doc_id + same section label -> left as
    distinct nodes (resolve_citations may then correctly surface AMBIGUOUS
    for those). No semantic/fuzzy deduplication -- identity is exact
    (doc_id, parsed-number-or-normalized-heading) equality only.

    Merging preserves ALL retrieved text from every contributing node --
    own_text is the concatenation of each contributing node's own_text, in
    deterministic order (ascending retrieval rank), never just the longer
    one. The merged node's rank is the minimum (best) rank among its
    contributors, used only for downstream ordering.
    """
    groups: dict[tuple, list[LogicalSection]] = {}
    order: list[tuple] = []
    for node in nodes:
        identity = node.number_tuple if node.number_tuple is not None else (
            "heading", normalize_citation_label(node.raw_heading),
        )
        key = (node.doc_id, identity)
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(node)

    merged: list[LogicalSection] = []
    for key in order:
        group = sorted(groups[key], key=lambda n: n.rank)
        if len(group) == 1:
            merged.append(group[0])
            continue
        primary = group[0]
        combined_text = "\n\n".join(t for t in (n.own_text.strip() for n in group) if t)
        merged.append(
            LogicalSection(
                number_str=primary.number_str,
                number_tuple=primary.number_tuple,
                title=primary.title,
                raw_heading=primary.raw_heading,
                own_text=combined_text,
                doc_id=primary.doc_id,
                rank=min(n.rank for n in group),
            )
        )
    return merged


def build_logical_section_index(
    retrieved_chunks: list[RetrievedSectionChunk],
) -> list[LogicalSection]:
    """Deterministically expand every retrieved chunk into its own logical
    section node PLUS one node per embedded subsection heading detected in
    its text (see _HEADING_LINE_START_RE / _extract_embedded_heading_title,
    which also handles a heading whose title and first body sentence share
    one physical source line -- see that function's docstring). No
    embeddings, no semantic similarity, no fuzzy title matching, no lookup
    outside `retrieved_chunks`.

    A synthetic retrieval placeholder section_path (see
    _SYNTHETIC_SECTION_PATH_VALUES, e.g. the exact value "unresolved") never
    becomes its own logical section node/citation target.

    Two nodes are collapsed into one ONLY when they share both the same
    doc_id AND the same canonical logical section identity (see
    _merge_same_document_section_nodes) -- e.g. the same real section
    retrieved as two separate chunks from the same document. Two nodes
    that share a canonical identity but come from DIFFERENT doc_ids are
    deliberately left distinct, so resolve_citations() can still surface a
    genuine cross-source collision as AMBIGUOUS rather than silently
    merging what might be two different sources. A heading detected as
    merely restating its OWN chunk's root section (identical parsed
    number, within that same chunk) is excluded before any of this --
    that is not a second candidate, just the root's own heading line.
    """
    index: list[LogicalSection] = []

    for chunk in retrieved_chunks:
        root_number_str, root_title = _parse_heading(chunk.section_path or "")
        root_number_tuple = _parse_number_tuple(root_number_str) if root_number_str else None

        text_for_scan = chunk.text or ""
        raw_matches = []
        for m in _HEADING_LINE_START_RE.finditer(text_for_scan):
            number_str = m.group(1)
            title = _extract_embedded_heading_title(text_for_scan, m.end())
            if title is None:
                continue
            number_tuple = _parse_number_tuple(number_str)
            raw_matches.append((number_tuple, number_str, title, m.start()))

        # A detected line that merely restates the chunk's own root heading
        # (same number) is not a separate child -- it only marks where the
        # root's own body text actually begins.
        child_matches = [pm for pm in raw_matches if pm[0] != root_number_tuple]
        child_matches.sort(key=lambda pm: pm[3])

        text = text_for_scan
        boundaries = [0] + [pm[3] for pm in child_matches] + [len(text)]
        root_own_text = text[boundaries[0] : boundaries[1]] if child_matches else text

        if chunk.section_path and not _is_synthetic_section_path(chunk.section_path):
            index.append(
                LogicalSection(
                    number_str=root_number_str,
                    number_tuple=root_number_tuple,
                    title=root_title,
                    raw_heading=chunk.section_path,
                    own_text=root_own_text,
                    doc_id=chunk.doc_id,
                    rank=chunk.rank,
                )
            )

        for i, (number_tuple, number_str, title, _start) in enumerate(child_matches):
            own_text = text[boundaries[i + 1] : boundaries[i + 2]]
            index.append(
                LogicalSection(
                    number_str=number_str,
                    number_tuple=number_tuple,
                    title=title,
                    raw_heading=f"{number_str}. {title}",
                    own_text=own_text,
                    doc_id=chunk.doc_id,
                    rank=chunk.rank,
                )
            )

    return _merge_same_document_section_nodes(index)


def _scope_for(node: LogicalSection, index: list[LogicalSection]) -> list[LogicalSection]:
    """Citation scope per the adopted policy: the cited node itself plus
    every STRICT descendant in the index (by dotted-number tuple prefix),
    regardless of which retrieved chunk a descendant came from. Never
    includes ancestors or siblings -- parent coverage is one-directional."""
    scope = [node]
    if node.number_tuple is not None:
        scope.extend(
            other for other in index
            if other is not node and _is_strict_descendant(other.number_tuple, node.number_tuple)
        )
    scope.sort(key=lambda n: (n.number_tuple is None, n.number_tuple or (), n.rank))
    return scope


@dataclass(frozen=True)
class CitationResolution:
    raw_citation: str
    normalized_citation: str
    resolution_status: str  # RESOLVED | UNRESOLVED | AMBIGUOUS
    resolved_section_id: str | None
    resolved_section_path: str | None
    resolved_doc_id: str | None
    # The resolved node's OWN text only (unchanged meaning from the
    # pre-hierarchy design) -- NOT the full allowed scope. Use
    # resolved_scope_retrieved_text for Citation Support evaluation.
    resolved_retrieved_text: str | None
    # NEW: the full allowed evidence scope per the hierarchy policy -- the
    # resolved node plus every retrieved descendant, regardless of which
    # chunk each came from. This is what must be checked for Citation
    # Support; a parent citation is not limited to its own chunk's text.
    resolved_scope_section_ids: tuple[str, ...] = field(default_factory=tuple)
    resolved_scope_retrieved_text: str | None = None
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
    """Resolve each raw citation against a logical-section index built ONLY
    from the chunks retrieved for this response. Never touches any
    corpus/handbook-wide section list.

    Two matching passes, in order, per citation:
      1. Full heading match: normalize(citation) == normalize("number. title")
         of exactly one logical section.
      2. Number-omitted fallback (ONLY when pass 1 finds nothing):
         normalize(citation) == normalize(title) [number stripped] of
         exactly one logical section. Still exact-equality after the same
         canonical normalization -- never fuzzy/substring/semantic.
    Either pass finding >1 distinct candidate is AMBIGUOUS, never guessed.
    """
    index = build_logical_section_index(retrieved_chunks)

    by_full_heading: dict[str, list[LogicalSection]] = {}
    by_title_only: dict[str, list[LogicalSection]] = {}
    for node in index:
        by_full_heading.setdefault(normalize_citation_label(node.raw_heading), []).append(node)
        if node.title:
            by_title_only.setdefault(normalize_citation_label(node.title), []).append(node)

    out: list[CitationResolution] = []
    for raw in raw_citations:
        norm = normalize_citation_label(raw)
        candidates = by_full_heading.get(norm, [])
        if not candidates:
            candidates = by_title_only.get(norm, [])

        if not candidates:
            out.append(
                CitationResolution(
                    raw_citation=raw, normalized_citation=norm, resolution_status=UNRESOLVED,
                    resolved_section_id=None, resolved_section_path=None, resolved_doc_id=None,
                    resolved_retrieved_text=None, resolved_scope_section_ids=(),
                    resolved_scope_retrieved_text=None, ambiguous_candidates=(),
                )
            )
            continue

        if len(candidates) > 1:
            amb = tuple(sorted({c.raw_heading for c in candidates}))
            out.append(
                CitationResolution(
                    raw_citation=raw, normalized_citation=norm, resolution_status=AMBIGUOUS,
                    resolved_section_id=None, resolved_section_path=None, resolved_doc_id=None,
                    resolved_retrieved_text=None, resolved_scope_section_ids=(),
                    resolved_scope_retrieved_text=None, ambiguous_candidates=amb,
                )
            )
            continue

        node = candidates[0]
        scope = _scope_for(node, index)
        scope_ids = tuple(n.number_str or n.raw_heading for n in scope)
        scope_text = "\n\n".join(t for t in (n.own_text.strip() for n in scope) if t)
        out.append(
            CitationResolution(
                raw_citation=raw,
                normalized_citation=norm,
                resolution_status=RESOLVED,
                resolved_section_id=node.number_str or node.raw_heading,
                resolved_section_path=node.raw_heading,
                resolved_doc_id=node.doc_id,
                resolved_retrieved_text=node.own_text,
                resolved_scope_section_ids=scope_ids,
                resolved_scope_retrieved_text=scope_text or None,
                ambiguous_candidates=(),
            )
        )
    return out
