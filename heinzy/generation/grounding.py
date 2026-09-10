"""
Grounding checks. Did the answer stay inside the retrieved chunks?

The generation prompt asks the model to cite the sections it used, in the form
`(see "4.1. Required Courses")`. This module turns that convention into a
machine-checkable signal: every section the answer claims to have used must
correspond to a section that retrieval actually returned.

Scope, stated honestly: a clean check here is a NECESSARY condition for a
grounded answer, not a sufficient one. It proves the answer did not cite
anything outside the retrieved set; it cannot prove every sentence is supported
by that set. Faithfulness scoring of claim-vs-chunk belongs to the eval harness
(A6), not here.

pre:  hits are the exact ScoredChunks the answer was generated from
post: extract_citations returns section labels in the order the model cited
      them; unsupported_citations returns the subset matching no retrieved hit
invariant: pure functions, no I/O and no config, so the eval harness and the
           unit tests can call them without a model, a store, or a network.
"""
from __future__ import annotations

import re

# The model is told to cite as: (see "4.1. Required Courses"). Anchor on that
# construct rather than on any quoted string. Models also quote handbook prose
# verbatim, and treating those quotes as citations produced false "unsupported"
# flags.
#
# Used only by substantive_word_count() below, which just needs to strip
# roughly-citation-shaped parentheticals before counting words -- it does not
# need exact block-boundary correctness. extract_citations() below does NOT
# use this regex, because [^)] stops at the FIRST ')' in the block, which
# incorrectly truncates a quoted label that itself contains a parenthetical,
# e.g. (see "Students may transfer to the MISM (non-BIDA) program") was cut
# to "Students may transfer to the MISM (non-BIDA" -- see _see_block_end().
_SEE_BLOCK = re.compile(r"\(\s*see[:\s]\s*([^)]{3,300})\)", re.IGNORECASE)

# extract_citations() uses this anchor plus _see_block_end() (a small
# deterministic quote-tracking scan, not a regex) to find the TRUE end of a
# (see ...) block: a ')' encountered while inside a quoted label is part of
# the label, not the block terminator.
_SEE_START = re.compile(r"\(\s*see[:\s]\s*", re.IGNORECASE)
_QUOTE_TOGGLE_CHARS = ('"', "“", "”")  # straight and curly double quotes

# Inside a (see ...) block, split multiple sections: quoted runs first, and if
# the model dropped the quotes, fall back to comma / "and" separation.
_QUOTED = re.compile(r'["“]([^"”]{2,150})["”]')
_SPLIT_UNQUOTED = re.compile(r",|\band\b|;", re.IGNORECASE)


def _see_block_end(text: str, start: int, *, window: int = 400) -> int | None:
    """Return the index of the ')' that closes a (see ...) block starting at
    `start` (the position right after '(see...'), or None if not found
    within `window` characters.

    Deterministic quote-boundary tracking, not fuzzy/semantic matching: any
    of '"'/'“'/'”' toggles whether we are "inside a quoted label";
    a ')' encountered while inside a quote is part of the label and does NOT
    terminate the block, so a quoted citation label may itself contain
    parentheses. A ')' encountered outside any quote terminates the block,
    exactly as the original [^)] boundary did for the common unquoted case.
    """
    in_quote = False
    limit = min(len(text), start + window)
    i = start
    while i < limit:
        ch = text[i]
        if ch in _QUOTE_TOGGLE_CHARS:
            in_quote = not in_quote
        elif ch == ")" and not in_quote:
            return i
        i += 1
    return None

# Models cite in prose at least as often as they follow the parenthetical
# format, as in "According to section 7.2, students must...". An extractor
# blind to those reports the answer as citing nothing, so the grounding check
# passes on an empty set instead of verifying anything.
#
# Captures the section number alone: the words after it are part of the
# sentence, not the label, and a citation matches on the number anyway. Note
# \b cannot lead the alternation, because § is not a word character and so
# there is no boundary between a preceding space and it.
_PROSE_SECTION = re.compile(r"(?:\bsections?\b|§)\s*[\"“]?(\d+(?:\.\d+)*)", re.IGNORECASE)

# gemma cites as a bare parenthetical such as "(9. Internship Requirement)",
# with no "see" keyword, so every one of its citations went unextracted and its
# clean grounding score was vacuous. Require a leading section number followed
# by a capitalised word, which admits "(8.1. Normal Courseload Expectation)"
# while rejecting course codes like "(94-879, 6 units)" and asides like
# "(described above)".
_BARE_SECTION_PAREN = re.compile(r"\(\s*(\d+(?:\.\d+)*\.?\s+[A-Z][^)]{2,120})\)")

# Models echo the context format `[section] (pN)` back into their citations, so
# a (see ...) block often carries a page marker alongside the section. A page
# number is not a section name and matches nothing, so it must not be treated
# as an uncited source.
_PAGE_LIKE = re.compile(r"^(p|pp|pg|page|pages)?[\s.:]*[\[\(]?[\d,\s\-]+[\]\)]?$", re.IGNORECASE)

# Section paths carry structure ("Handbook > 4. Curriculum > 4.1. Required
# Courses") that a citation usually only names the tail of, so matching is done
# on normalized text with punctuation stripped.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, for loose matching."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def extract_citations(text: str) -> list[str]:
    """Pull the section labels an answer claims to have used.

    Returns them de-duplicated, in first-mentioned order.
    """
    found: list[str] = []
    text = text or ""
    for start_match in _SEE_START.finditer(text):
        block_start = start_match.end()
        block_end = _see_block_end(text, block_start)
        if block_end is None:
            continue
        block = text[block_start:block_end]
        quoted = _QUOTED.findall(block)
        parts = quoted if quoted else _SPLIT_UNQUOTED.split(block)
        for part in parts:
            label = part.strip().strip('"“”.,;: ')
            # "section", "sections", "the" alone carry no information.
            if len(normalize(label)) < 3:
                continue
            if _PAGE_LIKE.match(label):
                continue
            if label not in found:
                found.append(label)

    for match in _BARE_SECTION_PAREN.findall(text):
        label = match.strip().strip('"“”.,;: ')
        if len(normalize(label)) < 3 or _PAGE_LIKE.match(label):
            continue
        if label not in found:
            found.append(label)

    for match in _PROSE_SECTION.findall(text):
        label = match.strip().strip('"“”.,;: ')
        if len(normalize(label)) < 1 or _PAGE_LIKE.match(label):
            continue
        # Skip anything already covered by a parenthetical citation, so one
        # section mentioned both ways isn't counted twice.
        if any(normalize(label) in normalize(f) for f in found):
            continue
        if label not in found:
            found.append(label)
    return found


def substantive_word_count(text: str) -> int:
    """Words in an answer that aren't part of a citation construct.

    An answer of just `(see "7. Concentrations")` points at a section without
    saying anything. It is not a refusal either, so the pass/fail check counted
    it as a successful answer. Stripping the citations and counting what is left
    separates a real short answer like "54 units per semester." from a pointer
    with no content.
    """
    stripped = _SEE_BLOCK.sub(" ", text or "")
    return len([w for w in normalize(stripped).split() if w])


def is_supported(
    citation: str,
    section_paths: list[str | None],
    chunk_texts: list[str] | None = None,
) -> bool:
    """True when a cited label is drawn from the retrieved set.

    Two ways to qualify:

    1. It matches a retrieved section path, in either direction. The model may
       cite the leaf ("4.1. Core Courses") of a longer stored path, or restate
       the whole path.
    2. The label appears verbatim in retrieved chunk *text*. Section paths are
       only as good as the structure pass: real subsection headings that the
       structure builder didn't promote ("8.1. Normal Courseload Expectation")
       live in the chunk body, and so do page footers the extractor swept up.
       A model citing those is still quoting the retrieved context, which is
       what "grounded" means here. The distinction worth catching is an
       *invented* reference, and matching on the full normalized phrase stays
       strict enough for that.
    """
    cite_n = normalize(citation)
    if not cite_n:
        return False
    for text in chunk_texts or []:
        if f" {cite_n} " in f" {normalize(text)} ":
            return True
    for path in section_paths:
        if not path:
            continue
        path_n = normalize(path)
        if not path_n:
            continue
        # Space-padded so a numeric citation matches whole tokens only:
        # unpadded, "7.2" ("7 2") is a substring of a "7.21" path ("7 21").
        if f" {cite_n} " in f" {path_n} " or f" {path_n} " in f" {cite_n} ":
            return True
    return False


def unsupported_citations(
    citations: list[str],
    section_paths: list[str | None],
    chunk_texts: list[str] | None = None,
) -> list[str]:
    """The cited labels that match nothing retrieval returned."""
    return [c for c in citations if not is_supported(c, section_paths, chunk_texts)]
