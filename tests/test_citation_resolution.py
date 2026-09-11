"""
Tests for heinzy/eval/citation_resolution.py -- the hierarchy-aware,
deterministic citation resolver. Fully offline: no embeddings, no LLM, no
network. These are the confirmed development regression cases (A-J) from the
Judge v1 hierarchy-aware citation patch, plus supporting unit coverage for
the logical-section index builder itself.
"""
from __future__ import annotations

from heinzy.eval.citation_resolution import (
    AMBIGUOUS,
    RESOLVED,
    UNRESOLVED,
    LogicalSection,
    RetrievedSectionChunk,
    build_logical_section_index,
    normalize_citation_label,
    resolve_citations,
)

# Real chunk text shape (from the actual pilot corpus): one retrieval chunk
# whose section_path is the top-level "8." heading, but whose body embeds
# four further logical subsections (8.1, 8.2, 8.2.1, 8.2.2) as their own
# heading lines -- no separate retrieval chunk exists per subsection.
CHUNK_8_TEXT = """## **8. Number of Units per Semester**

- 8.1. Normal Courseload Expectation for MISM and MISM-BIDA Students Students typically take 54 units per semester.  Completing 54 units per semester meets the degree requirement of 162 total units for MISM and MISM-BIDA.

- 8.2. Unit Increase Policy

Students who feel prepared to take on additional academic challenges beyond the standard 54 unit course load may request a unit increase.

- 8.2.1. Requirements

Students demonstrating strong academic performance with a 3.5 CQPA or higher may request advisor approval to increase to 60 units.

- 8.2.2. Unit Increase Timeline

First-semester students may request a unit increase only if they apply for and are selected to participate in 90-700 Heinz Journal, worth 3 units.
"""

CHUNK_8 = RetrievedSectionChunk(
    section_path="8. Number of Units per Semester", doc_id="doc-x", text=CHUNK_8_TEXT, rank=1,
)


# --------------------------------------------------------------------------- #
# A: q003-style -- embedded subsection 8.2.1 resolves from within a section-8
# parent chunk, with no separate retrieval chunk for 8.2.1 itself.
# --------------------------------------------------------------------------- #
def test_case_a_embedded_subsection_8_2_1_resolves():
    res = resolve_citations(["8.2.1. Requirements"], [CHUNK_8])[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_section_path == "8.2.1. Requirements"
    assert res.resolved_section_id == "8.2.1"
    assert "3.5 CQPA" in res.resolved_retrieved_text


# --------------------------------------------------------------------------- #
# B: q022/q023-style -- same principle for 8.2.2
# --------------------------------------------------------------------------- #
def test_case_b_embedded_subsection_8_2_2_resolves():
    res = resolve_citations(["8.2.2. Unit Increase Timeline"], [CHUNK_8])[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_section_id == "8.2.2"
    assert "Heinz Journal" in res.resolved_retrieved_text


# --------------------------------------------------------------------------- #
# C: q004-style -- parent citation's allowed scope includes a SEPARATELY
# retrieved descendant chunk's evidence.
# --------------------------------------------------------------------------- #
def test_case_c_parent_citation_scope_includes_separately_retrieved_descendant():
    chunk5 = RetrievedSectionChunk(
        section_path="5. MISM Business Intelligence & Data Analytics (MISM-BIDA) Curriculum",
        doc_id="doc-x", text="Overview text for the BIDA curriculum.", rank=1,
    )
    chunk51 = RetrievedSectionChunk(
        section_path="5.1 Core and Elective Requirements",
        doc_id="doc-x", text="The curriculum requires 126 core units and 36 elective units.",
        rank=2,
    )
    res = resolve_citations(
        ["5. MISM Business Intelligence & Data Analytics (MISM-BIDA) Curriculum"],
        [chunk5, chunk51],
    )[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_scope_section_ids == ("5", "5.1")
    assert "126 core units" in res.resolved_scope_retrieved_text
    assert "36 elective units" in res.resolved_scope_retrieved_text
    # resolved_retrieved_text (the cited node's OWN text) stays narrow --
    # only resolved_scope_retrieved_text aggregates descendants.
    assert "126 core units" not in res.resolved_retrieved_text


# --------------------------------------------------------------------------- #
# D: sibling exclusion -- citation 8.2.2, evidence only in sibling 8.2.1
# --------------------------------------------------------------------------- #
def test_case_d_sibling_evidence_excluded_from_child_citation_scope():
    res = resolve_citations(["8.2.2. Unit Increase Timeline"], [CHUNK_8])[0]
    assert res.resolution_status == RESOLVED
    assert "8.2.1" not in res.resolved_scope_section_ids
    assert "3.5 CQPA" not in res.resolved_scope_retrieved_text  # 8.2.1's evidence


# --------------------------------------------------------------------------- #
# E: parent exclusion -- citation 8.2.2, evidence only in parent 8.2
# --------------------------------------------------------------------------- #
def test_case_e_parent_text_not_automatically_included_in_child_scope():
    res = resolve_citations(["8.2.2. Unit Increase Timeline"], [CHUNK_8])[0]
    assert "8.2" not in res.resolved_scope_section_ids
    assert "additional academic challenges" not in res.resolved_scope_retrieved_text  # 8.2's own text


def test_parent_child_are_distinct_nodes_not_collapsed():
    """8, 8.2, and 8.2.2 must never be treated as the same section."""
    res_parent = resolve_citations(["8. Number of Units per Semester"], [CHUNK_8])[0]
    res_child = resolve_citations(["8.2.2. Unit Increase Timeline"], [CHUNK_8])[0]
    assert res_parent.resolved_section_id != res_child.resolved_section_id
    # parent's scope DOES include the child (one-directional coverage)...
    assert "8.2.2" in res_parent.resolved_scope_section_ids
    # ...but the child's scope does NOT include the parent.
    assert "8" not in res_child.resolved_scope_section_ids


# --------------------------------------------------------------------------- #
# F: q024-style -- unique number-stripped exact-title normalization
# --------------------------------------------------------------------------- #
def test_case_f_number_omitted_title_resolves_when_unique():
    chunk9 = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="doc-x",
        text="Interns must complete a minimum of 10 weeks.", rank=1,
    )
    res = resolve_citations(["Internship Requirement"], [chunk9])[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_section_path == "9. Internship Requirement"
    assert res.resolved_section_id == "9"


# --------------------------------------------------------------------------- #
# G: ambiguous title -- two retrieved logical sections share a
# number-stripped title
# --------------------------------------------------------------------------- #
def test_case_g_ambiguous_number_stripped_title_is_ambiguous_not_guessed():
    chunk9 = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="doc-x", text="text A", rank=1,
    )
    chunk12 = RetrievedSectionChunk(
        section_path="12. Internship Requirement", doc_id="doc-x", text="text B", rank=2,
    )
    res = resolve_citations(["Internship Requirement"], [chunk9, chunk12])[0]
    assert res.resolution_status == AMBIGUOUS
    assert res.resolved_section_path is None
    assert set(res.ambiguous_candidates) == {"9. Internship Requirement", "12. Internship Requirement"}


# --------------------------------------------------------------------------- #
# H: unrelated title -> UNRESOLVED
# --------------------------------------------------------------------------- #
def test_case_h_unrelated_title_is_unresolved():
    chunk9 = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="doc-x", text="text", rank=1,
    )
    res = resolve_citations(["Totally Unrelated Topic"], [chunk9])[0]
    assert res.resolution_status == UNRESOLVED
    assert res.resolved_section_path is None
    assert res.resolved_scope_section_ids == ()


# --------------------------------------------------------------------------- #
# I: no fuzzy/semantic matching regression
# --------------------------------------------------------------------------- #
def test_case_i_similar_but_not_exact_titles_remain_unresolved():
    chunk9 = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="doc-x", text="text", rank=1,
    )
    for near_miss in (
        "Internship Requirements",  # plural
        "Internship Requirement for Students",  # extra words
        "Requirement Internship",  # reordered
        "Internships Requirement",  # near-miss stem
    ):
        res = resolve_citations([near_miss], [chunk9])[0]
        assert res.resolution_status == UNRESOLVED, f"{near_miss!r} must not fuzzy-resolve"


def test_numeric_prefix_is_never_treated_as_fuzzy_containment():
    """8.1 must not resolve merely because a retrieved section 8.1.2 exists
    (a real prefix relationship, but not what was cited)."""
    chunk = RetrievedSectionChunk(section_path="8.1.2. Sub-detail", doc_id="d1", text="t", rank=1)
    res = resolve_citations(["8.1"], [chunk])[0]
    assert res.resolution_status == UNRESOLVED


# --------------------------------------------------------------------------- #
# J: existing exact citation behavior still passes (no hierarchy involved)
# --------------------------------------------------------------------------- #
def test_case_j_existing_exact_top_level_citation_still_resolves():
    chunk = RetrievedSectionChunk(section_path="4.1. Core Courses", doc_id="d1", text="t", rank=1)
    res = resolve_citations(["4.1. Core Courses"], [chunk])[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_section_path == "4.1. Core Courses"


def test_case_j_case_insensitive_and_whitespace_normalization_still_works():
    chunk = RetrievedSectionChunk(section_path="4.1. Core Courses", doc_id="d1", text="t", rank=1)
    res = resolve_citations(["  4.1.   core   courses  "], [chunk])[0]
    assert res.resolution_status == RESOLVED


def test_case_j_distinct_sources_with_same_normalized_heading_stay_ambiguous():
    """The pre-hierarchy 'never guess' invariant for genuinely distinct
    retrieved sources that happen to normalize identically (e.g. differing
    only by case) must still hold after the hierarchy-aware rewrite."""
    chunk_a = RetrievedSectionChunk(section_path="8. Units", doc_id="d1", text="t1", rank=1)
    chunk_b = RetrievedSectionChunk(section_path="8. units", doc_id="d2", text="t2", rank=2)
    res = resolve_citations(["8. Units"], [chunk_a, chunk_b])[0]
    assert res.resolution_status == AMBIGUOUS


# --------------------------------------------------------------------------- #
# Supporting coverage: the logical-section index builder itself
# --------------------------------------------------------------------------- #
def test_index_builds_root_plus_all_four_embedded_children_for_chunk_8():
    """8.2, 8.2.1, and 8.2.2 each sit on their own clean heading line. 8.1's
    heading and its own first body sentence run together on one physical
    source line with no blank-line separator -- it is still indexed as its
    own distinct node via the same-line run-on title-boundary rule (an
    immediate word-repetition marker), not silently absorbed into root. See
    test_same_line_runon_heading_is_indexed_via_word_repetition_boundary."""
    index = build_logical_section_index([CHUNK_8])
    numbers = sorted(n.number_str for n in index)
    assert numbers == ["8", "8.1", "8.2", "8.2.1", "8.2.2"]


def test_root_heading_restatement_inside_text_is_not_a_duplicate_child():
    """The literal '## **8. Number of Units per Semester**' line inside the
    chunk's own text must not become a second, redundant '8' node."""
    index = build_logical_section_index([CHUNK_8])
    eights = [n for n in index if n.number_str == "8"]
    assert len(eights) == 1


def test_embedded_heading_never_detected_across_chunk_boundaries():
    """A heading embedded in one chunk's text must never be attributed as a
    child of an unrelated, separately retrieved chunk."""
    unrelated = RetrievedSectionChunk(
        section_path="99. Unrelated Section", doc_id="doc-y", text="Nothing about units here.",
        rank=5,
    )
    index = build_logical_section_index([unrelated, CHUNK_8])
    for node in index:
        if node.number_str and node.number_str.startswith("8"):
            assert node.doc_id == "doc-x"


def test_same_line_runon_heading_is_indexed_via_word_repetition_boundary():
    """Regression case 1: a subsection heading whose title and first body
    sentence share one physical source line (real corpus artifact: the
    title's own final word is immediately restated as the body's first
    word, with no punctuation between them) is still indexed as its own
    distinct, directly resolvable logical section -- not silently absorbed
    into its parent's own_text."""
    index = build_logical_section_index([CHUNK_8])
    node_81 = next((n for n in index if n.number_str == "8.1"), None)
    assert node_81 is not None
    assert node_81.title == "Normal Courseload Expectation for MISM and MISM-BIDA Students"
    # the extracted title does not swallow the following body sentence
    assert "typically take 54 units" not in node_81.title


def test_regression_case_2_section_8_scope_includes_all_four_children():
    res = resolve_citations(["8. Number of Units per Semester"], [CHUNK_8])[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_scope_section_ids == ("8", "8.1", "8.2", "8.2.1", "8.2.2")


def test_regression_case_3_direct_citation_to_8_1_resolves():
    res = resolve_citations(
        ["8.1. Normal Courseload Expectation for MISM and MISM-BIDA Students"], [CHUNK_8]
    )[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_section_id == "8.1"
    assert "54 units per semester" in res.resolved_retrieved_text


def test_regression_case_4_ordinary_numbered_body_prose_not_indexed_as_heading():
    """Numbers embedded mid-sentence in body prose (a CQPA figure, a course
    worth N units, a unit count) must never be mistaken for a heading start
    -- only a dotted number at the very START of a line is ever a
    candidate, so none of this body-prose content produces a spurious
    logical-section node."""
    index = build_logical_section_index([CHUNK_8])
    numbers = {n.number_str for n in index}
    # None of these numeric mentions from the body text are indexed:
    # "3.5 Cumulative Quality Point Average (CQPA)", "60 units", "90-700
    # Heinz Journal, which is worth 3 units", "162 total units", "54 units".
    for spurious in ("3.5", "60", "90", "700", "162", "54", "3"):
        assert spurious not in numbers
    # exactly the five real logical sections, nothing else
    assert numbers == {"8", "8.1", "8.2", "8.2.1", "8.2.2"}


def test_regression_case_4_mid_sentence_number_not_indexed_even_at_high_precision():
    """A number that merely appears mid-sentence, even one that could parse
    as a plausible dotted section number, must not be indexed unless it
    actually starts a line."""
    chunk = RetrievedSectionChunk(
        section_path="1. Introduction", doc_id="d1",
        text="## **1. Introduction**\n\nSee also 1.2 and 1.3 for related policies in this same paragraph.\n",
        rank=1,
    )
    index = build_logical_section_index([chunk])
    numbers = {n.number_str for n in index}
    assert numbers == {"1"}


def test_scope_never_includes_the_cited_node_twice():
    index = build_logical_section_index([CHUNK_8])
    from heinzy.eval.citation_resolution import _scope_for

    node_8 = next(n for n in index if n.number_str == "8")
    scope = _scope_for(node_8, index)
    assert len(scope) == len(set(id(n) for n in scope))


def test_node_with_unparseable_number_has_empty_hierarchy_scope():
    """A retrieved section whose heading carries no leading dotted number
    gets a degenerate scope of just itself -- never guesses a hierarchy
    relationship it cannot parse."""
    chunk = RetrievedSectionChunk(
        section_path="Appendix: Miscellaneous Notes", doc_id="d1", text="Some notes.", rank=1,
    )
    res = resolve_citations(["Appendix: Miscellaneous Notes"], [chunk])[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_scope_section_ids == ("Appendix: Miscellaneous Notes",)


def test_normalize_citation_label_unchanged_contract():
    """Regression guard: normalization itself (unrelated to hierarchy) is
    unchanged -- case-fold, whitespace, edge punctuation only."""
    assert normalize_citation_label('  "8.1.  Foo   Bar"  ') == "8.1. foo bar"
    assert normalize_citation_label("8.1") != normalize_citation_label("8.1.2")


# =========================================================================== #
# Freeze cleanup fix 1: exclude synthetic "unresolved" retrieval placeholders
# (observed real pilot case q010).
# =========================================================================== #
def test_synthetic_unresolved_section_path_creates_no_logical_section():
    chunk = RetrievedSectionChunk(
        section_path="unresolved", doc_id="doc-x",
        text="MISM Student Handbook -- cover / front matter.", rank=1,
    )
    index = build_logical_section_index([chunk])
    assert index == []


def test_citation_unresolved_stays_unresolved():
    chunk = RetrievedSectionChunk(
        section_path="unresolved", doc_id="doc-x",
        text="MISM Student Handbook -- cover / front matter.", rank=1,
    )
    res = resolve_citations(["unresolved"], [chunk])[0]
    assert res.resolution_status == UNRESOLVED
    assert res.resolved_section_path is None


def test_unresolved_placeholder_does_not_block_other_retrieved_sections():
    """A synthetic placeholder chunk retrieved alongside real sections must
    not interfere with resolving those real sections."""
    placeholder = RetrievedSectionChunk(
        section_path="unresolved", doc_id="doc-x", text="cover page", rank=1,
    )
    real = RetrievedSectionChunk(
        section_path="4.1. Core Courses", doc_id="doc-x", text="Seven core courses.", rank=2,
    )
    res = resolve_citations(["4.1. Core Courses"], [placeholder, real])[0]
    assert res.resolution_status == RESOLVED


def test_placeholder_exclusion_is_narrow_not_a_broad_unnumbered_exclusion():
    """A genuinely real, unnumbered section title (e.g. an appendix) must
    still resolve -- only the exact canonical placeholder value is
    excluded, never every unnumbered section_path."""
    chunk = RetrievedSectionChunk(
        section_path="Appendix: Miscellaneous Notes", doc_id="doc-x", text="Some notes.", rank=1,
    )
    res = resolve_citations(["Appendix: Miscellaneous Notes"], [chunk])[0]
    assert res.resolution_status == RESOLVED


# =========================================================================== #
# Freeze cleanup fix 2: merge multiple retrieved chunks from the SAME real
# logical section (observed real pilot case q026: two chunks, same doc_id,
# both section_path="9. Internship Requirement").
# =========================================================================== #
def _q026_chunks():
    chunk_a = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="c5bd29780f864e866f552de33beb1758",
        text="Interns must complete a minimum of 10 weeks of full-time work.", rank=1,
    )
    chunk_b = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="c5bd29780f864e866f552de33beb1758",
        text="Internship credit requires prior advisor approval via petition.", rank=2,
    )
    return chunk_a, chunk_b


def test_case_a_same_doc_same_section_two_chunks_resolves_not_ambiguous():
    chunk_a, chunk_b = _q026_chunks()
    res = resolve_citations(["9. Internship Requirement"], [chunk_a, chunk_b])[0]
    assert res.resolution_status == RESOLVED
    assert res.resolved_section_id == "9"


def test_case_a_merged_evidence_contains_text_from_both_chunks():
    chunk_a, chunk_b = _q026_chunks()
    res = resolve_citations(["9. Internship Requirement"], [chunk_a, chunk_b])[0]
    assert "10 weeks" in res.resolved_retrieved_text
    assert "advisor approval" in res.resolved_retrieved_text
    # scope text (self + descendants -- no descendants here) must also
    # carry both, never just the longer chunk's text.
    assert "10 weeks" in res.resolved_scope_retrieved_text
    assert "advisor approval" in res.resolved_scope_retrieved_text


def test_case_b_different_doc_ids_same_heading_still_ambiguous():
    chunk_a, _ = _q026_chunks()
    chunk_other_doc = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="DIFFERENT-DOC-ID",
        text="A different source's internship text.", rank=2,
    )
    res = resolve_citations(["9. Internship Requirement"], [chunk_a, chunk_other_doc])[0]
    assert res.resolution_status == AMBIGUOUS
    assert set(res.ambiguous_candidates) == {"9. Internship Requirement"}
    # both distinct candidates counted even though their normalized
    # heading text is identical -- doc_id is what keeps them apart.


def test_case_c_q026_section_9_appears_only_once_in_index():
    chunk_a, chunk_b = _q026_chunks()
    index = build_logical_section_index([chunk_a, chunk_b])
    nines = [n for n in index if n.number_str == "9"]
    assert len(nines) == 1


def test_merge_does_not_drop_evidence_even_with_three_chunks():
    chunk_a, chunk_b = _q026_chunks()
    chunk_c = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="c5bd29780f864e866f552de33beb1758",
        text="A third retrieved passage from the same section.", rank=3,
    )
    res = resolve_citations(["9. Internship Requirement"], [chunk_a, chunk_b, chunk_c])[0]
    assert res.resolution_status == RESOLVED
    for expected in ("10 weeks", "advisor approval", "third retrieved passage"):
        assert expected in res.resolved_retrieved_text


def test_merge_rank_is_the_minimum_contributing_rank():
    chunk_a, chunk_b = _q026_chunks()  # ranks 1 and 2
    index = build_logical_section_index([chunk_b, chunk_a])  # built out of rank order
    node9 = next(n for n in index if n.number_str == "9")
    assert node9.rank == 1


def test_merge_identity_for_numbered_sections_is_the_parsed_number():
    """For a numbered section, canonical identity is the parsed dotted
    number itself (the same identity concept _scope_for/hierarchy already
    use throughout this module) -- not exact title-string equality. Two
    chunks from the same doc_id under the same number "9" merge into one
    node even if their extracted title text has a minor variance, since
    they are the same real numbered section. This is NUMBER equality, not
    title similarity -- see the next test for the still-exact (never
    fuzzy) fallback used when a section has no parseable number at all."""
    chunk_a, _ = _q026_chunks()
    title_variant = RetrievedSectionChunk(
        section_path="9. Internship Requirements",  # trailing 's' -- minor title variance
        doc_id="c5bd29780f864e866f552de33beb1758", text="a third passage", rank=2,
    )
    index = build_logical_section_index([chunk_a, title_variant])
    nines = [n for n in index if n.number_str == "9"]
    assert len(nines) == 1
    assert "a third passage" in nines[0].own_text  # merged evidence, not dropped


def test_merge_never_uses_fuzzy_matching_for_unnumbered_sections():
    """When a section has NO parseable number, merge identity falls back to
    the normalized raw heading -- still EXACT equality, never fuzzy/
    similarity-based. A near-identical but not-identical unnumbered heading
    from the same doc_id must NOT merge."""
    chunk_a = RetrievedSectionChunk(
        section_path="Appendix: Miscellaneous Notes", doc_id="doc-x", text="text A", rank=1,
    )
    near_miss = RetrievedSectionChunk(
        section_path="Appendix: Miscellaneous Note",  # singular -- not identical
        doc_id="doc-x", text="text B", rank=2,
    )
    index = build_logical_section_index([chunk_a, near_miss])
    assert len(index) == 2  # NOT merged -- normalized headings differ exactly


# --------------------------------------------------------------------------- #
# D & E: existing hierarchy / number-omitted-title / sibling-parent exclusion
# behavior is unaffected by the two freeze-cleanup fixes above.
# --------------------------------------------------------------------------- #
def test_case_d_hierarchy_scope_still_correct_after_freeze_fixes():
    res = resolve_citations(["8. Number of Units per Semester"], [CHUNK_8])[0]
    assert res.resolved_scope_section_ids == ("8", "8.1", "8.2", "8.2.1", "8.2.2")


def test_case_e_number_omitted_and_sibling_parent_exclusion_still_correct():
    chunk9 = RetrievedSectionChunk(
        section_path="9. Internship Requirement", doc_id="doc-x",
        text="Interns must complete 10 weeks.", rank=1,
    )
    res_title_only = resolve_citations(["Internship Requirement"], [chunk9])[0]
    assert res_title_only.resolution_status == RESOLVED

    res_822 = resolve_citations(["8.2.2. Unit Increase Timeline"], [CHUNK_8])[0]
    assert "8.2.1" not in res_822.resolved_scope_section_ids  # sibling excluded
    assert "8.2" not in res_822.resolved_scope_section_ids  # parent excluded
