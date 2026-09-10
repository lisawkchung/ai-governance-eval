"""
Grounded-answering contract tests.

These lock the behaviour that matters: the system refuses rather than invents,
and it never cites a section retrieval didn't return. They run fully offline,
with no Ollama, no Chroma and no model download, by stubbing the HTTP call. A
contract test that needs a GPU box reachable is a contract test nobody runs.

The end-to-end proof against a real model is scripts/eval_abstention.py; this
file is what keeps someone from quietly breaking the contract in between.
"""
from __future__ import annotations

import pytest

from heinzy.common.config import load_config
from heinzy.generation import generator as generator_module
from heinzy.generation.generator import (
    REASON_INSUFFICIENT,
    REASON_NO_CONTEXT,
    Generator,
)
from heinzy.generation.grounding import extract_citations, unsupported_citations
from heinzy.retrieval.store import ScoredChunk

HITS = [
    ScoredChunk(
        chunk_id="c1",
        text="Students must complete seven core courses totalling 144 units.",
        score=0.81,
        doc_id="doc-x",
        section_path="Handbook > 4. Curriculum > 4.1. Core Courses",
        source_pages=[11],
    ),
    ScoredChunk(
        chunk_id="c2",
        text="Students may take up to four electives.",
        score=0.74,
        doc_id="doc-x",
        section_path="Handbook > 4. Curriculum > 4.2. Electives",
        source_pages=[13],
    ),
]


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"message": {"content": self._content}}


@pytest.fixture
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the Ollama call. Returns a dict recording whether it was hit."""
    calls: dict = {"count": 0, "payload": None, "auth": None}

    def _install(content: str):
        def fake_post(url, json=None, timeout=None, auth=None, headers=None, **kwargs):
            calls["count"] += 1
            calls["payload"] = json
            calls["auth"] = auth
            return _FakeResponse(content)

        monkeypatch.setattr(generator_module.requests, "post", fake_post)
        return calls

    return _install


def test_no_hits_refuses_without_calling_the_model(cfg, stub_model):
    """The whole point of layer 1: no context -> no model call, no chance to invent."""
    calls = stub_model("144 units are required.")
    answer = Generator(cfg).generate("what is the tuition?", [])

    assert answer.refused
    assert answer.refusal_reason == REASON_NO_CONTEXT
    assert calls["count"] == 0, "model was called with no context to ground in"


def test_sentinel_response_becomes_a_refusal(cfg, stub_model):
    stub_model("INSUFFICIENT_CONTEXT")
    answer = Generator(cfg).generate("who is the university president?", HITS)

    assert answer.refused
    assert answer.refusal_reason == REASON_INSUFFICIENT
    assert answer.raw_text == "INSUFFICIENT_CONTEXT"


def test_sentinel_detected_even_when_model_adds_prose(cfg, stub_model):
    """Told to return the sentinel alone, models still wrap it in a sentence."""
    stub_model("I'm sorry, but INSUFFICIENT_CONTEXT to answer that question.")
    assert Generator(cfg).generate("what is the average salary?", HITS).refused


def test_refusal_text_is_deterministic_not_model_prose(cfg, stub_model):
    """Downstream branches on `refused`; the text must not vary per model mood."""
    stub_model("INSUFFICIENT_CONTEXT")
    gen = Generator(cfg)
    answer = gen.generate("anything unanswerable", HITS)

    assert answer.text == gen.refusal_text
    assert "INSUFFICIENT_CONTEXT" not in answer.text, "sentinel leaked to the user"


def test_grounded_answer_is_not_refused(cfg, stub_model):
    stub_model('Students complete seven core courses (see "4.1. Core Courses").')
    answer = Generator(cfg).generate("what are the core requirements?", HITS)

    assert not answer.refused
    assert answer.is_grounded
    assert answer.cited_sections == ["4.1. Core Courses"]
    assert answer.unsupported_citations == []


def test_citation_outside_retrieved_set_is_flagged(cfg, stub_model):
    """A cited section nobody retrieved is the signature of an invented source."""
    stub_model('Tuition is $50,000 (see "12.3. Tuition and Fees").')
    answer = Generator(cfg).generate("what is the tuition?", HITS)

    assert answer.unsupported_citations == ["12.3. Tuition and Fees"]
    assert not answer.is_grounded


def test_sentinel_and_min_hits_come_from_config(cfg, stub_model):
    """No abstention constants in source (S5)."""
    calls = stub_model("some answer")
    cfg.generation.abstain.sentinel = "NOPE_CANT_ANSWER"
    cfg.generation.abstain.min_hits = 3
    gen = Generator(cfg)

    assert gen.sentinel == "NOPE_CANT_ANSWER"
    assert "NOPE_CANT_ANSWER" in gen.system_prompt
    # 2 hits < min_hits of 3 -> layer 1 fires
    assert gen.generate("q", HITS).refused
    assert calls["count"] == 0


def test_prompt_carries_only_retrieved_text(cfg, stub_model):
    """'Constructed only from retrieved chunks' starts with what we send."""
    calls = stub_model("ok")
    Generator(cfg).generate("core requirements?", HITS)

    prompt = calls["payload"]["messages"][1]["content"]
    for hit in HITS:
        assert hit.text in prompt
    assert "ONLY" in calls["payload"]["messages"][0]["content"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ('See the handbook (see "4.1. Core Courses").', ["4.1. Core Courses"]),
        ('Two sources (see "4.1. Core Courses" and "4.2. Electives").',
         ["4.1. Core Courses", "4.2. Electives"]),
        ("No citation here at all.", []),
        # Quoted handbook prose is not a citation. Flagging it produced false
        # "unsupported" hits before the extractor was anchored on (see ...).
        ('The handbook says "students must complete seven core courses".', []),
    ],
)
def test_citation_extraction(text, expected):
    assert extract_citations(text) == expected


# --------------------------------------------------------------------------- #
# Regression: a quoted citation label containing its own parentheses must not
# be truncated at the inner ')' (discovered via q013 Judge-input inspection:
# (see "Students may transfer to the MISM (non-BIDA) program") was cut to
# "Students may transfer to the MISM (non-BIDA"). Fix is deterministic
# quote-boundary tracking (_see_block_end), never fuzzy/semantic matching.
# --------------------------------------------------------------------------- #
def test_normal_quoted_citation_still_extracts_cleanly():
    assert extract_citations('The plan is on file (see "7. Concentrations").') == [
        "7. Concentrations"
    ]


def test_quoted_label_containing_parentheses_extracts_entire_label():
    text = 'Contact your advisor (see "Policy (revised 2024) applies").'
    assert extract_citations(text) == ["Policy (revised 2024) applies"]


def test_q013_body_text_citation_with_internal_parens_is_not_truncated():
    text = (
        "Students may transfer to the MISM (non-BIDA) program. The "
        "pre-matriculation transfer deadline is June 1. The "
        'post-matriculation transfer deadline is October 15 (see '
        '"Students may transfer to the MISM (non-BIDA) program").'
    )
    cited = extract_citations(text)
    assert cited == ["Students may transfer to the MISM (non-BIDA) program"]
    # the historical bug truncated the label right before ") program" --
    # assert the exact buggy (truncated) string is NOT what was extracted.
    assert cited != ["Students may transfer to the MISM (non-BIDA"]


def test_existing_valid_citation_behavior_is_unchanged():
    """Locks that the boundary fix didn't alter any previously-correct case:
    plain quoted citation, multi-citation-in-one-block, and no-citation."""
    assert extract_citations('See the handbook (see "4.1. Core Courses").') == [
        "4.1. Core Courses"
    ]
    assert extract_citations(
        'Two sources (see "4.1. Core Courses" and "4.2. Electives").'
    ) == ["4.1. Core Courses", "4.2. Electives"]
    assert extract_citations("No citation here at all.") == []


def test_extraction_never_resolves_against_a_corpus_it_is_pure_text_pulling():
    """extract_citations() has no concept of 'valid' or 'resolved' -- it only
    pulls candidate label text. Resolution (exact-match only, never fuzzy) is
    a separate concern in heinzy.eval.citation_resolution."""
    import inspect

    assert "retrieved" not in inspect.signature(extract_citations).parameters
    assert "context" not in inspect.signature(extract_citations).parameters
    # a citation to a totally invented section still just extracts as text
    assert extract_citations('(see "99.9. Invented Section")') == ["99.9. Invented Section"]


def test_citation_matches_on_section_leaf_or_full_path():
    paths = ["Handbook > 4. Curriculum > 4.1. Core Courses"]
    assert unsupported_citations(["4.1. Core Courses"], paths) == []
    assert unsupported_citations(["Handbook > 4. Curriculum > 4.1. Core Courses"], paths) == []
    assert unsupported_citations(["9.9. Invented Section"], paths) == ["9.9. Invented Section"]


@pytest.mark.parametrize(
    "reply",
    [
        "INSUFFICIENT_CONTEXT",
        # What llama3.2 actually returns when told to emit the underscored
        # token. An exact-match check scored all 8 out-of-corpus refusals as
        # answers, i.e. layer 2 silently did nothing.
        "INSUFFICIENT CONTEXT",
        "insufficient_context",
        "**INSUFFICIENT_CONTEXT**",
        "INSUFFICIENT-CONTEXT.",
        "The excerpts provide INSUFFICIENT CONTEXT to answer this question.",
    ],
)
def test_sentinel_survives_model_reformatting(cfg, stub_model, reply):
    stub_model(reply)
    answer = Generator(cfg).generate("unanswerable question", HITS)
    assert answer.refused, f"missed refusal phrased as {reply!r}"
    assert answer.refusal_reason == REASON_INSUFFICIENT


@pytest.mark.parametrize(
    "text",
    [
        # Models echo the `[section] (pN)` context format into their citations.
        'Up to four electives (see 4.2 Electives, p[13]).',
        'See (see "4.2. Electives", page 13).',
        'See (see "4.2. Electives", pp. 13-14).',
    ],
)
def test_page_markers_are_not_citations(text):
    """A page number cites nothing; flagging it invented a grounding failure."""
    cited = extract_citations(text)
    assert cited, "the real section citation should still be extracted"
    assert all("13" not in c for c in cited), f"page marker leaked into {cited}"
    assert unsupported_citations(
        cited, ["Handbook > 4. Curriculum > 4.2. Electives"]
    ) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("According to section 7.2, a 3.0 QPA is required.", ["7.2"]),
        ("According to Section 5.1 of the Fixture Handbook, interns work 10 weeks.", ["5.1"]),
        ("See § 4.1 for details.", ["4.1"]),
        # Already cited parenthetically, so don't count the section twice.
        ('Core courses (see "4.1. Core Courses") are in section 4.1.', ["4.1. Core Courses"]),
    ],
)
def test_prose_citations_are_extracted(text, expected):
    """Models cite in prose as often as they use the (see ...) form; missing
    those made the grounding check pass on an empty citation set."""
    assert extract_citations(text) == expected


def test_numeric_citation_matches_whole_tokens_only():
    """Unpadded substring matching let "7.2" match an unrelated "7.21" section."""
    assert unsupported_citations(["7.2"], ["Handbook > 7.21. Something Else"]) == ["7.2"]
    assert unsupported_citations(["7.2"], ["Handbook > 7.2. Good Standing"]) == []


def test_citation_found_in_chunk_text_is_supported():
    """Real subsection headings often live in the chunk body, not section_path.

    The structure pass didn't promote "8.1. Normal Courseload Expectation", and
    the extractor swept page footers into the text; citing either is still
    quoting the retrieved context, not inventing a source.
    """
    paths = ["8. Number of Units per Semester"]
    texts = ["Number of Units per Semester - 8.1. Normal Courseload Expectation "
             "for MISM and MISM-BIDA Students Students typically take 54 units."]
    assert unsupported_citations(
        ["8.1. Normal Courseload Expectation for MISM and MISM-BIDA Students"],
        paths, texts,
    ) == []
    # An invented reference is still caught.
    assert unsupported_citations(["12.3. Tuition and Fees"], paths, texts) == [
        "12.3. Tuition and Fees"
    ]


def test_citations_require_false_disables_the_check(cfg, stub_model):
    """The config key has to actually do something, or it lies about behaviour."""
    stub_model('Tuition is $50,000 (see "12.3. Tuition and Fees").')
    cfg.generation.citations.require = False
    answer = Generator(cfg).generate("what is the tuition?", HITS)

    assert answer.unsupported_citations == []
    assert answer.cited_sections == ["12.3. Tuition and Fees"], \
        "citations should still be extracted and reported, just not enforced"


@pytest.mark.parametrize(
    "text, expected_words",
    [
        # A pointer with no content. Not a refusal, but not an answer either.
        ('(see "7. Concentrations")', 0),
        ("54 units per semester.", 4),
        ('Students take 54 units (see "8. Units").', 4),
    ],
)
def test_substantive_word_count_ignores_citations(text, expected_words):
    from heinzy.generation.grounding import substantive_word_count
    assert substantive_word_count(text) == expected_words


def test_sampling_is_pinned_from_config(cfg, stub_model):
    """An eval that doesn't reproduce isn't evidence, and config_hash promises
    reproducibility. Ollama's default temperature of 0.8 broke both."""
    calls = stub_model("some grounded answer here")
    Generator(cfg).generate("q", HITS)

    options = calls["payload"]["options"]
    assert options["temperature"] == 0.0
    assert options["seed"] == 0


@pytest.mark.parametrize(
    "text, expected",
    [
        # gemma's format: a bare parenthetical with no "see".
        ("Students must intern (9. Internship Requirement).", ["9. Internship Requirement"]),
        ("Typically 54 units. (8.1. Normal Courseload Expectation for MISM Students)",
         ["8.1. Normal Courseload Expectation for MISM Students"]),
        # Course codes and asides are not citations.
        ("Take Responsible AI (94-885, 6 units) this term.", []),
        ("The 36 unit cap (described above) applies.", []),
    ],
)
def test_bare_parenthetical_citations(text, expected):
    """gemma cites without the "see" keyword; missing those made its perfect
    grounding score vacuous, with nothing being checked at all."""
    assert extract_citations(text) == expected


def test_basic_auth_is_sent_when_configured(cfg, stub_model, monkeypatch):
    """The shared host sits behind a tunnel, and Ollama has no auth of its own.

    Credentials come from the environment rather than config, since config is
    committed and the tunnel URL and password are not.
    """
    monkeypatch.setenv("MODEL_BASIC_AUTH", "heinz:secret")
    calls = stub_model("some answer")
    Generator(cfg).generate("q", HITS)
    assert calls["auth"] == ("heinz", "secret")


def test_no_auth_sent_when_unset(cfg, stub_model, monkeypatch):
    monkeypatch.delenv("MODEL_BASIC_AUTH", raising=False)
    calls = stub_model("some answer")
    Generator(cfg).generate("q", HITS)
    assert calls["auth"] is None
