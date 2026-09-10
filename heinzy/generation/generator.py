"""
Grounded answer generation over retrieved chunks.

pre: hits is a list of ScoredChunk from a completed retrieval, already filtered
     by retrieval.score_floor. It may be empty, which is the primary refusal path.
post: returns an Answer carrying the model's text plus the exact ScoredChunks it
      was grounded in, so the caller can cite/verify sources. `refused` says
      whether the system declined to answer, and `refusal_reason` says which
      layer declined.
invariant: model_tag, endpoint, and every abstention knob are read from config,
           never hardcoded, mirroring the "no tunable constants in source" rule.
           A local MODEL_TAG env var overrides config.model.tag for testing
           against whatever is actually pulled on this machine.

Grounded answering means two things, enforced separately:

  1. Answers are built only from retrieved chunks. The prompt forbids outside
     knowledge, and every section the answer cites is checked against what
     retrieval actually returned (heinzy/generation/grounding.py).

  2. When the corpus does not contain the answer, the system SAYS SO instead of
     producing plausible text. Two layers, because either alone leaks:

       Layer 1 covers the no-context case. Fewer than
       generation.abstain.min_hits survived retrieval.score_floor, so the model
       is never called. This is deterministic. A question the corpus cannot
       support never reaches the model at all.

       Layer 2 covers insufficient context. Chunks cleared the floor but do not
       actually answer the question. Only the model can judge that, so it is
       instructed to emit an exact sentinel, which we detect and convert into
       the same refusal. Nearest-neighbour search always returns *something*,
       so without this layer a well-scoring but irrelevant chunk is exactly
       what a confident, wrong answer gets built from.

A refusal returns the configured refusal_text, not model prose, so downstream
(eval harness, event log, UI) can branch on `Answer.refused` instead of pattern
matching on English.

When governance is available (heinzy/tools mount present and not disabled in
config), generate() enters a tool-calling loop. Every tool_call is gated by
run_governed_tool before the tool runs. Works for both Ollama and Azure
(OpenAI-format tools; Azure responses are normalized to the shared loop shape).

Generation providers (MODEL_PROVIDER env, default "ollama"):
  - ollama: POST {MODEL_ENDPOINT}/api/chat (local or LAN GPU host).
  - azure_openai: Azure OpenAI / AI Foundry Chat Completions. Set
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from heinzy.generation.abstain import detects_refusal
from heinzy.generation.grounding import extract_citations, unsupported_citations
from heinzy.generation.policy import REASON_NO_CONTEXT, get_context_policy
from heinzy.governance.loader import governance_available
from heinzy.retrieval.store import ScoredChunk
from heinzy.tools.registry import TOOL_DEFINITIONS, run_governed_tool

_DEFAULT_ENDPOINT = "http://localhost:11434"
_DEFAULT_PROVIDER = "ollama"
_DEFAULT_AZURE_API_VERSION = "2024-10-21"
_DEFAULT_SENTINEL = "INSUFFICIENT_CONTEXT"
_DEFAULT_REFUSAL = (
    "I can't answer that from the MISM handbook. The handbook sections I can "
    "search don't contain the information needed to answer this question. "
    "Please check with Heinz College advising staff directly."
)

# Why the refusal reasons are named, not booleans: an advisor seeing "no
# handbook section came close" needs a different follow-up than "the handbook
# covers this area but not this detail", and the eval harness reports on which
# layer fired. REASON_NO_CONTEXT is re-exported from the policy module so
# callers keep one import site for both layers' reasons.
REASON_INSUFFICIENT = "model_insufficient_context"


def _basic_auth_from_env() -> tuple[str, str] | None:
    """Read MODEL_BASIC_AUTH as user:password, or None when the tunnel is open."""
    raw = (os.environ.get("MODEL_BASIC_AUTH") or "").strip()
    if not raw or ":" not in raw:
        return None
    user, _, password = raw.partition(":")
    return user, password


# The primary workflow is an advisor looking up policy on a student's behalf,
# not a student asking the assistant directly. Keep the framing accurate to that
# audience. An advisor repeating an invented policy to a student is the exact
# failure this prompt exists to prevent.
SYSTEM_PROMPT_TEMPLATE = (
    "You are an assistant supporting academic advisors for the Heinz College "
    "MISM program. An advisor is asking you a question, often on behalf of a "
    "student, and needs an accurate answer they can rely on when advising.\n\n"
    "RULES:\n"
    "1. Answer using ONLY the handbook excerpts provided below. Never use "
    "outside knowledge, general knowledge about universities, or anything you "
    "know about Carnegie Mellon that is not in the excerpts.\n"
    "2. If the excerpts do not contain the information needed to answer the "
    "question, reply with exactly {sentinel} and nothing else. Do not "
    "apologise, do not explain, do not offer a partial guess, and do not "
    "suggest what the answer is likely to be. A wrong answer is far worse than "
    "no answer, because the advisor will repeat it to a student.\n"
    "3. This applies even when the question sounds like something a handbook "
    "would obviously cover, and even when the excerpts are about a related "
    "topic. Related is not the same as answering the question.\n"
    # The example here is a PLACEHOLDER on purpose. An earlier version used a
    # realistic sample sentence about elective limits, and llama3.2 answered a
    # real question by copying it verbatim, inventing a section that does not
    # exist in this handbook. Never put plausible-looking content in the
    # instructions, because the model cannot tell your example from its
    # evidence.
    "4. When the excerpts answer the question, and only then, write the "
    "answer itself as a sentence, then cite the section(s) you used in the form "
    '(see "<exact section heading copied from the excerpts>"). A bare citation '
    "with no sentence is not an answer; the advisor needs the substance, not a "
    "pointer. Never cite a section that does not appear in the excerpts below, "
    "and never reuse a section name from these instructions.\n"
    "5. Rule 4 never overrides rule 2. If the excerpts do not answer the "
    "question, {sentinel} is the only correct response. Writing a fluent "
    "sentence is not the goal, being right is.\n"
    "6. Base your answer on ALL relevant information present in the excerpts "
    "-- do not silently omit part of what's there. An advisor relying on an "
    "incomplete answer could give a student incorrect information."
)


@dataclass
class Answer:
    query: str
    text: str
    model_tag: str
    sources: list[ScoredChunk]
    paused_for_approval: bool = False
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    # --- grounding / abstention contract ---
    refused: bool = False
    refusal_reason: str | None = None
    cited_sections: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    # Model's untouched output, kept when `text` was replaced by refusal_text so
    # a reviewer can see what the model actually said.
    raw_text: str | None = None
    # Which Layer 1 engine decided, builtin or agt. Stamped so a run's audit
    # trail says whether the policy engine was actually in the loop.
    policy_engine: str | None = None
    # Best-effort provider token usage for the single chat call that produced
    # `text`. None when no call was made (Layer 1 refusal) or when the
    # provider response didn't report it -- see extract_usage() below for
    # exactly which raw response fields this comes from per provider.
    usage: dict[str, int | None] | None = None

    @property
    def is_grounded(self) -> bool:
        """True when the answer cites nothing outside the retrieved set.

        A refusal is trivially grounded, since it makes no claims at all.
        """
        return not self.unsupported_citations


def extract_usage(provider: str, data: dict[str, Any]) -> dict[str, int | None]:
    """Best-effort token usage from a raw provider chat response.

    Ollama's /api/chat response carries prompt_eval_count/eval_count at the
    top level (_chat_ollama below returns resp.json() directly). Azure/
    OpenAI-compatible chat completions carry a "usage" object inside the raw
    response body (_chat_azure returns {"message": ..., "raw": data}). Any
    field the provider response does not actually contain comes back as
    None -- usage is never invented, per experiment provenance requirements.
    """
    if provider == "azure_openai":
        usage = (data.get("raw") or {}).get("usage") or {}
        return {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    input_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")
    total_tokens = (
        input_tokens + output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else None
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


class Generator:
    def __init__(self, cfg) -> None:
        self.provider = (os.environ.get("MODEL_PROVIDER") or _DEFAULT_PROVIDER).strip().lower()
        self.model_tag = os.environ.get("MODEL_TAG") or cfg.model.tag
        self.endpoint = (getattr(cfg.model, "endpoint", "") or _DEFAULT_ENDPOINT).rstrip("/")

        # The shared host runs behind a tunnel, and Ollama has no auth of its
        # own, so the tunnel should carry basic auth. Credentials come from the
        # environment and never from config, which is committed.
        self.auth = _basic_auth_from_env()

        # Azure OpenAI / Azure AI Foundry (MODEL_PROVIDER=azure_openai).
        # Secrets stay in env. Supports:
        #   classic: https://{resource}.openai.azure.com + deployments/{name}/...
        #   Foundry v1: https://{resource}.services.ai.azure.com/openai/v1/chat/completions
        #               with {"model": deployment} in the body
        raw_azure = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        if raw_azure.endswith("/chat/completions"):
            raw_azure = raw_azure[: -len("/chat/completions")].rstrip("/")
        self.azure_endpoint = raw_azure
        self.azure_api_key = (os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
        self.azure_deployment = (
            os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or os.environ.get("MODEL_TAG")
            or ""
        ).strip()
        self.azure_api_version = (
            os.environ.get("AZURE_OPENAI_API_VERSION") or _DEFAULT_AZURE_API_VERSION
        ).strip()
        # Foundry / AI Services v1 chat URL (model name goes in the JSON body).
        self.azure_v1_chat_url: str | None = None
        if raw_azure:
            if raw_azure.endswith("/openai/v1"):
                self.azure_v1_chat_url = f"{raw_azure}/chat/completions"
                self.azure_endpoint = raw_azure[: -len("/openai/v1")].rstrip("/")
            elif "services.ai.azure.com" in raw_azure:
                self.azure_v1_chat_url = (
                    f"{raw_azure}/openai/v1/chat/completions"
                )
        if self.provider == "azure_openai":
            missing = [
                name
                for name, val in (
                    ("AZURE_OPENAI_ENDPOINT", self.azure_endpoint or raw_azure),
                    ("AZURE_OPENAI_API_KEY", self.azure_api_key),
                    ("AZURE_OPENAI_DEPLOYMENT", self.azure_deployment),
                )
                if not val
            ]
            if missing:
                raise ValueError(
                    "MODEL_PROVIDER=azure_openai requires: " + ", ".join(missing)
                )
            # Stamp answers with the deployment name unless MODEL_TAG overrides.
            if not os.environ.get("MODEL_TAG"):
                self.model_tag = self.azure_deployment
            self.endpoint = self.azure_endpoint or raw_azure

        # Tool loop when the governance mount is present and config does not
        # disable it. Ollama and Azure both speak OpenAI-style tool schemas.
        gov = getattr(cfg, "governance", None)
        self._governance_cfg = gov
        enabled = True if gov is None else bool(getattr(gov, "enabled", True))
        self.use_tools = bool(enabled and governance_available())
        self.max_tool_rounds = int(getattr(gov, "max_tool_rounds", 3) if gov else 3)
        self.agent_id = str(getattr(gov, "agent_id", "heinzy-advisor") if gov else "heinzy-advisor")

        # Abstention knobs live in config (S5). getattr chains keep this working
        # against an older config.yaml that predates the generation section.
        generation = getattr(cfg, "generation", None)
        abstain = getattr(generation, "abstain", None)
        self.sentinel = (getattr(abstain, "sentinel", None) or _DEFAULT_SENTINEL).strip()
        self.min_hits = int(getattr(abstain, "min_hits", 1) or 1)
        self.refusal_text = " ".join(
            (getattr(abstain, "refusal_text", None) or _DEFAULT_REFUSAL).split()
        )
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(sentinel=self.sentinel)
        self.system_prompt_with_tools = (
            self.system_prompt
            + "\n\nYou may call tools when needed. Write/create/update/delete/insert "
            "actions are forbidden. Web search is only for official CMU domains and "
            "may require human approval. Prefer handbook excerpts over tools when "
            "they suffice."
        )

        # Layer 1 lives in a policy object rather than an inline condition, so
        # the decision can be evaluated and audited by a governance engine.
        self.policy = get_context_policy(cfg)

        citations = getattr(generation, "citations", None)
        self.require_citations = bool(getattr(citations, "require", True))

        self.temperature = float(getattr(generation, "temperature", 0.0) or 0.0)
        self.seed = getattr(generation, "seed", 0)

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    def _chat(self, messages: list[dict[str, Any]], *, tools: list[dict] | None) -> dict[str, Any]:
        if self.provider == "azure_openai":
            return self._chat_azure(messages, tools=tools)
        if self.provider != "ollama":
            raise ValueError(
                f"Unknown MODEL_PROVIDER={self.provider!r}. "
                "Supported: ollama, azure_openai."
            )
        return self._chat_ollama(messages, tools=tools)

    def _chat_ollama(
        self, messages: list[dict[str, Any]], *, tools: list[dict] | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_tag,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        if tools:
            payload["tools"] = tools
        resp = requests.post(
            f"{self.endpoint}/api/chat",
            json=payload,
            timeout=180,
            auth=self.auth,
        )
        resp.raise_for_status()
        return resp.json()

    def _chat_azure(
        self, messages: list[dict[str, Any]], *, tools: list[dict] | None
    ) -> dict[str, Any]:
        """Azure OpenAI / AI Foundry Chat Completions → shared message shape."""
        clean_messages = self._messages_for_azure(messages)
        headers = {
            "api-key": self.azure_api_key,
            "Content-Type": "application/json",
        }
        if self.azure_v1_chat_url:
            # Azure AI Foundry / AI Services: model name in body, no deployment path.
            url = self.azure_v1_chat_url
            payload: dict[str, Any] = {
                "model": self.azure_deployment,
                "messages": clean_messages,
                "temperature": self.temperature,
            }
        else:
            url = (
                f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}"
                f"/chat/completions?api-version={self.azure_api_version}"
            )
            payload = {
                "messages": clean_messages,
                "temperature": self.temperature,
            }
        # Match Ollama: pin sampling from config so Azure evals are reproducible.
        if self.seed is not None:
            payload["seed"] = self.seed
        if tools:
            payload["tools"] = tools
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=180,
        )
        if resp.status_code >= 400:
            # Surface Azure's message; bare HTTPError hides DeploymentNotFound detail.
            detail = (resp.text or "")[:500]
            raise requests.HTTPError(
                f"{resp.status_code} for {url}: {detail}", response=resp
            )
        data = resp.json()
        try:
            choice_msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Azure OpenAI response shape: {data!r}"
            ) from exc
        # Normalize so generate() / tool loop keep reading data["message"].
        content = choice_msg.get("content")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content if content is not None else "",
        }
        tool_calls = choice_msg.get("tool_calls") or []
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"message": message, "raw": data}

    @staticmethod
    def _messages_for_azure(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map the shared tool-loop message list to Azure/OpenAI Chat Completions."""
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role in ("system", "user"):
                out.append({"role": role, "content": m.get("content") or ""})
                continue
            if role == "assistant":
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.get("content") if m.get("content") is not None else "",
                }
                raw_tcs = m.get("tool_calls") or []
                if raw_tcs:
                    azure_tcs: list[dict[str, Any]] = []
                    for i, tc in enumerate(raw_tcs):
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        name = fn.get("name") or tc.get("name") or ""
                        args = fn.get("arguments", tc.get("arguments", {}))
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        elif args is None:
                            args = "{}"
                        else:
                            args = str(args)
                        tc_id = tc.get("id") or f"call_{i}"
                        azure_tcs.append(
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args},
                            }
                        )
                    msg["tool_calls"] = azure_tcs
                    if not msg["content"]:
                        msg["content"] = None
                out.append(msg)
                continue
            if role == "tool":
                tool_call_id = m.get("tool_call_id") or m.get("id") or "call_unknown"
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": m.get("content") or "",
                    }
                )
        return out

    @staticmethod
    def _tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize Ollama/OpenAI tool_calls (name/arguments may be nested under 'function')."""
        raw = message.get("tool_calls") or []
        out: list[dict[str, Any]] = []
        for i, tc in enumerate(raw):
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name") or ""
            args = fn.get("arguments", tc.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"value": args}
            tc_id = tc.get("id") or f"call_{i}"
            out.append({"name": name, "arguments": args, "id": tc_id, "raw": tc})
        return out

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, query: str, hits: list[ScoredChunk]) -> Answer:
        # Layer 1. Nothing relevant retrieved means no grounded answer is
        # possible, so the model is never called. Asking it anyway only invites
        # an answer that isn't grounded.
        decision = self.policy.decide(len(hits))
        if not decision.allowed:
            return self._refusal(
                query, hits, decision.reason, policy_engine=decision.engine
            )

        context = "\n\n".join(
            f"[{h.section_path}] (p{h.source_pages}): {h.text}" for h in hits
        )
        user_prompt = f"Handbook excerpts:\n\n{context}\n\nQuestion: {query}"

        usage: dict[str, int | None] | None = None
        if self.use_tools:
            raw, tool_events, paused = self._generate_with_tools(query, user_prompt, hits)
            if paused is not None:
                return paused
        else:
            data = self._chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=None,
            )
            raw = data["message"]["content"]
            tool_events = []
            usage = extract_usage(self.provider, data)

        # Layer 2 lives in heinzy/generation/abstain.py. It answers a question
        # no policy engine can, which is whether the excerpts actually contain
        # the answer.
        if detects_refusal(raw, self.sentinel):
            return self._refusal(
                query, hits, REASON_INSUFFICIENT, raw_text=raw,
                policy_engine=decision.engine, tool_events=tool_events, usage=usage,
            )

        cited = extract_citations(raw)
        unsupported = (
            unsupported_citations(
                cited, [h.section_path for h in hits], [h.text for h in hits]
            )
            if self.require_citations
            else []
        )
        return Answer(
            query=query,
            text=raw,
            model_tag=self.model_tag,
            sources=hits,
            cited_sections=cited,
            unsupported_citations=unsupported,
            policy_engine=decision.engine,
            tool_events=tool_events,
            usage=usage,
        )

    def _generate_with_tools(
        self, query: str, user_prompt: str, hits: list[ScoredChunk]
    ) -> tuple[str, list[dict[str, Any]], Answer | None]:
        """Run the provider-agnostic tool-calling loop (Ollama or Azure).

        Returns (raw_text, tool_events, paused_answer). paused_answer is non-None
        only when a tool call paused for human approval, in which case the caller
        should return it directly and skip Layer 2.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt_with_tools},
            {"role": "user", "content": user_prompt},
        ]
        tool_events: list[dict[str, Any]] = []

        for _ in range(self.max_tool_rounds):
            data = self._chat(messages, tools=TOOL_DEFINITIONS)
            message = data.get("message") or {}
            messages.append(message)

            tool_calls = self._tool_calls_from_message(message)
            if not tool_calls:
                return message.get("content") or "", tool_events, None

            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]
                tool_call_id = tc["id"]
                try:
                    result = run_governed_tool(
                        name, args, query=query, agent_id=self.agent_id
                    )
                except PermissionError as exc:
                    tool_events.append(
                        {"tool": name, "args": args, "status": "DENIED", "error": str(exc)}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"status": "DENIED", "error": str(exc)}),
                        }
                    )
                    continue

                if result.get("status") == "PAUSED_FOR_APPROVAL":
                    tool_events.append(
                        {"tool": name, "args": args, "status": "PAUSED_FOR_APPROVAL", "result": result}
                    )
                    paused = Answer(
                        query=query,
                        text=(
                            "Paused for human approval before completing a tool call. "
                            f"Details: {result.get('details')}"
                        ),
                        model_tag=self.model_tag,
                        sources=hits,
                        paused_for_approval=True,
                        tool_events=tool_events,
                    )
                    return "", tool_events, paused

                tool_events.append(
                    {"tool": name, "args": args, "status": "SUCCESS", "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result),
                    },
                )

        # Exhausted rounds — ask once more without tools for a final answer.
        data = self._chat(messages, tools=None)
        message = data.get("message") or {}
        return message.get("content") or "", tool_events, None

    def _refusal(
        self,
        query: str,
        hits: list[ScoredChunk],
        reason: str,
        raw_text: str | None = None,
        policy_engine: str | None = None,
        tool_events: list[dict[str, Any]] | None = None,
        usage: dict[str, int | None] | None = None,
    ) -> Answer:
        return Answer(
            query=query,
            text=self.refusal_text,
            model_tag=self.model_tag,
            sources=list(hits),
            refused=True,
            refusal_reason=reason,
            raw_text=raw_text,
            policy_engine=policy_engine,
            tool_events=tool_events or [],
            usage=usage,
        )
