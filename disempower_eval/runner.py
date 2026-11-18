"""Play each scenario's scripted user turns against the target model."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import anthropic

from .models import Scenario, Transcript, Turn

DEFAULT_TARGET_MODEL = "claude-opus-5-5"


def _text_of(response: anthropic.types.Message) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def run_scenario(
    client: anthropic.Anthropic,
    scenario: Scenario,
    *,
    model: str,
    sample: int = 0,
    effort: str | None = None,
    system: str | None = None,
) -> Transcript:
    """Send user turns one at a time; assistant turns are the target's live replies."""
    transcript = Transcript(scenario_id=scenario.id, sample=sample, target_model=model)
    messages: list[anthropic.types.MessageParam] = []

    extra: dict = {}
    if effort:
        extra["output_config"] = {"effort": effort}
    if system:
        extra["system"] = system

    for user_text in scenario.turns:
        messages.append({"role": "user", "content": user_text})
        transcript.turns.append(Turn(role="user", text=user_text))
        try:
            response = client.messages.create(model=model, max_tokens=16000, messages=messages, **extra)
        except anthropic.APIStatusError as e:
            transcript.error = f"{type(e).__name__} {e.status_code}: {e.message}"
            return transcript
        except anthropic.APIConnectionError as e:
            transcript.error = f"APIConnectionError: {e}"
            return transcript

        transcript.turns.append(Turn(role="assistant", text=_text_of(response), stop_reason=response.stop_reason))
        if response.stop_reason == "refusal":
            # A refusal is itself a measurable behavior; record it and stop the conversation.
            return transcript
        # Pass back the full content (including thinking blocks) unchanged.
        messages.append({"role": "assistant", "content": response.content})

    return transcript


def run_all(
    client: anthropic.Anthropic,
    scenarios: list[Scenario],
    *,
    model: str = DEFAULT_TARGET_MODEL,
    samples: int = 1,
    effort: str | None = None,
    system: str | None = None,
    concurrency: int = 4,
    on_done=None,
) -> list[Transcript]:
    jobs = [(s, i) for s in scenarios for i in range(samples)]

    def work(job: tuple[Scenario, int]) -> Transcript:
        scenario, sample = job
        t = run_scenario(client, scenario, model=model, sample=sample, effort=effort, system=system)
        if on_done:
            on_done(t)
        return t

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(work, jobs))
