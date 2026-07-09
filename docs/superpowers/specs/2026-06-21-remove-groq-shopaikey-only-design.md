# Remove Groq, Keep ShopAIKey Only Design

## Goal

Remove all Groq-specific code, configuration, dependencies, tests, and documentation from the project while preserving the current LLM data contracts, retry behavior, parse-repair flow, and service-level error semantics.

## Scope

- Remove Groq as a supported provider from backend runtime code.
- Keep ShopAIKey as the only hosted LLM provider used by the application.
- Preserve current `LLMResponse` shape and the service-level exceptions used by callers.
- Preserve resume parsing retry flow, JSON repair prompts, trace logging, and scoring JSON retry behavior.
- Update environment samples, compose defaults, dependency manifests, and user-facing docs so they no longer reference Groq.

## Non-Goals

- Reworking the broader LLM abstraction beyond what is needed to remove Groq.
- Changing Ollama support unless a direct Groq cleanup requires a small compatibility adjustment.
- Changing scoring, parsing, or chat output schemas.

## Design

### LLM runtime

`backend/src/services/llm_service.py` will keep the provider wrapper and shared retry helpers, but Groq-specific code paths will be removed. `LLMProvider` will default to `shopaikey` and construct a ShopAIKey adapter for hosted requests. Existing retry, rate-limit detection, backoff, and `LLMProviderLimitError` behavior will stay intact so upstream flows keep the same failure contract.

### Resume parsing retries

`backend/src/services/resume_service.py` currently rotates Groq and ShopAIKey during text JSON parsing retries. That provider sequence will be replaced with repeated ShopAIKey attempts that keep the same prompt-repair flow, attempt counting, and trace logging fields. The goal is to preserve the data flow and observability without preserving Groq-specific branching.

### Scoring JSON retries

`backend/src/services/score_candidate.py` currently contains Groq/llama-specific switching heuristics. Those heuristics will be removed. JSON retry behavior will remain, but retries will stay on the same ShopAIKey-backed provider instance or model selection path.

### Config and docs

Groq environment keys, model settings, and dependency declarations will be removed from config, examples, compose files, and docs. ShopAIKey settings become the canonical hosted LLM configuration.

## Risks and Controls

- Risk: retry behavior changes can alter parsing success rate.
  Control: rewrite tests around attempt count, prompt repair flow, and trace metadata before code changes.
- Risk: removing Groq breaks callers relying on provider labels.
  Control: preserve provider/model fields and update tests to assert the new ShopAIKey-only contract.
- Risk: stale docs or env keys mislead deployment.
  Control: search the repo for remaining Groq references after code changes and update the operational docs in the same pass.
