# Remove Groq, Keep ShopAIKey Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Groq-specific code and references while keeping the current LLM retry, parsing, and error contracts stable on ShopAIKey.

**Architecture:** Keep the existing LLM service boundary and shared retry helpers, but collapse hosted-provider behavior onto ShopAIKey only. Preserve caller-facing response/error contracts and adjust retry sequences in resume parsing and scoring so they no longer depend on Groq-specific switching.

**Tech Stack:** Python, pytest, FastAPI, urllib, Pydantic settings, Docker Compose

---

### Task 1: Lock retry contract with failing tests

**Files:**
- Modify: `backend/tests/test_llm_service_error_handling.py`
- Modify: `backend/tests/test_resume_service_public_fallback.py`

- [ ] **Step 1: Write the failing test**

```python
def test_shopaikey_adapter_retries_rate_limits_before_raising_limit_error(...):
    ...

def test_generate_resume_json_with_retries_uses_shopaikey_only_sequence(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_llm_service_error_handling.py backend/tests/test_resume_service_public_fallback.py -k "shopaikey or sequence" -v`
Expected: FAIL because the current code still assumes Groq-specific adapters and retry order.

- [ ] **Step 3: Write minimal implementation**

```python
class _ShopAIKeyAdapter(_BaseAdapter):
    ...

def _resume_text_parse_provider_specs() -> list[tuple[str, str]]:
    return [
        (ProviderType.SHOPAIKEY.value, settings.RESUME_PARSE_MODEL_NAME),
        (ProviderType.SHOPAIKEY.value, settings.SHOPAIKEY_MODEL_NAME),
        ...
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_llm_service_error_handling.py backend/tests/test_resume_service_public_fallback.py -k "shopaikey or sequence" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_llm_service_error_handling.py backend/tests/test_resume_service_public_fallback.py backend/src/services/llm_service.py backend/src/services/resume_service.py
git commit -m "refactor: preserve retry contracts on shopaikey"
```

### Task 2: Remove Groq runtime paths from LLM services

**Files:**
- Modify: `backend/src/services/llm_service.py`
- Modify: `backend/src/services/score_candidate.py`
- Modify: `backend/src/services/resume_service.py`

- [ ] **Step 1: Write the failing test**

```python
def test_llm_provider_defaults_to_shopaikey(...):
    ...

def test_generate_json_with_retries_stays_on_shopaikey(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_llm_service_error_handling.py backend/tests/test_resume_service_public_fallback.py backend/tests/test_score_candidate_service.py -k "defaults_to_shopaikey or stays_on_shopaikey" -v`
Expected: FAIL because the provider wrapper still defaults to Groq and scoring still has Groq-specific switching.

- [ ] **Step 3: Write minimal implementation**

```python
class ProviderType(str, Enum):
    SHOPAIKEY = "shopaikey"
    OLLAMA = "ollama"

selected_provider = (provider or settings.LLM_PROVIDER or ProviderType.SHOPAIKEY.value).lower()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_llm_service_error_handling.py backend/tests/test_resume_service_public_fallback.py backend/tests/test_score_candidate_service.py -k "defaults_to_shopaikey or stays_on_shopaikey" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/llm_service.py backend/src/services/score_candidate.py backend/src/services/resume_service.py backend/tests/test_score_candidate_service.py
git commit -m "refactor: remove groq runtime paths"
```

### Task 3: Remove Groq config, dependency, and docs

**Files:**
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `backend/requirements.txt`
- Modify: `docker-compose.yml`
- Modify: `QUICKSTART.md`
- Modify: `docs/DEPLOY_1VM_SIMPLE_GUIDE.md`
- Modify: `docs/INTEGRATION_PLAN.md`

- [ ] **Step 1: Write the failing test**

```python
def test_repo_has_no_groq_runtime_references():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rg -n "Groq|groq|GROQ_" .`
Expected: existing matches in env, docs, and requirements.

- [ ] **Step 3: Write minimal implementation**

```text
LLM_PROVIDER=shopaikey
SHOPAIKEY_API_KEY=...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rg -n "Groq|groq|GROQ_" .`
Expected: no remaining runtime/config/doc references except intentionally updated historical notes if any remain.

- [ ] **Step 5: Commit**

```bash
git add .env.example backend/.env.example backend/requirements.txt docker-compose.yml QUICKSTART.md docs/DEPLOY_1VM_SIMPLE_GUIDE.md docs/INTEGRATION_PLAN.md
git commit -m "chore: remove groq configuration references"
```

### Task 4: Final verification

**Files:**
- Modify: none

- [ ] **Step 1: Run targeted backend tests**

Run: `python -m pytest backend/tests/test_llm_service_error_handling.py backend/tests/test_resume_service_public_fallback.py backend/tests/test_score_candidate_service.py backend/tests/test_score_candidate_error_handling.py -v`
Expected: PASS

- [ ] **Step 2: Run focused reference scan**

Run: `rg -n "Groq|groq|GROQ_" backend .env.example backend/.env.example QUICKSTART.md docs docker-compose.yml`
Expected: no relevant matches.

- [ ] **Step 3: Review changed files**

Run: `git diff -- backend/src/services/llm_service.py backend/src/services/resume_service.py backend/src/services/score_candidate.py backend/src/core/config.py`
Expected: Groq paths removed, retry and error contracts preserved.

- [ ] **Step 4: Commit**

```bash
git add backend/src/core/config.py
git commit -m "test: verify shopaikey-only llm configuration"
```
