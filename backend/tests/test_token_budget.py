from src.services.token_budget import BudgetWindow, estimate_json_tokens, estimate_tokens, fits_budget


def test_estimate_tokens_uses_conservative_char_ratio():
    assert estimate_tokens("abcd") == 2
    assert estimate_tokens("") == 0


def test_estimate_json_tokens_counts_serialized_payload():
    payload = {"candidate": {"id": "1", "skills_text": "Python FastAPI"}}
    assert estimate_json_tokens(payload) >= estimate_tokens('"skills_text"')


def test_budget_window_computes_input_budget_after_output_and_reserve():
    window = BudgetWindow(context_window=8192, output_budget=2048, reserve=512)
    assert window.input_budget == 5632


def test_fits_budget_rejects_payload_over_input_budget():
    window = BudgetWindow(context_window=100, output_budget=30, reserve=10)
    assert fits_budget(estimate_tokens("x" * 500), window) is False
