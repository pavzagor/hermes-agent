"""Real Chromium fixture; dummy vault only. Run with the canonical test runner.
Requires playwright plus `python -m playwright install chromium`.
"""
import json
from types import SimpleNamespace

import pytest

from agent.vault_store import VaultStore
from tools import browser_vault_tool as vault


@pytest.fixture
def browser(monkeypatch, tmp_path):
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as p:
        chromium = p.chromium.launch()
        page = chromium.new_page()
        page.route("**/*", lambda route: route.fulfill(body="<html></html>", content_type="text/html"))
        page.goto("https://fixture.test/users/register/")
        cdp = page.context.new_cdp_session(page)
        def evaluate(expression):
            result = cdp.send("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            assert "exceptionDetails" not in result
            return {"ok": True, "result": result["result"].get("value")}
        supervisor = SimpleNamespace(evaluate_runtime=evaluate)
        monkeypatch.setattr(vault, "_ensure_supervisor", lambda task: supervisor)
        monkeypatch.setattr(vault, "_eval_js", lambda task, expr: {"success": True, "result": evaluate(expr)["result"]})
        monkeypatch.setattr(vault, "_focus_bound_origin", lambda *args: None)
        store = VaultStore(base_dir=tmp_path / "vault")
        item = store.add_item("login", "dummy", {"identifier": "dummy@example.test", "identifier_type": "email", "password": "dummy-fixture-secret"}, origin="https://fixture.test")
        backend = SimpleNamespace(needs_unlock=False, name="fixture", get_meta=store.get_meta, resolve_password=lambda handle: store.resolve_secret(handle)["password"])
        monkeypatch.setattr("agent.vault_backends.backend_for_handle", lambda handle: backend)
        def fill():
            raw = vault._handle_vault_fill({"handle": item.id}, task_id="fixture")
            assert "dummy-fixture-secret" not in raw
            return json.loads(raw)
        yield page, fill
        chromium.close()


@pytest.mark.parametrize("variant", ["evidea", "autocomplete", "reset", "current", "other_form", "ambiguous", "hidden", "opacity"])
def test_signup_pair_is_same_form_visible_and_unambiguous(browser, variant):
    page, fill = browser
    first = '<input type=password name=password>'
    second = '<input type=password name=password-repeat>'
    action = "/users/register/"
    extra = '<form action=/login><input type=password name=unrelated autocomplete=current-password></form>'
    if variant == "autocomplete":
        first = '<input type=password name=a autocomplete=new-password>'
        second = '<input type=password name=b autocomplete=new-password>'
    if variant == "reset": action = "/reset-password/"
    if variant == "current": first = '<input type=password name=password autocomplete=current-password>'
    if variant == "other_form": second = '</form><form action=/users/register/>' + second
    if variant == "ambiguous": second += '<input type=password name=another>'
    if variant == "hidden": second = '<div hidden>' + second + '</div>'
    if variant == "opacity": second = '<div style="opacity:0">' + second + '</div>'
    page.set_content(f'<form action="{action}">{first}{second}</form>{extra}')
    result = fill()
    values = page.locator('input').evaluate_all('(els) => els.map(e => Boolean(e.value))')
    if variant in ("evidea", "autocomplete"):
        assert result["filled_fields"] == 2
        assert values == [True, True, False]
        assert page.locator('input').evaluate_all("els => els.slice(0, 2).every(e => e.value === 'dummy-fixture-secret')")
    else:
        assert not any(values)


@pytest.mark.parametrize("mutation", ["origin", "hidden", "form", "current", "type", "disabled", "focus"])
def test_fill_rechecks_origin_and_pair_at_secret_write(browser, monkeypatch, mutation):
    page, fill = browser
    page.set_content('<form action=/users/register/><input type=password name=password><input type=password name=password-repeat></form><form id=other></form>')
    original = vault._eval_js_secret
    def mutate(task, expression):
        if mutation == "origin": page.goto("https://other.test/users/register/")
        else:
            js = {"hidden": "e.style.display='none'", "form": "e.setAttribute('form','other')", "current": "e.autocomplete='current-password'", "type": "e.type='text'", "disabled": "e.disabled=true", "focus": "document.querySelector('[name=password]').addEventListener('focus', event => {event.target.type='text'}, {once:true})"}[mutation]
            page.evaluate("() => { const e=document.querySelector('[name=password-repeat]'); " + js + "; }")
        return original(task, expression)
    monkeypatch.setattr(vault, "_eval_js_secret", mutate)
    result = fill()
    assert not result["success"]
    assert not page.locator('input').evaluate_all('(els) => els.some(e => Boolean(e.value))')
