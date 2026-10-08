"""The retained Hindsight provider must not enter a historical updater shim."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from plugins.memory.hindsight import dependencies as deps


def test_uv_lookup_does_not_enter_retired_updater(monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_cli import managed_uv
    import pm

    takeover = Mock(side_effect=AssertionError("runtime dependency lookup entered updater"))
    monkeypatch.setattr(managed_uv, "stop_for_relaunch", takeover)
    monkeypatch.setattr(pm, "uv_launcher", lambda name: Path("/fixture/uv"))
    monkeypatch.setattr(deps.shutil, "which", lambda name: "/fallback/uv")
    assert deps._uv_binary() == "/fixture/uv"
    takeover.assert_not_called()


def test_compatible_client_is_not_downgraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "_installed_version", lambda spec: "0.9.3")
    install = Mock(side_effect=AssertionError("compatible client must not be reinstalled"))
    monkeypatch.setattr(deps, "_venv_pip_install", install)
    deps.ensure("memory.hindsight", prompt=False)
    install.assert_not_called()


@pytest.mark.parametrize("spec", ["unrelated-package>=1,<2", "hindsight-client;evil", "https://example.com/pkg.whl"])
def test_private_installer_rejects_other_packages(spec: str, monkeypatch: pytest.MonkeyPatch) -> None:
    install = Mock(side_effect=AssertionError("untrusted spec reached installer"))
    monkeypatch.setattr(deps, "_venv_pip_install", install)
    result = deps.install_specs([spec])
    assert result.blocked and not result.ok
    install.assert_not_called()


def test_disabled_install_never_spawns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "_allow_lazy_installs", lambda: False)
    install = Mock(side_effect=AssertionError("disabled install reached installer"))
    monkeypatch.setattr(deps, "_venv_pip_install", install)
    result = deps.install_specs(["hindsight-all>=0.6.1,<1"])
    assert result.blocked and not result.ok
    install.assert_not_called()


def test_installer_failure_is_not_reported_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "_allow_lazy_installs", lambda: True)
    monkeypatch.setattr(deps, "_venv_pip_install", lambda *a, **kw: deps._InstallResult(False, "", "resolution failed"))
    result = deps.install_specs(["hindsight-client>=0.6.1,<1"])
    assert not result.ok and not result.blocked
    assert result.stderr == "resolution failed"


def test_provider_discovery_retains_bundled_hindsight() -> None:
    from plugins.memory import discover_memory_providers, load_memory_provider

    assert "hindsight" in [row[0] for row in discover_memory_providers()]
    provider = load_memory_provider("hindsight")
    assert provider.name == "hindsight"
