"""Carried dependency support for the retained in-tree Hindsight provider.

Upstream retired tools.lazy_deps when it moved providers to the plugin catalog.
This provider still needs its environment-aware install/upgrade path, including
sealed-runtime and durable-target gates. Keep that compatibility private here;
all core backends continue to use PM and the old updater shim remains retired.
"""

from __future__ import annotations

import configparser
import contextlib
import logging
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)


# Private compatibility for the retained in-tree Hindsight provider only.
LAZY_DEPS: dict[str, tuple[str, ...]] = {
    "memory.hindsight": ("hindsight-client>=0.6.1,<1",),
}


# Spec validation: name[extras]specifier only — no URLs, paths, or shell metachars.
_NAME_RE = r"[A-Za-z0-9_][A-Za-z0-9_.\-]*"
_NAME_EXTRAS_RE = re.compile(rf"^{_NAME_RE}(?:\[[A-Za-z0-9_,\-]+\])?")
_SAFE_SPEC = re.compile(rf"^{_NAME_RE}(?:\[[A-Za-z0-9_,\-]+\])?(?:[<>=!~]=?[A-Za-z0-9_.\-+,*<>=!~]+)?$")


class FeatureUnavailable(RuntimeError):
    """A lazily-installable feature is missing and cannot be made available (installs disabled or failed)."""

    def __init__(self, feature: str, missing: tuple[str, ...], reason: str):
        self.feature, self.missing, self.reason = feature, missing, reason
        spec_list = " ".join(repr(s) for s in missing)
        super().__init__(
            f"Feature {feature!r} unavailable: {reason}. "
            f"To enable manually: uv pip install {spec_list}  (or: pip install {spec_list}).")


@dataclass(frozen=True)
class _InstallResult:
    success: bool
    stdout: str
    stderr: str


# Internal bridge var (set by the Docker image, not user config) redirecting lazy installs from the
# sealed venv to a writable durable volume.
_LAZY_TARGET_ENV = "HERMES_LAZY_INSTALL_TARGET"
# Stamp of the Python X.Y + ABI the target was populated for; a mismatch after an image rebuild
# wipes the store so stale .so files are never imported.
_TARGET_STAMP_NAME = ".python-abi"

_SUBPROCESS_KW = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)


def _python_abi_tag() -> str:
    """X.Y + EXT_SUFFIX (ABI tag + platform); interpreters sharing compiled wheels match."""
    return f"{sys.version_info.major}.{sys.version_info.minor}:{sysconfig.get_config_var('EXT_SUFFIX') or ''}"


def _lazy_install_target() -> Optional[Path]:
    """Durable install-target dir (:data:`_LAZY_TARGET_ENV`), or None for venv-scoped mode."""
    raw = os.environ.get(_LAZY_TARGET_ENV, "").strip()
    return Path(raw) if raw else None


def _ensure_target_ready(target: Path) -> Optional[str]:
    """Create the target dir and validate its ABI stamp; a different-ABI stamp wipes the contents
    first (stale .so must never import). None on success, else an error string."""
    want = _python_abi_tag()
    stamp = target / _TARGET_STAMP_NAME
    try:
        if target.exists():
            have = ""
            with contextlib.suppress(OSError):
                have = stamp.read_text(encoding="utf-8").strip()
            if have and have != want:
                logger.info("Lazy install target %s was built for ABI %r but running ABI is %r; wiping stale packages.", target, have, want)
                for child in target.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        with contextlib.suppress(OSError):
                            child.unlink()
        target.mkdir(parents=True, exist_ok=True)
        stamp.write_text(want, encoding="utf-8")
    except OSError as e:
        return f"lazy install target {target} is not writable: {e}"
    return None


def _activate_target_on_syspath(target: Path) -> None:
    """Append the durable target to ``sys.path`` (idempotent). ``site.addsitedir`` honours ``.pth``
    files but inserts near the front, so new entries are moved to the END — core venv site-packages must win collisions."""
    target_str = str(target)
    before = list(sys.path)
    if target_str not in before:
        site.addsitedir(target_str)
    new_entries = [p for p in sys.path if p not in before]
    if new_entries:
        sys.path[:] = [p for p in sys.path if p not in new_entries] + new_entries
    _invalidate_import_caches()


def _invalidate_import_caches() -> None:
    """Make just-installed dists visible to importers and importlib.metadata in this process."""
    with contextlib.suppress(Exception):
        import importlib
        importlib.invalidate_caches()
        import importlib.metadata as _md
        if hasattr(_md, "_cache_clear"):
            _md._cache_clear()  # type: ignore[attr-defined]


def activate_durable_lazy_target() -> None:
    """Wire the durable target onto ``sys.path`` at startup so packages installed on a previous run
    import on this one. No-op when unset or absent. Never raises."""
    target = _lazy_install_target()
    try:
        if target is not None and target.exists():
            _activate_target_on_syspath(target)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to activate durable lazy target %s: %s", target, e)


def _allow_lazy_installs() -> bool:
    """Whether lazy installs are permitted: (1) ``security.allow_lazy_installs: false`` blocks in BOTH
    modes; (2) the sealed venv (``HERMES_DISABLE_LAZY_INSTALLS=1``) blocks only without a durable
    target to redirect into. Unreadable config fails OPEN — blocking is an explicit opt-in."""
    cfg = None
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        cfg = load_config()
    if cfg is not None and not bool((cfg.get("security") or {}).get("allow_lazy_installs", True)):
        return False
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1":
        return _lazy_install_target() is not None
    return True


def _unsupported_feature_reason(feature: str) -> Optional[str]:
    """Platform capability gate (not policy): why a feature cannot work on this host, or None."""
    if sys.platform == "win32" and feature == "platform.matrix":
        return ("unsupported on Windows: Matrix E2EE depends on python-olm, which has no Windows wheel and "
                "requires make + libolm to build from sdist. Run Hermes under WSL to use Matrix on Windows.")
    return None


def _spec_is_safe(spec: str) -> bool:
    """Reject pip specs that contain URLs, paths, or shell metacharacters."""
    return bool(spec and len(spec) <= 200
                and not any(ch in spec for ch in (";", "|", "&", "`", "$", "\n", "\r", "\t", "\\"))
                and not spec.startswith(("-", "/", ".")) and "://" not in spec and "@" not in spec
                and _SAFE_SPEC.match(spec))


def _pkg_name_from_spec(spec: str) -> str:
    """``"mautrix[encryption]>=0.20"`` -> ``"mautrix"``."""
    m = re.match(rf"^({_NAME_RE})", spec)
    return m.group(1) if m else spec


def _specifier_from_spec(spec: str) -> str:
    """``"mautrix[encryption]>=0.20,<1"`` -> ``">=0.20,<1"``; ``""`` if unconstrained."""
    m = _NAME_EXTRAS_RE.match(spec)
    return spec[m.end():] if m else ""


def _installed_version(spec: str) -> Optional[str]:
    """Installed version of the spec's package, or None when absent."""
    try:
        from importlib.metadata import version

        return version(_pkg_name_from_spec(spec))
    except Exception:
        return None


def _is_satisfied(spec: str) -> bool:
    """Present AND inside the spec's version range, so ``hermes update`` propagates pin bumps to
    installed backends. Unparseable specs/versions or a missing ``packaging`` count as satisfied — err
    toward "don't churn"."""
    installed = _installed_version(spec)
    if installed is None:
        return False
    if not (spec_tail := _specifier_from_spec(spec)):
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(installed) in SpecifierSet(spec_tail)
    except Exception:
        return True


def _is_present(spec: str) -> bool:
    """Presence-only check (any version) — how :func:`active_features` detects activated backends."""
    return _installed_version(spec) is not None


def _core_constraints_file() -> Optional[Path]:
    """Temp ``--constraint`` file pinning every core-venv package to its installed version for
    durable-target installs: shared deps resolve as satisfied (store stays minimal) and a conflicting
    backend fails loudly instead of installing a shadowed copy that can never win on sys.path. None if
    enumeration failed (install unconstrained)."""
    try:
        import tempfile
        from importlib.metadata import distributions

        pins: dict[str, str] = {}
        for dist in distributions():
            name = dist.metadata["Name"] if dist.metadata else None
            if name and dist.version and name.lower() not in pins:
                pins[name.lower()] = f"{name}=={dist.version}"
        if not pins:
            return None
        fd, path = tempfile.mkstemp(prefix="hermes-core-constraints-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(pins.values())) + "\n")
        return Path(path)
    except Exception as e:
        logger.debug("Could not build core constraints file: %s", e)
        return None


def _pip_config_candidates(env: dict[str, str]) -> list[Path]:
    """pip's config files, lowest precedence first, as ``pip._internal.configuration`` ranks them:
    global, then user (skipped entirely when ``PIP_CONFIG_FILE`` names an existing file), then the
    venv's ``sys.prefix`` site file, then ``PIP_CONFIG_FILE`` itself on top. ``RawConfigParser.read``
    applies them in order, so the last file wins. ``PIP_CONFIG_FILE=os.devnull`` disables all of them."""
    explicit = env.get("PIP_CONFIG_FILE", "")
    if explicit == os.devnull:
        return []
    home = Path.home()
    if sys.platform == "win32":
        name = "pip.ini"
        global_files = [Path(env.get("ProgramData") or r"C:\ProgramData") / "pip" / name]
        user_files = [home / "pip" / name, Path(env.get("APPDATA") or home / "AppData" / "Roaming") / "pip" / name]
    elif sys.platform == "darwin":
        name = "pip.conf"
        global_files = [Path("/Library/Application Support/pip") / name]
        app_support = home / "Library" / "Application Support" / "pip"
        user_files = [home / ".pip" / name, (app_support if app_support.is_dir() else home / ".config" / "pip") / name]
    else:
        name = "pip.conf"
        xdg_dirs = (env.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(os.pathsep)
        global_files = [Path(d) / "pip" / name for d in xdg_dirs if d] + [Path("/etc") / name]
        user_files = [home / ".pip" / name, Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "pip" / name]
    explicit_files = [Path(explicit)] if explicit else []
    if explicit_files and explicit_files[0].is_file():
        user_files = []
    return global_files + user_files + [Path(sys.prefix) / name] + explicit_files


def _pip_conf_index_url(env: dict[str, str]) -> Optional[str]:
    """Read pip's configured index-url so uv can use the same mirror.

    uv does not read ``pip.conf`` — without this bridge a user whose pip is
    mirrored (common behind restricted networks) watches every lazy install
    hit the default pypi.org and time out (#95608).
    """
    # Raw: pip does not interpolate, and mirror URLs carry percent-encoded credentials.
    parser = configparser.RawConfigParser()
    try:
        parser.read(str(p) for p in _pip_config_candidates(env))
        if not parser.has_section("global"):
            return None
        return parser.get("global", "index-url", fallback="").strip() or None
    except configparser.Error as e:
        logger.debug("Could not read pip.conf for index-url: %s", e)
        return None


def _installed_dist_roots(spec: str, target: Optional[Path]) -> set[Path]:
    """Package dirs a freshly installed *spec* owns, from the dist's file list (``python-telegram-bot``
    ships ``telegram``; some ship several)."""
    name = _pkg_name_from_spec(spec)
    roots: set[Path] = set()
    try:
        import importlib.metadata as _md

        dist = next(iter(_md.distributions(name=name, path=[str(target)])), None) if target is not None else _md.distribution(name)
        for entry in dist.files or () if dist is not None else ():
            top = entry.parts[0] if entry.parts else ""
            # Skip hidden entries, __pycache__ and metadata dirs (no importable code).
            if not top or top.startswith(".") or top == "__pycache__" or top.endswith((".dist-info", ".egg-info")):
                continue
            root = Path(dist.locate_file(top))
            if root.is_dir():
                roots.add(root)
    except Exception:
        return set()
    return roots


def _warm_installed_bytecode(specs: tuple[str, ...], target: Optional[Path]) -> None:
    """Byte-compile what was just installed: a fresh install writes no ``__pycache__``, so the next
    import (often a user request, ~2-10s for a big SDK, reading as a hang) would pay the compile. Pay
    it here while the caller already waits. Best-effort; never fails the install.

    A pip/uv install writes ``.py`` sources and no ``__pycache__`` — and an install of the *same* version
    still deletes the cache the old copy had. Whoever imports the package next pays the whole compile: for
    ``anthropic==0.87.0`` (541 modules) on cpython-3.12.13 that measured 2.2-2.7s cold against 0.7-1.0s
    warm, and 10.5s cold under concurrent load. That bill lands wherever the first import happens, and for a
    lazily-installed backend that is the foreground of a user request (#100461) — with nothing printed while
    it runs, so it reads as a hang. Worse, N per-profile daemons cold-starting together each pay it in full
    before any of them has written the cache.
    """
    if sys.dont_write_bytecode:
        return
    try:
        import compileall
    except Exception:  # pragma: no cover — stdlib, but never break an install
        return
    for spec in specs:
        try:
            for root in _installed_dist_roots(spec, target):
                try:
                    compileall.compile_dir(str(root), quiet=2, force=False, workers=1)
                except Exception as exc:
                    logger.debug("Bytecode warm skipped for %s: %s", root, exc)
        except Exception as exc:
            logger.debug("Bytecode warm skipped for %s: %s", spec, exc)


def _run_installer(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # _SUBPROCESS_KW carries stdin=DEVNULL  # noqa: subprocess-stdin
    return subprocess.run(cmd, **_SUBPROCESS_KW, creationflags=windows_hide_flags(), **kw)


# Whose dependency-security policy an install runs under. ``core``: Hermes's own packages (LAZY_DEPS
# extras, refreshed by ``hermes update``) resolve under the checkout's ``[tool.uv]`` policy — the 14-day
# ``exclude-newer`` quarantine and its per-package exceptions. ``plugin``: a plugin's declared
# ``python_dependencies`` follow the PLUGIN's own policy (maintainer ruling: "plugins don't have to abide
# by our 14 day rule; they can have their own security policy on that. Only Hermes' dependencies themselves
# have to"), so Hermes's project config is not applied — a plugin floored on a release younger than 14 days
# would otherwise be uninstallable through Hermes while installing fine everywhere else.
INSTALL_POLICIES = ("core", "plugin")


def _uv_policy_args(policy: str) -> tuple[list[str], Optional[str]]:
    """``(extra uv args, cwd)`` that pin the resolver to *policy* regardless of the caller's cwd.

    uv reads ``[tool.uv]`` from the project discovered at the *current directory*, so cwd is the seam:
    ``core`` runs from the checkout root (a lazy install launched from ``$HOME``, a gateway service or the
    Desktop backend still gets the quarantine); ``plugin`` passes ``--no-config`` so no project file is
    discovered from any cwd (env knobs such as ``UV_INDEX_URL`` / ``UV_EXCLUDE_NEWER`` still apply, so an
    operator can quarantine plugin deps themselves). Verified with ``uv pip install --show-settings``."""
    if policy not in INSTALL_POLICIES:
        raise ValueError(f"unknown install policy {policy!r}; expected one of {INSTALL_POLICIES}")
    if policy == "plugin":
        return ["--no-config"], None
    root = Path(__file__).resolve().parents[3]
    return [], (str(root) if (root / "pyproject.toml").is_file() else None)


def _uv_binary() -> Optional[str]:
    """Managed uv first ($HERMES_HOME/bin is never on PATH), then PATH. A lookup, not ensure_uv():
    downloading uv mid-turn is more than the caller asked for; pip covers no-uv."""
    try:
        from pm import uv_launcher

        installed = uv_launcher("uv")
        return str(installed) if installed is not None else shutil.which("uv")
    except Exception:
        return shutil.which("uv")


def _write_constraints_file(lines: tuple[str, ...]) -> Path:
    import tempfile
    fd, path = tempfile.mkstemp(prefix="hermes-plugin-constraints-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return Path(path)


def _after_successful_install(specs: tuple[str, ...], target: Optional[Path], dry_run: bool) -> None:
    """Post-install bookkeeping; a dry run installed nothing, so there is nothing to activate/warm."""
    if dry_run:
        return
    if target is not None:
        _activate_target_on_syspath(target)
    _warm_installed_bytecode(specs, target)


def _venv_pip_install(specs: tuple[str, ...], *, timeout: int = 300, constraint_lines: tuple[str, ...] = (),
                      dry_run: bool = False, policy: str = "core") -> _InstallResult:
    """Install ``specs`` via the uv -> pip -> ensurepip ladder, venv-scoped or into the durable
    ``--target`` (constrained to core versions) when :data:`_LAZY_TARGET_ENV` is set. Independent of
    ``hermes_cli.tools_config._pip_install`` (no CLI dependency).

    *constraint_lines* pins the resolver (plugin installs pass Hermes' own declared ranges so a plugin
    can never move a core package out of range); *dry_run* resolves without installing; *policy* is one
    of :data:`INSTALL_POLICIES` (see :func:`_uv_policy_args`)."""
    if not specs:
        return _InstallResult(True, "", "")
    policy_args, uv_cwd = _uv_policy_args(policy)
    target = _lazy_install_target()
    constraints: Optional[Path] = None
    extra_args: list[str] = ["--dry-run"] if dry_run else []
    if target is not None:
        if err := _ensure_target_ready(target):
            return _InstallResult(False, "", err)
        constraints = _core_constraints_file()
        extra_args += ["--target", str(target)]
    elif constraint_lines:
        constraints = _write_constraints_file(constraint_lines)
    if constraints is not None:
        extra_args += ["--constraint", str(constraints)]

    def _finish(r: subprocess.CompletedProcess) -> _InstallResult:
        if r.returncode == 0:
            _after_successful_install(specs, target, dry_run)
        return _InstallResult(r.returncode == 0, r.stdout or "", r.stderr or "")

    try:
        from tools.environments.local import hermes_subprocess_env
        uv_env = hermes_subprocess_env(inherit_credentials=False)
        uv_env["VIRTUAL_ENV"] = str(Path(sys.executable).parent.parent)
        # Tier 1: uv. --compile-bytecode because uv writes no __pycache__ by default, so the first
        # import would recompile the backend AND its transitives (_warm_installed_bytecode is the
        # belt-and-braces pass for the spec's own roots on any tier).
        if uv_bin := _uv_binary():
            # Bridge pip's index unless any uv index knob is set; PIP_INDEX_URL beats pip.conf, as in
            # pip (see _pip_conf_index_url for why uv needs this at all).
            if not any(uv_env.get(k) for k in ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX")):
                pip_index_url = (uv_env.get("PIP_INDEX_URL") or "").strip() or _pip_conf_index_url(uv_env)
                if pip_index_url:
                    uv_env["UV_INDEX_URL"] = pip_index_url
            try:
                r = _run_installer([uv_bin, "pip", "install", "--compile-bytecode", *policy_args, *extra_args, *specs],
                                   timeout=timeout, env=uv_env, cwd=uv_cwd)
                if r.returncode != 0:
                    logger.debug("uv pip install failed: %s", r.stderr)
                # A uv resolver failure is authoritative: falling through to pip would discard uv
                # policy (exclude-newer for core; the constraints file for both) and could install a
                # quarantined or out-of-range release.
                return _finish(r)
            except subprocess.TimeoutExpired as e:
                logger.debug("uv invocation failed: %s", e)
                # Actionable context for the #95608 shape: a 300s stall with
                # no feedback, then silence. The failure string flows into
                # FeatureUnavailable, which callers surface as warnings.
                hint = (
                    f"uv pip install timed out after {timeout}s. If your network needs a package "
                    "mirror, set index-url in pip.conf (bridged to uv automatically) or UV_INDEX_URL."
                )
                return _InstallResult(False, "", hint)
            except FileNotFoundError as e:  # uv vanished between lookup and spawn; it never evaluated the requirements
                logger.debug("uv invocation failed: %s", e)
        # Tier 2: python -m pip (ensurepip bootstrap if needed)
        pip_cmd = [sys.executable, "-m", "pip"]
        try:
            if _run_installer(pip_cmd + ["--version"], timeout=15).returncode != 0:
                raise FileNotFoundError("pip not in venv")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                _run_installer([sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"], timeout=120, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                return _InstallResult(False, "", f"pip not available and ensurepip failed: {e}")
        try:
            return _finish(_run_installer(pip_cmd + ["install", *extra_args, *specs], timeout=timeout))
        except subprocess.TimeoutExpired as e:
            return _InstallResult(False, "", f"pip install timed out: {e}")
        except Exception as e:
            return _InstallResult(False, "", f"pip install failed: {e}")
    finally:
        if constraints is not None:
            with contextlib.suppress(OSError):
                constraints.unlink()


def feature_missing(feature: str) -> tuple[str, ...]:
    """Return the subset of specs for ``feature`` not currently installed."""
    if feature not in LAZY_DEPS:
        raise KeyError(f"Unknown lazy feature: {feature!r}")
    return tuple(s for s in LAZY_DEPS[feature] if not _is_satisfied(s))


def _prompt_toolkit_active() -> bool:
    """A bare input() deadlocks while a prompt_toolkit app owns the terminal, so ensure() skips the
    confirmation under the TUI — reaching it is already gated by security.allow_lazy_installs."""
    if "prompt_toolkit.application.current" not in sys.modules:
        return False
    try:
        from prompt_toolkit.application.current import get_app_or_none
        return bool(getattr(get_app_or_none(), "is_running", False))
    except Exception:
        return False


def ensure(feature: str, *, prompt: bool = True) -> None:
    """Make every package for ``feature`` importable, installing if needed; raises
    :class:`FeatureUnavailable` when installs are disabled or fail. ``prompt``: confirm on a TTY first
    (non-interactive callers pass False and rely on the config gate)."""
    if feature not in LAZY_DEPS:
        raise FeatureUnavailable(feature, (), f"feature {feature!r} not in LAZY_DEPS allowlist")
    missing = feature_missing(feature)
    if not missing:
        return
    if unsupported := _unsupported_feature_reason(feature):
        raise FeatureUnavailable(feature, missing, unsupported)
    # Package-manager installs (NixOS etc.) have read-only site-packages: fail fast instead of burning
    # ~15s on ensurepip — unless a durable target is configured. The reason MUST start with
    # "unsupported ": _refresh_features classifies skips by that prefix.
    if _lazy_install_target() is None:
        managed_by = ""  # config unreadable — proceed with the install
        with contextlib.suppress(Exception):
            from hermes_cli.config import get_managed_system
            managed_by = get_managed_system()
        if managed_by:
            raise FeatureUnavailable(
                feature, missing,
                f"unsupported on {managed_by}-managed installs: this build's packages come from {managed_by}, "
                f"so Hermes cannot install them at runtime. Add the dependencies for {feature!r} via "
                f"{managed_by} (or run a pip/uv install of Hermes instead).")
    for spec in missing:  # belt and braces on top of the allowlist
        if not _spec_is_safe(spec):
            raise FeatureUnavailable(feature, missing, f"refusing to install unsafe spec {spec!r}")
    if not _allow_lazy_installs():
        raise FeatureUnavailable(feature, missing, "lazy installs disabled (security.allow_lazy_installs=false)")
    if prompt and not _prompt_toolkit_active() and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            answer = input(f"\nFeature {feature!r} requires: {', '.join(missing)}\nInstall into the active venv now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in {"y", "yes"}:
            raise FeatureUnavailable(feature, missing, "user declined install at prompt")
    logger.info("Lazy-installing %s for feature %r", " ".join(missing), feature)
    result = _venv_pip_install(missing)
    if not result.success:  # surface pip's own error (quarantine 404, network), tail-clipped
        snippet = (result.stderr or result.stdout or "").strip()[-2000:]
        raise FeatureUnavailable(feature, missing, f"pip install failed: {snippet or 'no error output'}")
    _invalidate_import_caches()
    if still_missing := feature_missing(feature):
        raise FeatureUnavailable(feature, still_missing, "install reported success but packages still not importable (may require Python restart)")
    logger.info("Lazy install complete for feature %r", feature)


def is_available(feature: str) -> bool:
    """Return True if the feature's deps are already satisfied."""
    return feature in LAZY_DEPS and not feature_missing(feature)


def feature_install_command(feature: str, *, venv_pip: bool = False) -> Optional[str]:
    """Manual install command for a feature, or None. ``venv_pip=True`` uses ``{sys.executable} -m pip``
    — immune to PEP 668 failures a bare ``pip install`` invites."""
    if feature not in LAZY_DEPS:
        return None
    joined = " ".join(repr(s) for s in LAZY_DEPS[feature])
    return f"{sys.executable} -m pip install {joined}" if venv_pip else "uv pip install " + joined


@dataclass
class InstallSpecsResult:
    """Outcome of :func:`install_specs` for one batch of pip specs. ``blocked`` means installs are gated
    off (config kill switch, sealed venv without a durable target) or a spec failed validation — nothing
    was executed, ``reason`` says why. ``command`` is the human-readable description of what ran."""
    ok: bool
    blocked: bool = False
    reason: str = ""
    command: str = ""
    stdout: str = ""
    stderr: str = ""


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300,
                  constraints: list[str] | tuple[str, ...] = (), dry_run: bool = False,
                  policy: str = "plugin") -> InstallSpecsResult:
    """Install data-driven pip specs (plugin manifest ``python_dependencies``) with the same routing and
    gating as :func:`ensure`, but unknown packages are allowed — the caller owns manifest trust, this
    owns spec hygiene. *constraints* are requirement lines the resolver must honour; *dry_run* only
    resolves; *policy* defaults to ``"plugin"`` (the plugin's own dependency policy, not Hermes's
    ``exclude-newer`` quarantine — :data:`INSTALL_POLICIES`). Never raises; inspect the
    :class:`InstallSpecsResult`."""
    cleaned = tuple(str(s).strip() for s in specs if str(s).strip())
    if not cleaned:
        return InstallSpecsResult(ok=True, command="")
    if policy not in INSTALL_POLICIES:
        return InstallSpecsResult(ok=False, blocked=True, reason=f"unknown install policy {policy!r}")
    for spec in cleaned:
        if _pkg_name_from_spec(spec).lower() not in {"hindsight-client", "hindsight-all"}:
            return InstallSpecsResult(ok=False, blocked=True, reason="only retained Hindsight dependencies are supported")
        if not _spec_is_safe(spec):
            return InstallSpecsResult(ok=False, blocked=True, reason=f"refusing to install unsafe spec {spec!r}")
    target = _lazy_install_target()
    if not _allow_lazy_installs():
        sealed = os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1" and target is None
        reason = ("runtime installs are disabled on this deployment: the agent environment is immutable "
                  "and no writable install target is configured (HERMES_LAZY_INSTALL_TARGET)"
                  ) if sealed else "runtime installs disabled (security.allow_lazy_installs=false)"
        return InstallSpecsResult(ok=False, blocked=True, reason=reason)
    display = "uv pip install " + (f"--target {target} " if target is not None else "") + " ".join(cleaned)
    logger.info("%s pip specs %s (target=%s)", "Resolving" if dry_run else "Installing", " ".join(cleaned), target or "venv")
    try:
        result = _venv_pip_install(cleaned, timeout=timeout, constraint_lines=tuple(constraints), dry_run=dry_run,
                                   policy=policy)
    except Exception as exc:
        logger.warning("install_specs failed unexpectedly: %s", exc)
        return InstallSpecsResult(ok=False, command=display, stderr=f"install failed: {exc}")

    _invalidate_import_caches()  # dashboard rechecks availability inline
    return InstallSpecsResult(ok=result.success, command=display, stdout=result.stdout, stderr=result.stderr)
