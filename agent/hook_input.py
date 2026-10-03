"""Source-bound, process-local prompt provenance for lifecycle hook consumers.

Only a builder can carry this string subtype. JSON/database round trips deliberately
lose provenance; matching marker text or a cache prefix never reconstructs it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class _EnvelopeParts:
    request: str
    prefix: str
    labels: tuple[str, ...]
    home: str


class HostInputEnvelope(str):
    """Serving text stays byte-identical; the hook view names excluded host context."""

    _parts: _EnvelopeParts

    def __new__(cls, prefix: str, request: str, labels: tuple[str, ...], home: str) -> HostInputEnvelope:
        value = super().__new__(cls, prefix + request)
        object.__setattr__(value, '_parts', _EnvelopeParts(request, prefix, labels, home))
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('Host input provenance is immutable')

    def __reduce__(self) -> tuple:
        parts = self._parts
        return type(self), (parts.prefix, parts.request, parts.labels, parts.home)


def skill_input_envelope(assembled: str, prefix: str, request: str, labels: list[str]) -> str:
    """Bind only an exact, scanner-approved builder result; otherwise keep plain text."""
    if not prefix or not request.strip() or assembled != prefix + request:
        return assembled
    return HostInputEnvelope(prefix, request, tuple(labels), str(get_hermes_home().resolve()))


def preserve_input_envelope(original: Any, prepared: Any) -> Any:
    """Keep pre-existing provenance only across a byte-unchanged preparation step.

    A plain string cannot acquire provenance here. Sanitization that changes the
    serving text invalidates the source binding and returns its ordinary result.
    """
    if (isinstance(prepared, str) and original == prepared
            and input_envelope_snapshot(original) is not None):
        return original
    return prepared


def input_envelope_snapshot(value: Any) -> dict | None:
    """Plain data for hooks; no mutable agent, credentials, registry or authority by text."""
    if not isinstance(value, HostInputEnvelope):
        return None
    parts = value._parts
    if parts.home != str(get_hermes_home().resolve()) or value != parts.prefix + parts.request:
        return None
    return {
        'version': 1,
        'source_sha256': hashlib.sha256(value.encode('utf-8')).hexdigest(),
        'request': parts.request,
        'excluded_context': {
            'kind': 'skill_instructions',
            'labels': list(parts.labels),
            'chars': len(parts.prefix),
            'utf8_bytes': len(parts.prefix.encode('utf-8')),
        },
    }
