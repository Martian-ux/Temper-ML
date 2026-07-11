"""Canonical JSON bytes for Temper-owned identities."""

from __future__ import annotations

from decimal import Decimal
import json
import math
from typing import Any, Iterable


class CanonicalJsonError(ValueError):
    """Raised when a value cannot be represented as Temper canonical JSON."""


def dumps_canonical_json(value: Any) -> bytes:
    """Encode a JSON value as UTF-8 canonical bytes with one trailing newline."""

    try:
        return (_encode(value) + "\n").encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalJsonError("strings must be valid UTF-8") from exc


def loads_canonical_json(data: bytes | str) -> Any:
    """Read exactly one Temper canonical JSON value."""

    try:
        encoded = data if isinstance(data, bytes) else data.encode("utf-8")
        text = encoded.decode("utf-8")
    except UnicodeError as exc:
        raise CanonicalJsonError("input is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=Decimal,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise CanonicalJsonError(str(exc)) from exc
    _validate(value)
    if encoded != dumps_canonical_json(value):
        raise CanonicalJsonError("JSON is not in Temper canonical form")
    return value


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJsonError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CanonicalJsonError(f"non-finite number is not allowed: {value}")


def _validate(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, Decimal):
        _format_decimal(value)
        return
    if isinstance(value, float):
        raise CanonicalJsonError("float values are not allowed; use normalized decimals")
    if isinstance(value, list):
        for item in value:
            _validate(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError("object keys must be strings")
            _validate(item)
        return
    raise CanonicalJsonError(f"unsupported JSON value type: {type(value).__name__}")


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Decimal):
        return _format_decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError("non-finite float values are not allowed")
        raise CanonicalJsonError("float values are not allowed; use normalized decimals")
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise CanonicalJsonError("object keys must be strings")
        return "{" + ",".join(
            f"{_encode(key)}:{_encode(value[key])}" for key in sorted(value)
        ) + "}"
    raise CanonicalJsonError(f"unsupported JSON value type: {type(value).__name__}")


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise CanonicalJsonError("non-finite decimal values are not allowed")
    sign, digits, exponent = value.as_tuple()
    if exponent >= 0 or not digits or digits[-1] == 0:
        raise CanonicalJsonError("decimal values must be normalized decimal fractions")

    digit_text = "".join(str(digit) for digit in digits)
    places = -exponent
    if len(digit_text) <= places:
        integer = "0"
        fraction = "0" * (places - len(digit_text)) + digit_text
    else:
        integer = digit_text[:-places]
        fraction = digit_text[-places:]
    prefix = "-" if sign else ""
    return f"{prefix}{integer}.{fraction}"
