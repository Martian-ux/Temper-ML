from decimal import Decimal

import pytest

from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)


def test_canonical_json_sorts_keys_and_uses_one_trailing_newline():
    value = {"z": [3, True, None], "a": {"b": "alpha"}}

    encoded = dumps_canonical_json(value)

    assert encoded == b'{"a":{"b":"alpha"},"z":[3,true,null]}\n'


def test_canonical_json_rejects_duplicate_keys_on_read():
    with pytest.raises(CanonicalJsonError, match="duplicate key"):
        loads_canonical_json(b'{"a":1,"a":2}\n')


def test_canonical_json_rejects_noncanonical_input_on_read():
    with pytest.raises(CanonicalJsonError, match="canonical form"):
        loads_canonical_json(b'{ "a" : 1 }\n')


def test_canonical_json_rejects_invalid_utf8_on_read():
    with pytest.raises(CanonicalJsonError, match="UTF-8"):
        loads_canonical_json(b"\xff")


def test_canonical_json_rejects_floats_and_accepts_normalized_decimals():
    assert dumps_canonical_json({"score": Decimal("12.34")}) == b'{"score":12.34}\n'

    with pytest.raises(CanonicalJsonError, match="float"):
        dumps_canonical_json({"score": 12.34})

    with pytest.raises(CanonicalJsonError, match="normalized decimal"):
        dumps_canonical_json({"score": Decimal("12.340")})
