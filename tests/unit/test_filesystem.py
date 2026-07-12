from types import SimpleNamespace
import stat

from temper_ml.filesystem import is_link_or_reparse


def test_windows_reparse_attribute_is_treated_as_a_link() -> None:
    info = SimpleNamespace(
        st_mode=stat.S_IFREG,
        st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
    )

    assert is_link_or_reparse(info) is True  # type: ignore[arg-type]
