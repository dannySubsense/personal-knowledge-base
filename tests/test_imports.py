"""Tests for package imports."""


class TestImports:
    """Test that src package can be imported."""

    def test_import_src(self) -> None:
        """Verify src package can be imported."""
        import src

        assert src is not None

    def test_src_is_package(self) -> None:
        """Verify src is a package with __path__."""
        import src

        assert hasattr(src, "__path__")
        assert hasattr(src, "__name__")
        assert src.__name__ == "src"
