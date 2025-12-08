"""Tests for zarr v2/v3 compatibility utilities."""

import pytest
import zarr

from funtracks.utils import (
    detect_zarr_spec_version,
    get_store_path,
    is_zarr_v3,
    open_zarr_store,
    remove_tilde,
    setup_zarr_array,
    setup_zarr_group,
)


class TestIsZarrV3:
    def test_returns_bool(self):
        result = is_zarr_v3()
        assert isinstance(result, bool)

    def test_matches_version_string(self):
        expected = zarr.__version__.startswith("3")
        assert is_zarr_v3() == expected


class TestRemoveTilde:
    def test_expands_tilde(self):
        result = remove_tilde("~/test/path")
        assert "~" not in str(result)

    def test_no_tilde_unchanged(self):
        path = "/absolute/path/to/file"
        result = remove_tilde(path)
        assert str(result) == path


class TestDetectZarrSpecVersion:
    def test_detect_v2_from_zgroup(self, tmp_path):
        # Create a v2-style zarr with .zgroup
        zarr_path = tmp_path / "test.zarr"
        if is_zarr_v3():
            zarr.open_group(zarr_path, mode="w", zarr_format=2)
        else:
            zarr.open_group(zarr_path, mode="w")

        result = detect_zarr_spec_version(zarr_path)
        assert result == 2

    @pytest.mark.skipif(not is_zarr_v3(), reason="Requires zarr-python v3")
    def test_detect_v3_from_zarr_json(self, tmp_path):
        # Create a v3-style zarr with zarr.json
        zarr_path = tmp_path / "test.zarr"
        zarr.open_group(zarr_path, mode="w", zarr_format=3)

        result = detect_zarr_spec_version(zarr_path)
        assert result == 3

    def test_nonexistent_path_returns_none(self, tmp_path):
        result = detect_zarr_spec_version(tmp_path / "nonexistent")
        assert result is None


class TestSetupZarrGroup:
    def test_creates_group(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        group = setup_zarr_group(zarr_path, zarr_format=2, mode="w")

        assert isinstance(group, zarr.Group)
        assert zarr_path.exists()

    def test_default_format_is_v2(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        setup_zarr_group(zarr_path, mode="w")

        # Should be v2 format
        assert detect_zarr_spec_version(zarr_path) == 2

    @pytest.mark.skipif(not is_zarr_v3(), reason="Requires zarr-python v3")
    def test_creates_v3_format(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        setup_zarr_group(zarr_path, zarr_format=3, mode="w")

        assert detect_zarr_spec_version(zarr_path) == 3

    @pytest.mark.skipif(is_zarr_v3(), reason="Only for zarr-python v2")
    def test_v3_format_warns_on_zarr_v2(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        with pytest.warns(UserWarning, match="zarr-python v2 does not support spec v3"):
            setup_zarr_group(zarr_path, zarr_format=3, mode="w")


class TestSetupZarrArray:
    def test_creates_array(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        arr = setup_zarr_array(
            zarr_path,
            zarr_format=2,
            shape=(10, 10),
            dtype="int32",
        )

        assert isinstance(arr, zarr.Array)
        assert arr.shape == (10, 10)
        assert zarr_path.exists()

    def test_with_chunks(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        arr = setup_zarr_array(
            zarr_path,
            zarr_format=2,
            shape=(100, 100),
            dtype="float64",
            chunks=(10, 10),
        )

        assert arr.chunks == (10, 10)

    @pytest.mark.skipif(is_zarr_v3(), reason="Only for zarr-python v2")
    def test_v3_format_warns_on_zarr_v2(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        with pytest.warns(UserWarning, match="zarr-python v2 does not support spec v3"):
            setup_zarr_array(zarr_path, zarr_format=3, shape=(10,), dtype="int32")


class TestOpenZarrStore:
    def test_opens_existing_store(self, tmp_path):
        # Create a zarr first
        zarr_path = tmp_path / "test.zarr"
        setup_zarr_group(zarr_path, mode="w")

        store = open_zarr_store(zarr_path)
        assert store is not None

    def test_raises_on_nonexistent_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_zarr_store(tmp_path / "nonexistent")

    def test_returns_correct_store_type(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        setup_zarr_group(zarr_path, mode="w")

        store = open_zarr_store(zarr_path)

        if is_zarr_v3():
            assert isinstance(store, zarr.storage.LocalStore)
        else:
            assert isinstance(store, zarr.storage.FSStore)

    @pytest.mark.skipif(is_zarr_v3(), reason="Only for zarr-python v2")
    def test_warns_opening_v3_with_zarr_v2(self, tmp_path):
        # Create a fake v3 zarr by adding zarr.json
        zarr_path = tmp_path / "test.zarr"
        zarr_path.mkdir()
        (zarr_path / "zarr.json").write_text("{}")

        with pytest.warns(UserWarning, match="zarr spec v3 file with zarr-python v2"):
            open_zarr_store(zarr_path)


class TestGetStorePath:
    def test_gets_path_from_store(self, tmp_path):
        zarr_path = tmp_path / "test.zarr"
        setup_zarr_group(zarr_path, mode="w")

        store = open_zarr_store(zarr_path)
        result = get_store_path(store)

        assert result == zarr_path

    def test_raises_on_unknown_store_type(self):
        class FakeStore:
            pass

        with pytest.raises(ValueError, match="Cannot determine store path"):
            get_store_path(FakeStore())
