import numpy as np
import pytest
import zarr

from funtracks.data_model.solution_tracks import SolutionTracks
from funtracks.data_model.tracks import Tracks
from funtracks.import_export.export_to_geff import export_to_geff, split_position_attr


@pytest.mark.parametrize(
    "ndim,graph_nd,segmentation_nd",
    [(2, 2, 2), (3, 3, 3)],
    indirect=["graph_nd", "segmentation_nd"],
)
@pytest.mark.parametrize("track_type", (Tracks, SolutionTracks))
@pytest.mark.parametrize("pos_attr_type", (str, list))
class TestExportToGeff:
    @pytest.fixture(autouse=True)
    def setup(self, ndim, track_type, pos_attr_type, graph_nd, segmentation_nd, tmp_path):
        self.tracks = track_type(graph_nd, segmentation=segmentation_nd, ndim=ndim + 1)
        if pos_attr_type is list:
            self.tracks.graph = split_position_attr(self.tracks)
            self.tracks.pos_attr = ["y", "x"] if ndim == 2 else ["z", "y", "x"]

        # Create unique subdirectories for each test
        self.test_dir = tmp_path / "test_export"
        self.test_dir.mkdir()

    def test_basic_export(self):
        export_dir = self.test_dir / "basic"
        export_dir.mkdir()
        export_to_geff(self.tracks, export_dir)
        z = zarr.open((export_dir / "tracks").as_posix(), mode="r")
        assert isinstance(z, zarr.Group)

        # Check that segmentation was saved
        seg_path = export_dir / "segmentation"
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        np.testing.assert_array_equal(seg_zarr[:], self.tracks.segmentation)

        # Check that affine is present in metadata
        attrs = dict(z.attrs)
        assert "geff" in attrs
        assert "affine" in attrs["geff"]
        affine = attrs["geff"]["affine"]
        assert affine is None or isinstance(affine, dict)

    def test_nonexisting_dir(self):
        file_path = self.test_dir / "nonexisting" / "target.zarr"
        with pytest.raises(ValueError, match="does not exist"):
            export_to_geff(self.tracks, file_path)

    def test_not_a_directory(self):
        file_path = self.test_dir / "not_a_dir"
        file_path.write_text("test")
        with pytest.raises(ValueError, match="not a directory"):
            export_to_geff(self.tracks, file_path)

    def test_non_empty_dir(self):
        export_dir = self.test_dir / "non_empty"
        export_dir.mkdir()
        (export_dir / "existing_file.txt").write_text("already here")
        with pytest.raises(ValueError, match="not empty"):
            export_to_geff(self.tracks, export_dir)

    def test_overwrite(self):
        export_dir = self.test_dir / "overwrite"
        export_dir.mkdir()
        (export_dir / "existing_file.txt").write_text("already here")

        export_to_geff(self.tracks, export_dir, overwrite=True)
        z = zarr.open((export_dir / "tracks").as_posix(), mode="r")
        assert isinstance(z, zarr.Group)

        seg_path = export_dir / "segmentation"
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        np.testing.assert_array_equal(seg_zarr[:], self.tracks.segmentation)
