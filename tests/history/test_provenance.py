import json
import subprocess
from importlib import metadata

from funtracks.history import package_provenance


class FakeDistribution:
    def __init__(self, direct_url):
        self.version = "1.2.3"
        self._direct_url = direct_url

    def read_text(self, name):
        return None if self._direct_url is None else json.dumps(self._direct_url)


def fake_distribution(monkeypatch, direct_url):
    monkeypatch.setattr(
        metadata, "distribution", lambda name: FakeDistribution(direct_url)
    )


def test_editable_install_reports_git_commit(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "--allow-empty", "-m", "x"],
        check=True,
    )  # fmt: skip
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    url = {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    fake_distribution(monkeypatch, url)
    info = package_provenance("pkg")
    assert info == {
        "version": "1.2.3",
        "commit": commit,
        "dirty": False,
        "editable": True,
        "source": "git",
    }

    (tmp_path / "new_file").write_text("x")
    assert package_provenance("pkg")["dirty"] is True


def test_vcs_install_reports_commit_id(monkeypatch):
    fake_distribution(monkeypatch, {"url": "https://x", "vcs_info": {"commit_id": "abc"}})
    info = package_provenance("pkg")
    assert (info["source"], info["commit"], info["editable"]) == ("vcs", "abc", False)


def test_wheel_inside_a_git_repo_is_still_a_wheel(monkeypatch):
    # No direct_url.json: where the package sits on disk does not matter
    fake_distribution(monkeypatch, None)
    info = package_provenance("pkg")
    assert (info["source"], info["commit"]) == ("wheel", None)


def test_missing_package():
    assert package_provenance("surely-not-installed-xyz")["version"] is None
