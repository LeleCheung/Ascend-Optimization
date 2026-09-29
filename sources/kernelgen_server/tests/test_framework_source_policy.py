"""Deployment source defaults are independent of KGS release and Protocol."""
from pathlib import Path

import yaml


def test_gems_default_tracks_the_official_development_branch():
    manifest = yaml.safe_load((Path(__file__).parents[1] / "compatibility.yaml").read_text())
    assert manifest["frameworks"]["flaggems"] == {
        "repository": "https://github.com/flagos-ai/FlagGems.git",
        "branch": "kernelgen-dev",
        "revision_policy": "branch",
    }
    assert manifest["server_release"]["api_version"] == "v6.2"
