"""Identity differences between repositories sharing the Gems source contract."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GemsSourceProfile:
    repository: str
    package: str
    framework: str

    @property
    def remote(self) -> str:
        return f"https://github.com/{self.repository}.git"

    def package_root(self, checkout: str | Path) -> Path:
        return Path(checkout) / "src" / self.package


PROFILES = (
    GemsSourceProfile("flagos-ai/FlagGems", "flag_gems", "flaggems"),
    GemsSourceProfile("flagos-ai/FlagGems-vllm", "flaggems_vllm", "flaggems_vllm"),
)


def repository_profile(repository: str) -> GemsSourceProfile:
    for profile in PROFILES:
        if profile.repository.lower() == repository.lower():
            return profile
    raise ValueError(f"unsupported Gems repository: {repository}")


def checkout_profile(checkout: str | Path) -> GemsSourceProfile:
    matches = [p for p in PROFILES if p.package_root(checkout).is_dir()]
    if len(matches) != 1:
        raise ValueError("expected exactly one supported Gems source package in checkout")
    return matches[0]
