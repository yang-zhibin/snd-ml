"""Minimal Unix AppDirs interface required by the bundled Snakemake 7 CLI."""

from __future__ import annotations

import os
from pathlib import Path


class AppDirs:
    """Provide the three directory properties used by this Snakemake version."""

    def __init__(self, appname: str, appauthor: str | None = None):
        self.appname = appname
        self.appauthor = appauthor

    @property
    def user_config_dir(self) -> str:
        root = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return str(Path(root) / self.appname)

    @property
    def site_config_dir(self) -> str:
        roots = os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(os.pathsep)
        return str(Path(roots[0]) / self.appname)

    @property
    def user_cache_dir(self) -> str:
        root = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return str(Path(root) / self.appname)
