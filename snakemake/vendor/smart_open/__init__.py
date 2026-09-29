"""Local-file subset of smart_open needed by the worker Snakemake bootstrap."""

from __future__ import annotations

import builtins
import os
from typing import NamedTuple
from urllib.parse import unquote, urlparse


class Uri(NamedTuple):
    scheme: str
    uri_path: str


def parse_uri(path_or_uri) -> Uri:
    value = os.fspath(path_or_uri)
    parsed = urlparse(value)
    if parsed.scheme in ("", "file"):
        path = unquote(parsed.path) if parsed.scheme == "file" else value
        return Uri("file", path)
    return Uri(parsed.scheme, parsed.path)


def open(
    path_or_uri,
    mode="r",
    encoding=None,
    compression="infer_from_extension",
    transport_params=None,
    **kwargs,
):
    """Open a local file while accepting ``smart_open.open`` options.

    Snakemake 9 passes ``compression="disable"`` while populating its source
    cache.  These two options belong to smart_open itself and must not be
    forwarded to Python's built-in ``open``, which does not accept them.
    """
    uri = parse_uri(path_or_uri)
    if uri.scheme != "file":
        raise NotImplementedError(
            f"Worker bootstrap supports local files only, not {uri.scheme!r}"
        )
    if compression not in (None, "disable", "infer_from_extension"):
        raise NotImplementedError(
            f"Worker bootstrap does not support compression {compression!r}"
        )
    if transport_params:
        raise NotImplementedError(
            "Worker bootstrap does not support smart_open transport parameters"
        )
    return builtins.open(uri.uri_path, mode, encoding=encoding, **kwargs)
