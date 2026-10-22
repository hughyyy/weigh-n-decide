"""Normalize image inputs (file path, data URI, http(s) URL) to (media_type, bytes)."""

from __future__ import annotations

import base64
import binascii
import mimetypes
import re
import urllib.request

DEFAULT_MEDIA_TYPE = "image/png"
# Hard cap on decoded image bytes; providers reject larger payloads anyway.
MAX_IMAGE_BYTES = 25 * 1024 * 1024

_DATA_URI_RE = re.compile(
    r"^data:(?P<media>[^;,]*)(?P<params>[^,]*),(?P<data>.*)$", re.DOTALL
)


class ImageInputError(ValueError):
    """Raised when an image input cannot be resolved."""


class _HttpOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that refuses non-http(s) targets (file:, ftp:)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        if not str(newurl).startswith(("http://", "https://")):
            raise ImageInputError(
                f"refusing non-http redirect target: {newurl}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpOnlyRedirectHandler())


def load_image(image: str) -> tuple[str, bytes]:
    """Resolve `image` to (media_type, bytes). Path, data URI, or http(s) URL."""
    if not isinstance(image, str) or not image.strip():
        raise ImageInputError(
            "image input must be a non-empty string (path, data URI, or http(s) URL)"
        )
    if image.startswith("data:"):
        return _from_data_uri(image)
    if image.startswith(("http://", "https://")):
        return _from_url(image)
    return _from_path(image)


def _check_bytes(data: bytes, what: str) -> bytes:
    if not data:
        raise ImageInputError(f"{what} is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(
            f"{what} exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit"
        )
    return data


def _from_data_uri(uri: str) -> tuple[str, bytes]:
    match = _DATA_URI_RE.match(uri)
    if not match:
        raise ImageInputError("malformed data URI")
    media = match.group("media") or DEFAULT_MEDIA_TYPE
    try:
        data = base64.b64decode(match.group("data"))
    except (ValueError, binascii.Error) as exc:
        raise ImageInputError(f"malformed base64 in data URI: {exc}") from exc
    return media, _check_bytes(data, "data URI image")


def _from_path(path: str) -> tuple[str, bytes]:
    media = mimetypes.guess_type(path)[0] or DEFAULT_MEDIA_TYPE
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise ImageInputError(f"cannot read image file {path}: {exc}") from exc
    return media, _check_bytes(data, f"image file {path}")


def _from_url(url: str, timeout: float = 30.0) -> tuple[str, bytes]:
    try:
        with _OPENER.open(url, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            data = resp.read()
    except urllib.error.URLError as exc:
        raise ImageInputError(f"cannot fetch image URL: {exc}") from exc
    media = content_type if content_type.startswith("image/") else DEFAULT_MEDIA_TYPE
    return media, _check_bytes(data, f"image URL {url}")