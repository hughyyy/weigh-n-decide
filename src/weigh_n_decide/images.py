"""Normalize image inputs (file path, data URI, http(s) URL) to (media_type, bytes).

URL fetching is bounded and, by default, restricted to globally routable
addresses. Image inputs are agent-supplied, so an unrestricted fetch would
turn the server into an SSRF and local-file-read primitive.
"""

from __future__ import annotations

import base64
import binascii
import http.client
import ipaddress
import mimetypes
import os
import re
import socket
import stat
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_MEDIA_TYPE = "image/png"
# Hard cap on decoded image bytes; providers reject larger payloads anyway.
MAX_IMAGE_BYTES = 25 * 1024 * 1024
# Aggregate cap across all images in one call, checked by the engine.
MAX_TOTAL_IMAGE_BYTES = 100 * 1024 * 1024
URL_POLICIES = ("public", "any")

_DATA_URI_RE = re.compile(
    r"^data:(?P<media>[^;,]*)(?P<params>[^,]*),(?P<data>.*)$",
    re.DOTALL | re.IGNORECASE,
)


class ImageInputError(ValueError):
    """Raised when an image input cannot be resolved."""


def _check_url_policy(policy: str) -> str:
    if policy not in URL_POLICIES:
        raise ImageInputError(
            f"unknown image URL policy {policy!r} (policies: {', '.join(URL_POLICIES)})"
        )
    return policy


def _ensure_url_allowed(url: str, policy: str) -> None:
    """Reject non-public URL targets when the policy is `public`.

    This covers loopback, link-local (cloud metadata), private ranges, and
    anything else not globally routable. DNS is resolved here and again by the
    HTTP stack; a determined rebinding attack is out of scope for a local
    MCP server, but the obvious SSRF surface is closed.
    """
    if policy == "any":
        return
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ImageInputError(f"image URL must be an http(s) URL: {url}")
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise ImageInputError(f"invalid image URL port: {url}") from exc
    try:
        infos = socket.getaddrinfo(
            parts.hostname, port, type=socket.SOCK_STREAM
        )
    except OSError as exc:
        raise ImageInputError(
            f"cannot resolve image URL host {parts.hostname!r}: {exc}"
        ) from exc
    for info in infos:
        address = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:  # pragma: no cover - getaddrinfo yields IP literals
            continue
        if not ip.is_global:
            raise ImageInputError(
                f"image URL host {parts.hostname!r} resolves to non-public "
                f"address {ip} (set WEIGH_N_DECIDE_IMAGE_URL_POLICY=any to allow)"
            )


class _HttpOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that refuses non-http(s) targets and non-public hosts."""

    def __init__(self, policy: str):
        super().__init__()
        self.policy = policy

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        if not str(newurl).lower().startswith(("http://", "https://")):
            raise ImageInputError(
                f"refusing non-http redirect target: {newurl}"
            )
        _ensure_url_allowed(str(newurl), self.policy)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENERS = {
    policy: urllib.request.build_opener(_HttpOnlyRedirectHandler(policy))
    for policy in URL_POLICIES
}


def load_image(
    image: str, *, url_policy: str = "any", timeout: float = 30.0
) -> tuple[str, bytes]:
    """Resolve `image` to (media_type, bytes). Path, data URI, or http(s) URL.

    `url_policy` is "public" (default secure engine setting) or "any"; the
    library default is "any" so local callers keep the original behavior.
    """
    if not isinstance(image, str) or not image.strip():
        raise ImageInputError(
            "image input must be a non-empty string (path, data URI, or http(s) URL)"
        )
    scheme, sep, _ = image.partition(":")
    scheme = scheme.lower() if sep else ""
    if scheme == "data":
        return _from_data_uri(image)
    if scheme in ("http", "https"):
        policy = _check_url_policy(url_policy)
        return _from_url(image, policy=policy, timeout=timeout)
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
        data = base64.b64decode(match.group("data"), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ImageInputError(f"malformed base64 in data URI: {exc}") from exc
    return media, _check_bytes(data, "data URI image")


def _from_path(path: str) -> tuple[str, bytes]:
    path = os.path.expanduser(path)
    media = mimetypes.guess_type(path)[0] or DEFAULT_MEDIA_TYPE
    try:
        info = os.stat(path)
    except OSError as exc:
        raise ImageInputError(f"cannot read image file {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ImageInputError(f"image path {path} is not a regular file")
    if info.st_size > MAX_IMAGE_BYTES:
        raise ImageInputError(
            f"image file {path} exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit"
        )
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_IMAGE_BYTES + 1)
    except OSError as exc:
        raise ImageInputError(f"cannot read image file {path}: {exc}") from exc
    return media, _check_bytes(data, f"image file {path}")


def _from_url(url: str, *, policy: str, timeout: float) -> tuple[str, bytes]:
    _ensure_url_allowed(url, policy)
    try:
        with _OPENERS[policy].open(url, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            declared = resp.headers.get("Content-Length")
            declared_size: int | None = None
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > MAX_IMAGE_BYTES:
                    raise ImageInputError(
                        f"image URL {url} exceeds the "
                        f"{MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit"
                    )
            data = resp.read(MAX_IMAGE_BYTES + 1)
            if declared_size is not None and len(data) < declared_size:
                raise ImageInputError(
                    f"cannot fetch image URL: truncated response "
                    f"({len(data)} of {declared_size} bytes)"
                )
    except ImageInputError:
        raise
    except urllib.error.HTTPError as exc:
        exc.close()
        raise ImageInputError(f"cannot fetch image URL: {exc}") from exc
    except (OSError, http.client.HTTPException) as exc:
        # URLError is an OSError; HTTPException covers truncated bodies
        # (IncompleteRead) and malformed responses.
        raise ImageInputError(f"cannot fetch image URL: {exc}") from exc
    media = content_type if content_type.startswith("image/") else DEFAULT_MEDIA_TYPE
    return media, _check_bytes(data, f"image URL {url}")
