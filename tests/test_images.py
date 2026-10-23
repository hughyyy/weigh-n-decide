"""Image input normalization tests: paths, data URIs, http(s) URLs, redirects,
size limits, and error paths — against a local http.server (stdlib, no
network)."""

import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

from weigh_n_decide.images import (
    ImageInputError,
    MAX_IMAGE_BYTES,
    load_image,
)

# 1x1 PNG
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

import base64

PNG_BYTES = base64.b64decode(PNG_B64)
DATA_URI = f"data:image/png;base64,{PNG_B64}"


class _ImageServer(BaseHTTPRequestHandler):
    """Serves PNG bytes with per-server status/content type/redirects."""

    def do_GET(self):  # noqa: N802 - http.server API
        config = self.server.config
        if self.path in config["path_map"]:
            target = config["path_map"][self.path]
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return
        data = config["body"]
        declared = config["declared_length"]
        self.send_response(config["status"])
        if config["content_type"] is not None:
            self.send_header("Content-Type", config["content_type"])
        if not config["omit_length"]:
            self.send_header(
                "Content-Length", str(declared if declared is not None else len(data))
            )
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client rejected the headers and closed early
        if declared is not None:
            self.close_connection = True

    def log_message(self, *args):  # silence test output
        pass


class ImageServer:
    def __init__(
        self, body=PNG_BYTES, status=200, content_type="image/png",
        path_map=None, declared_length=None, omit_length=False,
    ):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ImageServer)
        self.httpd.config = {
            "body": body,
            "status": status,
            "content_type": content_type,
            "path_map": path_map or {},
            "declared_length": declared_length,
            "omit_length": omit_length,
        }
        self.thread = threading.Thread(
            target=lambda: self.httpd.serve_forever(poll_interval=0.01),
            daemon=True,
        )
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def set_path_map(self, mapping):
        self.httpd.config["path_map"] = mapping

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class BasicTests(unittest.TestCase):
    def test_non_string_input_rejected(self):
        for bad in (None, "", "   ", 42, ["a.png"], b"bytes"):
            with self.assertRaises(ImageInputError, msg=repr(bad)):
                load_image(bad)

    def test_data_uri_roundtrip(self):
        media, data = load_image(DATA_URI)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_data_uri_defaults_media_type(self):
        media, data = load_image("data:;base64," + PNG_B64)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_data_uri_with_parameters(self):
        media, _ = load_image(f"data:image/jpeg;charset=utf-8;base64,{PNG_B64}")
        self.assertEqual(media, "image/jpeg")

    def test_malformed_data_uri(self):
        # Not base64 at all: `data:` with a non-base64 payload.
        with self.assertRaises(ImageInputError):
            load_image("data:image/png;base64,@@@")
        # No comma separator at all.
        with self.assertRaises(ImageInputError):
            load_image("data:image/png")

    def test_empty_data_uri_is_an_error(self):
        with self.assertRaises(ImageInputError) as ctx:
            load_image("data:image/png;base64,")
        self.assertIn("empty", str(ctx.exception))

    def test_oversized_data_uri_is_an_error(self):
        # Multiple of 4 so base64 itself decodes; decoded size must exceed 25 MiB.
        chars = (MAX_IMAGE_BYTES * 4 + 2) // 3
        big = "A" * (chars - chars % 4 + 4 if chars % 4 else chars)
        self.assertEqual(len(big) % 4, 0)
        with self.assertRaises(ImageInputError) as ctx:
            load_image(f"data:image/png;base64,{big}")
        self.assertIn("25 MiB limit", str(ctx.exception))

    def test_uppercase_data_scheme_is_accepted(self):
        media, data = load_image("DATA:image/png;base64," + PNG_B64)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_invalid_oversized_payload_reports_bad_base64(self):
        # Precedence: an undecodable payload reports the base64 error even
        # when it would also exceed the size limit.
        big = "A" * (MAX_IMAGE_BYTES + 1)
        with self.assertRaises(ImageInputError) as ctx:
            load_image(f"data:image/png;base64,{big}")
        self.assertIn("malformed base64", str(ctx.exception))

    def test_base64_parameter_is_case_insensitive(self):
        media, data = load_image("data:image/png;BASE64," + PNG_B64)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_non_base64_data_uri_is_rejected(self):
        with self.assertRaises(ImageInputError) as ctx:
            load_image("data:image/png,hello")
        self.assertIn("malformed base64", str(ctx.exception))


class PathTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "pic.png")
        with open(self.path, "wb") as fh:
            fh.write(PNG_BYTES)

    def test_path_roundtrip(self):
        media, data = load_image(self.path)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_path_without_suffix_defaults_to_png(self):
        bare = os.path.join(self.dir.name, "naked")
        with open(bare, "wb") as fh:
            fh.write(PNG_BYTES)
        media, _ = load_image(bare)
        self.assertEqual(media, "image/png")

    def test_missing_file_is_an_error(self):
        with self.assertRaises(ImageInputError) as ctx:
            load_image(os.path.join(self.dir.name, "nope.png"))
        self.assertIn("cannot read image file", str(ctx.exception))

    def test_directory_is_an_error(self):
        with self.assertRaises(ImageInputError):
            load_image(self.dir.name)

    def test_empty_file_is_an_error(self):
        empty = os.path.join(self.dir.name, "empty.png")
        open(empty, "wb").close()
        with self.assertRaises(ImageInputError) as ctx:
            load_image(empty)
        self.assertIn("is empty", str(ctx.exception))

    def test_oversized_file_is_an_error(self):
        big = os.path.join(self.dir.name, "big.png")
        with open(big, "wb") as fh:
            fh.truncate(MAX_IMAGE_BYTES + 1)
        with self.assertRaises(ImageInputError) as ctx:
            load_image(big)
        self.assertIn("25 MiB limit", str(ctx.exception))

    def test_path_with_a_colon_is_still_a_path(self):
        path = os.path.join(self.dir.name, "note:1.png")
        with open(path, "wb") as fh:
            fh.write(PNG_BYTES)
        media, data = load_image(path)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_uppercase_suffix_reports_the_right_media_type(self):
        path = os.path.join(self.dir.name, "pic.PNG")
        with open(path, "wb") as fh:
            fh.write(PNG_BYTES)
        media, _ = load_image(path)
        self.assertEqual(media, "image/png")

    def test_webp_is_reported_not_guessed(self):
        path = os.path.join(self.dir.name, "pic.webp")
        with open(path, "wb") as fh:
            fh.write(PNG_BYTES)
        media, _ = load_image(path)
        self.assertEqual(media, "image/webp")


class UrlTests(unittest.TestCase):
    def setUp(self):
        self.server = ImageServer()

    def tearDown(self):
        self.server.stop()

    def test_url_roundtrip(self):
        media, data = load_image(self.server.url + "/pic.png")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_uppercase_url_scheme_is_fetched(self):
        url = self.server.url.replace("http://", "HTTP://") + "/pic.png"
        media, data = load_image(url)
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_content_type_non_image_defaults_to_png(self):
        self.server.stop()
        self.server = ImageServer(content_type="application/octet-stream")
        media, _ = load_image(self.server.url + "/pic")
        self.assertEqual(media, "image/png")

    def test_content_type_with_parameters(self):
        self.server.stop()
        self.server = ImageServer(content_type="image/jpeg; charset=utf-8")
        media, _ = load_image(self.server.url + "/pic")
        self.assertEqual(media, "image/jpeg")

    def test_missing_content_type_defaults_to_png(self):
        self.server.stop()
        self.server = ImageServer(content_type=None)
        media, _ = load_image(self.server.url + "/pic")
        self.assertEqual(media, "image/png")

    def test_uppercase_content_type_defaults_to_png(self):
        # The image/ prefix check is case-sensitive; a server sending
        # 'IMAGE/PNG' falls back to the PNG default rather than guessing.
        self.server.stop()
        self.server = ImageServer(content_type="IMAGE/PNG")
        media, _ = load_image(self.server.url + "/pic")
        self.assertEqual(media, "image/png")

    def test_redirect_loop_is_a_clear_error(self):
        self.server.stop()
        self.server = ImageServer(path_map={"/a": "/a"})
        with self.assertRaises(ImageInputError) as ctx:
            load_image(self.server.url + "/a")
        self.assertIn("cannot fetch image URL", str(ctx.exception))

    def test_http_redirect_is_followed(self):
        self.server.stop()
        self.server = ImageServer(path_map={"/old.png": "/new.png"})
        media, data = load_image(self.server.url + "/old.png")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_uppercase_redirect_scheme_is_followed(self):
        self.server.stop()
        self.server = ImageServer()
        self.server.set_path_map(
            {"/old.png": self.server.url.replace("http://", "HTTP://") + "/pic.png"}
        )
        media, data = load_image(self.server.url + "/old.png")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_redirect_to_ftp_is_refused_by_our_handler(self):
        # urllib follows ftp redirects by default; the http(s)-only redirect
        # handler is the guard. A file: target never reaches it — urllib
        # rejects file:/ftp: schemes in http_error_302 first... except ftp,
        # which urllib allows, so our handler must refuse it.
        self.server.stop()
        self.server = ImageServer(path_map={"/old": "ftp://example.com/x.png"})
        with self.assertRaises(ImageInputError) as ctx:
            load_image(self.server.url + "/old")
        self.assertIn("refusing non-http redirect target", str(ctx.exception))

    def test_redirect_to_file_is_refused_by_urllib(self):
        # file: targets are rejected by urllib's own scheme check inside the
        # opener, surfacing as a fetch error before our handler is consulted.
        self.server.stop()
        self.server = ImageServer(path_map={"/old": "file:///etc/passwd"})
        with self.assertRaises(ImageInputError) as ctx:
            load_image(self.server.url + "/old")
        self.assertIn("cannot fetch image URL", str(ctx.exception))
        self.assertIn("is not allowed", str(ctx.exception))

    def test_unreachable_url_is_an_error(self):
        # Bind and release a port so nothing listens on it.
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        with self.assertRaises(ImageInputError) as ctx:
            load_image(f"http://127.0.0.1:{port}/pic.png")
        self.assertIn("cannot fetch image URL", str(ctx.exception))

    def test_http_error_status_is_an_error(self):
        self.server.stop()
        self.server = ImageServer(status=500, body=b"boom")
        with self.assertRaises(ImageInputError):
            load_image(self.server.url + "/pic.png")

    def test_empty_body_is_an_error(self):
        self.server.stop()
        self.server = ImageServer(body=b"")
        with self.assertRaises(ImageInputError) as ctx:
            load_image(self.server.url + "/pic.png")
        self.assertIn("empty", str(ctx.exception))

    def test_oversized_body_is_an_error(self):
        self.server.stop()
        self.server = ImageServer(body=b"x" * (MAX_IMAGE_BYTES + 1))
        with self.assertRaises(ImageInputError) as ctx:
            load_image(self.server.url + "/pic.png")
        self.assertIn("25 MiB limit", str(ctx.exception))


class StrictDataUriTests(unittest.TestCase):
    def test_trailing_garbage_after_valid_base64_is_rejected(self):
        with self.assertRaises(ImageInputError) as ctx:
            load_image(f"data:image/png;base64,{PNG_B64}@@@@")
        self.assertIn("malformed base64", str(ctx.exception))

    def test_whitespace_inside_base64_is_rejected(self):
        payload = PNG_B64[:8] + "\n" + PNG_B64[8:]
        with self.assertRaises(ImageInputError) as ctx:
            load_image(f"data:image/png;base64,{payload}")
        self.assertIn("malformed base64", str(ctx.exception))

    def test_url_safe_alphabet_is_rejected_rather_than_corrupted(self):
        with self.assertRaises(ImageInputError) as ctx:
            load_image("data:image/png;base64,____")
        self.assertIn("malformed base64", str(ctx.exception))


class MediaTypeSuffixTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_jpeg_suffixes_report_image_jpeg(self):
        for suffix in (".jpg", ".jpeg"):
            path = os.path.join(self.dir.name, "pic" + suffix)
            with open(path, "wb") as fh:
                fh.write(PNG_BYTES)
            media, data = load_image(path)
            self.assertEqual(media, "image/jpeg", suffix)
            self.assertEqual(data, PNG_BYTES)

    def test_exactly_the_size_limit_is_allowed(self):
        path = os.path.join(self.dir.name, "at-limit.png")
        with open(path, "wb") as fh:
            fh.truncate(MAX_IMAGE_BYTES)
        media, data = load_image(path)
        self.assertEqual(media, "image/png")
        self.assertEqual(len(data), MAX_IMAGE_BYTES)


class TruncatedResponseTests(unittest.TestCase):
    def test_truncated_body_is_a_clear_error(self):
        server = ImageServer(declared_length=len(PNG_BYTES) + 100)
        try:
            with self.assertRaises(ImageInputError) as ctx:
                load_image(server.url + "/pic.png")
            self.assertIn("cannot fetch image URL", str(ctx.exception))
        finally:
            server.stop()


class PolicyAndBoundsTests(unittest.TestCase):
    def setUp(self):
        self.server = ImageServer()
        self.addCleanup(self.server.stop)

    def test_public_policy_blocks_loopback(self):
        with self.assertRaises(ImageInputError) as ctx:
            load_image(
                self.server.url + "/pic.png", url_policy="public"
            )
        self.assertIn("non-public address", str(ctx.exception))

    def test_any_policy_allows_loopback(self):
        media, data = load_image(
            self.server.url + "/pic.png", url_policy="any"
        )
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_unknown_url_policy_is_rejected(self):
        with self.assertRaises(ImageInputError) as ctx:
            load_image(self.server.url + "/pic.png", url_policy="sometimes")
        self.assertIn("unknown image URL policy", str(ctx.exception))

    def test_declared_length_over_the_limit_is_rejected_before_reading(self):
        self.server.stop()
        self.server = ImageServer(
            body=PNG_BYTES, declared_length=MAX_IMAGE_BYTES + 1
        )
        self.addCleanup(self.server.stop)
        with self.assertRaises(ImageInputError) as ctx:
            load_image(self.server.url + "/pic.png")
        self.assertIn("MiB limit", str(ctx.exception))

    def test_non_regular_file_is_rejected_without_opening_it(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            fifo = os.path.join(directory, "pipe.png")
            os.mkfifo(fifo)
            with self.assertRaises(ImageInputError) as ctx:
                load_image(fifo)
        self.assertIn("not a regular file", str(ctx.exception))

    def test_invalid_port_is_a_clear_error(self):
        with self.assertRaises(ImageInputError) as ctx:
            load_image("http://127.0.0.1:notaport/x.png", url_policy="public")
        self.assertIn("invalid image URL port", str(ctx.exception))

    def test_dns_failure_is_a_clear_error(self):
        from unittest import mock

        with mock.patch(
            "weigh_n_decide.images.socket.getaddrinfo",
            side_effect=OSError("no dns"),
        ):
            with self.assertRaises(ImageInputError) as ctx:
                load_image("http://example.com/x.png", url_policy="public")
        self.assertIn("cannot resolve image URL host", str(ctx.exception))

    def test_all_global_addresses_pass_the_public_policy(self):
        from unittest import mock

        from weigh_n_decide.images import _ensure_url_allowed

        with mock.patch(
            "weigh_n_decide.images.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("8.8.8.8", 80))],
        ):
            _ensure_url_allowed("http://example.com/x.png", "public")

    def test_non_http_policy_target_is_rejected(self):
        from weigh_n_decide.images import _ensure_url_allowed

        with self.assertRaises(ImageInputError) as ctx:
            _ensure_url_allowed("ftp://example.com/x.png", "public")
        self.assertIn("must be an http(s) URL", str(ctx.exception))

    def test_open_failure_after_stat_is_wrapped(self):
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "pic.png")
            with open(path, "wb") as fh:
                fh.write(PNG_BYTES)
            with mock.patch("builtins.open", side_effect=OSError("denied")):
                with self.assertRaises(ImageInputError) as ctx:
                    load_image(path)
        self.assertIn("cannot read image file", str(ctx.exception))

    def test_missing_content_length_reads_to_the_bound(self):
        self.server.stop()
        self.server = ImageServer(omit_length=True)
        self.addCleanup(self.server.stop)
        media, data = load_image(self.server.url + "/pic.png")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)

    def test_malformed_content_length_is_ignored(self):
        self.server.stop()
        self.server = ImageServer(declared_length="not-a-number")
        self.addCleanup(self.server.stop)
        media, data = load_image(self.server.url + "/pic.png")
        self.assertEqual(media, "image/png")
        self.assertEqual(data, PNG_BYTES)


if __name__ == "__main__":
    unittest.main()
