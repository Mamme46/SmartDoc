import json
import os
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from database import delete_session, get_session_user_id

COOKIE_NAME = "smartdoc_session"
COOKIE_MAX_AGE = 12 * 60 * 60
COOKIE_SECURE = os.environ.get("SMARTDOC_COOKIE_SECURE", "false").lower() in {
    "1",
    "true",
    "yes",
}


class SessionCookieHandler(BaseHTTPRequestHandler):
    def _send_response(self, status, cookie_header=None):
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        if cookie_header:
            self.send_header("Set-Cookie", cookie_header)
        self.end_headers()

    def _is_same_origin(self):
        origin = urlsplit(self.headers.get("Origin", ""))
        host = self.headers.get("Host", "")
        return (
            origin.scheme in {"http", "https"}
            and bool(origin.netloc)
            and origin.netloc.lower() == host.lower()
        )

    def _read_json_body(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if content_length <= 0 or content_length > 2048:
            return None
        try:
            return json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def do_POST(self):
        if self.path not in {"/set", "/clear"}:
            self._send_response(404)
            return
        if not self._is_same_origin():
            self._send_response(403)
            return

        if self.path == "/set":
            payload = self._read_json_body()
            token = payload.get("token") if isinstance(payload, dict) else None
            if not isinstance(token, str) or get_session_user_id(token) is None:
                self._send_response(401)
                return

            cookie = (
                f"{COOKIE_NAME}={token}; Path=/; Max-Age={COOKIE_MAX_AGE}; "
                "HttpOnly; SameSite=Strict"
            )
            if COOKIE_SECURE:
                cookie += "; Secure"
            self._send_response(204, cookie)
            return

        try:
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            token = cookies.get(COOKIE_NAME)
        except CookieError:
            token = None
        if token:
            delete_session(token.value)
        self._send_response(
            204,
            f"{COOKIE_NAME}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"
            + ("; Secure" if COOKIE_SECURE else ""),
        )

    def log_request(self, code="-", size="-"):
        self.log_message("%s %s", self.command, self.path.split("?", 1)[0])

    def log_message(self, format_string, *args):
        super().log_message(format_string, *args)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8502), SessionCookieHandler)
    server.serve_forever()
