"""Minimal ASGI compatibility layer for local tests when FastAPI is unavailable.

Production environments install FastAPI from ``requirements.txt`` and use its
implementation. This fallback is intentionally limited to the two public
routes defined by this phase.
"""

import json
from types import SimpleNamespace

from pydantic import ValidationError


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


class Request:
    def __init__(self, app, scope):
        self.app = app
        self.scope = scope


class CORSMiddleware:
    def __init__(self, app, *, allow_origins, allow_credentials, allow_methods, allow_headers):
        self.app = app
        self.allow_origins = allow_origins

    async def __call__(self, scope, receive, send):
        origin = dict(scope.get("headers", [])).get(b"origin", b"").decode()
        async def send_with_cors(message):
            if message["type"] == "http.response.start" and origin in self.allow_origins:
                headers = list(message.get("headers", []))
                headers.append((b"access-control-allow-origin", origin.encode()))
                message = {**message, "headers": headers}
            await send(message)
        await self.app(scope, receive, send_with_cors)


class FastAPI:
    def __init__(self, **kwargs):
        self.routes = {}
        self.state = SimpleNamespace()

    def add_middleware(self, middleware, **kwargs):
        self._middleware = middleware(self._dispatch, **kwargs)

    def get(self, path, response_model=None):
        return self._register("GET", path, response_model)

    def post(self, path, response_model=None):
        return self._register("POST", path, response_model)

    def _register(self, method, path, response_model=None):
        def decorator(function):
            self.routes[(method, path)] = (function, response_model)
            return function
        return decorator

    async def __call__(self, scope, receive, send):
        if hasattr(self, "_middleware"):
            return await self._middleware(scope, receive, send)
        return await self._dispatch(scope, receive, send)

    async def _dispatch(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            
        if scope["type"] != "http":
            return
        route = self.routes.get((scope["method"], scope["path"]))
        if route is None:
            return await self._send(send, 404, {"detail": "Not found"})
        function, response_model = route
        try:
            if scope["method"] == "POST":
                body = b""
                while True:
                    message = await receive()
                    body += message.get("body", b"")
                    if not message.get("more_body"):
                        break
                try:
                    payload = function.__annotations__["payload"].model_validate(json.loads(body))
                except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                    return await self._send(send, 422, {"detail": "Invalid request body"})
                value = function(payload, Request(self, scope))
            else:
                value = function()
            data = response_model.model_validate(value).model_dump(mode="json") if response_model else value
            return await self._send(send, 200, data)
        except HTTPException as exc:
            return await self._send(send, exc.status_code, {"detail": exc.detail})

    @staticmethod
    async def _send(send, status, data):
        body = json.dumps(data).encode()
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
