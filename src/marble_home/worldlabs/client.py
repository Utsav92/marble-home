"""World Labs "World API" client (Marble).

Endpoints below were read from the official reference on 2026-10-24 (docs.worldlabs.ai/api/reference/...):
    POST {base}/marble/v1/worlds:generate          body: WorldsGenerateRequest  -> {operation_id, done, ...}
    GET  {base}/marble/v1/operations/{operation_id}                              -> {done, error, response, metadata}
    GET  {base}/marble/v1/worlds/{world_id}                                      -> World (assets.mesh.collider_mesh_url, ...)
    POST {base}/marble/v1/worlds/{world_id}:export body {asset_type: splats|mesh, format: ply|glb, ...}
Auth header: ``WLT-Api-Key``. Credits are billed (402 = insufficient credits). Rate limits are NOT documented.

Network access is injectable (``session``) so offline tests never touch the API or spend credits.
The API key is read from WORLDLABS_API_KEY and is never logged or written to disk.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE = "https://api.worldlabs.ai"
MODELS = ("marble-1.0-draft", "marble-1.0", "marble-1.1", "marble-1.1-plus")


class WorldLabsError(RuntimeError):
    pass


class InsufficientCredits(WorldLabsError):
    pass


class MissingApiKey(WorldLabsError):
    pass


class WorldLabsClient:
    def __init__(self, api_key: str | None = None, base: str | None = None, session: Any = None, timeout: float = 60.0):
        self._key = api_key if api_key is not None else os.environ.get("WORLDLABS_API_KEY", "")
        self.base = (base or os.environ.get("WORLDLABS_API_BASE") or DEFAULT_BASE).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    @property
    def has_key(self) -> bool:
        return bool(self._key)

    # ------------------------------------------------------------------ plumbing
    def _headers(self) -> dict:
        if not self._key:
            raise MissingApiKey("WORLDLABS_API_KEY is not set (see .env.example); use the offline importer instead")
        return {"WLT-Api-Key": self._key, "Content-Type": "application/json"}

    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        r = self.session.request(method, f"{self.base}{path}", headers=self._headers(), json=json, timeout=self.timeout)
        if r.status_code == 402:
            raise InsufficientCredits("402: insufficient World Labs API credits (add credits / enable auto-refill)")
        if r.status_code >= 400:
            raise WorldLabsError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    # -------------------------------------------------------------------- API
    @staticmethod
    def build_generate_request(prompt: str, model: str = "marble-1.1", seed: int | None = None,
                               display_name: str | None = None, tags: list[str] | None = None, public: bool = False) -> dict:
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        body: dict = {"model": model, "permission": {"public": public},
                      "world_prompt": {"type": "text", "text_prompt": prompt}}
        if seed is not None:
            if not 0 <= seed <= 4294967295:
                raise ValueError("seed must be in [0, 2^32-1]")
            body["seed"] = seed
        if display_name:
            body["display_name"] = display_name[:64]
        if tags:
            body["tags"] = [t[:32] for t in tags[:10]]
        return body

    def generate(self, prompt: str, **kw) -> dict:
        return self._request("POST", "/marble/v1/worlds:generate", self.build_generate_request(prompt, **kw))

    def get_operation(self, operation_id: str) -> dict:
        return self._request("GET", f"/marble/v1/operations/{operation_id}")

    def wait(self, operation_id: str, timeout: float = 1800.0, poll: float = 10.0) -> dict:
        t0 = time.time()
        while True:
            op = self.get_operation(operation_id)
            if op.get("done"):
                if op.get("error"):
                    raise WorldLabsError(f"operation {operation_id} failed: {op['error']}")
                return op
            if time.time() - t0 > timeout:
                raise WorldLabsError(f"operation {operation_id} not done after {timeout}s")
            time.sleep(poll)

    def get_world(self, world_id: str) -> dict:
        return self._request("GET", f"/marble/v1/worlds/{world_id}")

    def export(self, world_id: str, asset_type: str, fmt: str, mesh_variant: str | None = None,
               resolution: str | None = None) -> dict:
        if asset_type not in ("splats", "mesh") or fmt not in ("ply", "glb"):
            raise ValueError("asset_type in {splats, mesh}; format in {ply, glb}")
        body = {"asset_type": asset_type, "format": fmt}
        if mesh_variant:
            body["mesh_variant"] = mesh_variant
        if resolution:
            body["resolution"] = resolution
        return self._request("POST", f"/marble/v1/worlds/{world_id}:export", body)

    def download(self, url: str, dest: str | Path) -> dict:
        """Stream a (pre-signed) asset URL to disk; returns {path, bytes, sha256}. No auth header is sent."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        h, n = hashlib.sha256(), 0
        with self.session.get(url, stream=True, timeout=self.timeout) as r:
            if r.status_code >= 400:
                raise WorldLabsError(f"download failed HTTP {r.status_code}")
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
        return {"path": str(dest), "bytes": n, "sha256": h.hexdigest()}
