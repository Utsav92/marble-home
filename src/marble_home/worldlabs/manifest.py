"""Generation manifest + versioned asset registry for Marble worlds (spec section 4.2 / 4.4).

``data/marble_raw/<scene_id>/`` keeps the ORIGINAL downloaded files untouched (no destructive conversion);
processed outputs live under ``data/processed_worlds/<scene_id>/v<N>/``.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from marble_home.utils import provenance
from marble_home.utils.paths import DATA, ensure

CATEGORIES = ("kitchen", "living_room", "bedroom", "laundry_room", "multiroom_apartment")


@dataclass
class SceneRecord:
    scene_id: str
    category: str
    prompt: str
    seed: int | None = None
    model: str | None = None
    world_id: str | None = None
    operation_id: str | None = None
    api_status: str = "not_requested"     # not_requested | dry_run | pending | done | failed | offline_import | synthetic_fixture
    export_paths: dict = field(default_factory=dict)   # {"collider_glb": "...", "splat_ply": "...", ...}
    checksums: dict = field(default_factory=dict)      # sha256 per exported file
    scale: float | None = None                         # metric_scale_factor (raw coords -> metres); None = unknown
    scale_source: str = "unknown"                      # api | manual | estimated | fixture
    ground_plane_offset: float | None = None
    coordinate_frame: str = "glTF: +Y up, metres after scale; ground y=0 after subtracting ground_plane_offset"
    complexity: dict = field(default_factory=dict)     # faces / vertices / components
    generation_time_s: float | None = None
    cost_credits: int | None = None
    created_at: float = field(default_factory=time.time)
    notes: list = field(default_factory=list)


class AssetRegistry:
    """JSON-backed registry: data/marble_raw/registry.json (append-only versions per scene)."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else DATA / "marble_raw"
        ensure(self.root)
        self.path = self.root / "registry.json"
        self.db = json.loads(self.path.read_text()) if self.path.exists() else {"scenes": {}}

    def register(self, rec: SceneRecord) -> int:
        versions = self.db["scenes"].setdefault(rec.scene_id, [])
        versions.append(asdict(rec))
        self.path.write_text(json.dumps(self.db, indent=2))
        return len(versions)

    def latest(self, scene_id: str) -> dict | None:
        v = self.db["scenes"].get(scene_id)
        return v[-1] if v else None

    def scenes(self) -> list[str]:
        return sorted(self.db["scenes"])


def checksum_files(paths: dict) -> dict:
    return {k: provenance.file_sha256(p) for k, p in paths.items() if p and Path(p).exists()}
