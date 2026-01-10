from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List

try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore


@dataclass
class PrincipleSpec:
    id: str
    name: str
    module: str
    detector: str = "detect"
    description: str | None = None
    visualize: str | None = None  # Optional visualize function name
    aliases: list[str] | None = None  # Optional list of alias ids (strings)


class PrinciplesEngine:
    """Loads principle detectors from a registry and evaluates them on positions."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.specs: List[PrincipleSpec] = []
        self.detectors: Dict[str, Callable[[Any], bool]] = {}
        # Optional rich detectors returning structured dicts (see detect_info contract)
        self.detectors_info: Dict[str, Callable[[Any], Dict[str, Any]]] = {}
        self.visualizers: Dict[str, Callable[[Any], Any]] = {}
        self.alias_map: Dict[str, str] = {}  # alias -> canonical id
        self._load_registry()

    def _load_registry(self):
        import json
        # reset
        self.specs = []
        self.detectors = {}
        self.detectors_info = {}
        self.visualizers = {}
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        specs: List[PrincipleSpec] = []
        for p in data.get("principles", []):
            specs.append(PrincipleSpec(
                id=p["id"],
                name=p.get("name", p["id"]),
                module=p["module"],
                detector=p.get("detector", "detect"),
                description=p.get("description"),
                visualize=p.get("visualize"),
                aliases=p.get("aliases") or []
            ))
        self.specs = specs
        # Build alias map (case-insensitive exact match on provided alias strings)
        self.alias_map = {}
        for s in self.specs:
            for a in (s.aliases or []):
                if not a:
                    continue
                key = a.strip()
                if key and key not in self.alias_map:
                    self.alias_map[key] = s.id
        # Load detectors from sibling modules under backend/principles
        base_pkg = __package__  # 'principles'
        for s in self.specs:
            try:
                mod = import_module(f"{base_pkg}.{s.module}")
                fn = getattr(mod, s.detector)
                self.detectors[s.id] = fn  # callable(board)->bool OR legacy bool
                # Structured detector optional name 'detect_info'
                if hasattr(mod, 'detect_info'):
                    info_fn = getattr(mod, 'detect_info')
                    if callable(info_fn):
                        self.detectors_info[s.id] = info_fn  # callable(board)->dict
                # Optional visualizer
                viz_name = s.visualize or "visualize"
                if hasattr(mod, viz_name):
                    viz_fn = getattr(mod, viz_name)
                    self.visualizers[s.id] = viz_fn  # callable(board)->overlay dict or list
            except Exception:
                # Skip missing detectors/visualizers
                continue

    def reload(self):
        self._load_registry()

    def list_ids(self) -> List[str]:
        return [s.id for s in self.specs]

    def list_aliases(self) -> Dict[str, str]:
        return dict(self.alias_map)

    def resolve_id(self, pid: str) -> str:
        """Return canonical principle id for a given id or alias. Falls back to input."""
        if not pid:
            return pid
        # Exact match first
        if any(s.id == pid for s in self.specs):
            return pid
        # Alias map (case-sensitive then case-insensitive)
        if pid in self.alias_map:
            return self.alias_map[pid]
        for k, v in self.alias_map.items():
            if k.lower() == pid.lower():
                return v
        return pid

    def list_specs(self) -> List[PrincipleSpec]:
        return self.specs

    def analyze(self, board: Any) -> List[str]:
        if chess is None or board is None:
            return []
        tags: List[str] = []
        for pid, fn in self.detectors.items():
            try:
                res = fn(board)
                if isinstance(res, dict):  # allow detectors that upgraded detect() itself
                    if res.get('detected'):
                        tags.append(pid)
                elif res:
                    tags.append(pid)
            except Exception:
                continue
        return tags

    def visualize(self, board: Any, tags: List[str]) -> Dict[str, Any]:
        """Aggregate overlays from visualizers for the given tags.
        Returns a structure like {"arrows": [...], "highlights": [...]} suitable for the frontend.
        """
        overlays: Dict[str, Any] = {"arrows": [], "highlights": []}
        if board is None:
            return overlays
        for pid in tags:
            viz = self.visualizers.get(pid)
            if not viz:
                continue
            try:
                out = viz(board)
                if not out:
                    continue
                # Merge known keys
                for key in ("arrows", "highlights"):
                    if key in out and isinstance(out[key], list):
                        overlays[key].extend(out[key])
            except Exception:
                continue
        return overlays
