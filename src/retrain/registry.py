from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.observability.logging import get_logger

logger = get_logger("retrain.registry")

_STATUSES = ("candidate", "production", "retired")


@dataclass
class ModelVersion:
    version: str
    path: str
    base_model: str
    metrics: dict[str, Any] = field(default_factory=dict)
    dataset_stats: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Versioned classifier artifacts with promote/rollback (Phase 10.4).

    Layout: models/<version>/{config.json, model.safetensors, tokenizer files,...}
    Registry index: models/registry.json — list of ModelVersion dicts.
    """

    def __init__(self, registry_path: str | Path | None = None) -> None:
        settings = get_settings()
        self.registry_path = Path(registry_path or settings.retrain_registry_path)

    # -- index ---------------------------------------------------------------

    def _load(self) -> list[ModelVersion]:
        if not self.registry_path.exists():
            return []
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return [ModelVersion(**item) for item in data]
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("registry unreadable, starting empty", error=str(e))
            return []

    def _save(self, versions: list[ModelVersion]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps([v.to_dict() for v in versions], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- operations ----------------------------------------------------------

    def register(
        self,
        version: str,
        path: str | Path,
        *,
        base_model: str,
        metrics: dict[str, Any] | None = None,
        dataset_stats: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """Add a trained artifact as candidate (artifact dir is copied if remote)."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"model artifact not found: {path}")
        versions = self._load()
        if any(v.version == version for v in versions):
            raise ValueError(f"version already registered: {version}")
        entry = ModelVersion(
            version=version,
            path=str(path),
            base_model=base_model,
            metrics=metrics or {},
            dataset_stats=dataset_stats or {},
        )
        versions.append(entry)
        self._save(versions)
        logger.info("model registered", version=version, path=str(path))
        return entry

    def promote(self, version: str) -> ModelVersion:
        """Make one version production; previous production is retired (rollback = promote it back)."""
        versions = self._load()
        target = next((v for v in versions if v.version == version), None)
        if target is None:
            raise KeyError(f"unknown version: {version}")
        for v in versions:
            v.status = (
                "production"
                if v.version == version
                else ("retired" if v.status == "production" else v.status)
            )
        self._save(versions)
        logger.info("model promoted", version=version)
        return target

    def production(self) -> ModelVersion | None:
        versions = self._load()
        return next((v for v in versions if v.status == "production"), None)

    def candidates(self) -> list[ModelVersion]:
        return [v for v in self._load() if v.status == "candidate"]

    def list(self) -> list[ModelVersion]:
        return self._load()

    def remove(self, version: str, *, delete_files: bool = False) -> bool:
        """Drop a version from the index (production cannot be removed)."""
        versions = self._load()
        target = next((v for v in versions if v.version == version), None)
        if target is None:
            return False
        if target.status == "production":
            raise ValueError("cannot remove production version; promote another first")
        versions.remove(target)
        self._save(versions)
        if delete_files:
            shutil.rmtree(target.path, ignore_errors=True)
        logger.info("model removed", version=version, deleted_files=delete_files)
        return True


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.retrain.registry",
        description="Model registry: list / promote / remove classifier versions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list versions")
    p_promote = sub.add_parser("promote", help="promote version to production")
    p_promote.add_argument("version")
    p_remove = sub.add_parser("remove", help="remove a version from the index")
    p_remove.add_argument("version")
    p_remove.add_argument("--delete-files", action="store_true")
    sub.add_parser("show", help="show production version")
    args = parser.parse_args()

    registry = ModelRegistry()
    if args.command == "list":
        for v in registry.list():
            print(f"{v.version:16} {v.status:10} {v.path}")
    elif args.command == "promote":
        entry = registry.promote(args.version)
        print(json.dumps(entry.to_dict(), indent=2, ensure_ascii=False))
    elif args.command == "remove":
        removed = registry.remove(args.version, delete_files=args.delete_files)
        print("removed" if removed else "not found")
    elif args.command == "show":
        prod = registry.production()
        print(
            json.dumps(prod.to_dict(), indent=2, ensure_ascii=False)
            if prod
            else "no production version"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
