#!/usr/bin/env python3
"""One-time resumable parallel bootstrap for QuantDB datasets.

Uses the public SDK download API with one client per worker. Files are validated
against manifest size/ETag. Per-dataset SDK reconciliation is optional because it
can take hours; production state should be rebuilt once after the bootstrap. No
credential is stored here; credentials are supplied by sync_quantdb.make_client().
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/app")
import sync_quantdb  # type: ignore  # mounted production sync module

ROOT = Path(os.environ.get("QM_QUANTDB_DATA_DIR", "/data/quantdb")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
LOG = logging.getLogger("quantdb-bootstrap")
LOG.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
for handler in (
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(ROOT / "bootstrap.log", encoding="utf-8"),
):
    handler.setFormatter(_fmt)
    LOG.addHandler(handler)

_tls = threading.local()


def _client():
    if not hasattr(_tls, "client"):
        _tls.client = sync_quantdb.make_client()
    return _tls.client


def _target(obj: dict) -> Path:
    path = (ROOT / obj["relative_path"]).resolve()
    if ROOT != path and ROOT not in path.parents:
        raise RuntimeError(f"manifest path escapes data root: {obj['relative_path']}")
    return path


def _is_complete(path: Path, obj: dict) -> bool:
    expected = obj.get("size")
    return path.is_file() and (expected is None or path.stat().st_size == int(expected))


def _validate(path: Path, obj: dict) -> None:
    expected = obj.get("size")
    if expected is not None and path.stat().st_size != int(expected):
        raise RuntimeError(f"size mismatch for {path}: {path.stat().st_size} != {expected}")
    etag = str(obj.get("etag") or "").strip('"')
    if etag and "-" not in etag:
        digest = hashlib.md5()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != etag.lower():
            raise RuntimeError(f"ETag mismatch for {path}")


def _download(ds: dict, obj: dict, retries: int) -> str:
    target = _target(obj)
    if _is_complete(target, obj):
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    layout_name = str(obj.get("layout") or "")
    layout = "v2" if layout_name.startswith("v2") else "v1"
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            kwargs = dict(
                category_id=ds["category_id"],
                sub_category=ds["sub_category"],
                save_dir=str(target.parent),
                layout=layout,
            )
            if layout == "v2":
                kwargs["object_key"] = obj.get("key")
            else:
                kwargs["symbol"] = obj.get("symbol")
                if obj.get("trade_date"):
                    kwargs["trade_date"] = obj.get("trade_date")
            downloaded = Path(_client().download_file(**kwargs)).resolve()
            if downloaded != target:
                os.replace(downloaded, target)
            _validate(target, obj)
            return "downloaded"
        except Exception as exc:
            error = exc
            part = Path(str(target) + ".part")
            if part.exists():
                part.unlink()
            if target.exists() and not _is_complete(target, obj):
                target.unlink()
            _tls.client = sync_quantdb.make_client()
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{target}: {error}")


def _run_dataset(
    ds: dict, workers: int, retries: int, reconcile_state: bool
) -> tuple[int, int]:
    name = f"{ds['dir']}/{ds['sub_category']}"
    manifest = sync_quantdb.make_client().query_manifest(ds["category_id"], ds["sub_category"])
    missing = [obj for obj in manifest if not _is_complete(_target(obj), obj)]
    LOG.info("[%s] manifest=%d missing=%d workers=%d", name, len(manifest), len(missing), workers)
    downloaded = 0
    errors: list[str] = []
    started = time.time()
    if missing:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qdb") as pool:
            futures = [pool.submit(_download, ds, obj, retries) for obj in missing]
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    if future.result() == "downloaded":
                        downloaded += 1
                except Exception as exc:
                    errors.append(str(exc))
                if done % 250 == 0 or done == len(futures):
                    elapsed = max(time.time() - started, 0.001)
                    LOG.info(
                        "[%s] progress=%d/%d rate=%.2f files/s errors=%d",
                        name,
                        done,
                        len(futures),
                        done / elapsed,
                        len(errors),
                    )
    if errors:
        for err in errors[:10]:
            LOG.error("[%s] %s", name, err)
        raise RuntimeError(f"{name}: {len(errors)} files failed")

    reconciled: int | str = "skipped"
    if reconcile_state:
        reconcile = sync_quantdb.make_client().sync_dataset(
            dataset=ds["sub_category"], save_dir=str(ROOT)
        )
        reconciled = len(reconcile.get("matched", []))
    local = sum(
        1
        for _ in (ROOT / ds["dir"] / ds["sub_category"]).rglob("*.parquet")
    )
    if local < len(manifest):
        raise RuntimeError(f"{name}: local count {local} < manifest count {len(manifest)}")
    LOG.info(
        "[%s] complete local=%d manifest=%d downloaded=%d reconciled=%s",
        name,
        local,
        len(manifest),
        downloaded,
        reconciled,
    )
    return local, len(manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--dataset")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="run slow SDK reconciliation after every dataset",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 160:
        parser.error("--workers must be between 1 and 160")
    datasets = sync_quantdb.V2_DATASETS + sync_quantdb.V1_DATASETS
    if args.dataset:
        datasets = [ds for ds in datasets if ds["sub_category"] == args.dataset]
        if not datasets:
            parser.error(f"unknown dataset: {args.dataset}")
    LOG.info("bootstrap start root=%s datasets=%d workers=%d", ROOT, len(datasets), args.workers)
    completed = 0
    try:
        for ds in datasets:
            _run_dataset(ds, args.workers, args.retries, args.reconcile)
            completed += 1
    except Exception:
        LOG.exception("bootstrap failed after %d/%d datasets", completed, len(datasets))
        return 1
    LOG.info("bootstrap complete datasets=%d", completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
