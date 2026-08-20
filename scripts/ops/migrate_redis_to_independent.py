#!/usr/bin/env python3
"""Copy selected logical Redis databases to an independent instance."""

from __future__ import annotations

import argparse
import json
import os

import redis


def parse_mapping(value: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in value.split(","):
        source, target = item.strip().split(":", 1)
        mapping[int(source)] = int(target)
    if not mapping:
        raise argparse.ArgumentTypeError("mapping must not be empty")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-host", default=os.environ.get("REDIS_HOST", "localhost")
    )
    parser.add_argument(
        "--source-port", type=int, default=int(os.environ.get("REDIS_PORT", "6379"))
    )
    parser.add_argument("--source-username", default=os.environ.get("REDIS_USERNAME"))
    parser.add_argument("--source-password", default=os.environ.get("REDIS_PASSWORD"))
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, default=6379)
    parser.add_argument("--target-username")
    parser.add_argument("--target-password")
    parser.add_argument(
        "--mapping",
        type=parse_mapping,
        default=parse_mapping("8:0,9:1,10:2,11:3,13:5,14:0"),
        help="comma-separated source:target logical DB mapping",
    )
    parser.add_argument(
        "--flush-target",
        action="store_true",
        help="flush mapped target DBs before copying; use only during cutover",
    )
    return parser


def client(
    *, host: str, port: int, db: int, username: str | None, password: str | None
) -> redis.Redis:
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        username=username or None,
        password=password or None,
        socket_connect_timeout=10,
        socket_timeout=30,
    )


def main() -> None:
    args = build_parser().parse_args()

    if args.flush_target:
        for target_db in sorted(set(args.mapping.values())):
            target = client(
                host=args.target_host,
                port=args.target_port,
                db=target_db,
                username=args.target_username,
                password=args.target_password,
            )
            target.flushdb()

    result: dict[str, dict[str, int]] = {}
    for source_db, target_db in args.mapping.items():
        source = client(
            host=args.source_host,
            port=args.source_port,
            db=source_db,
            username=args.source_username,
            password=args.source_password,
        )
        target = client(
            host=args.target_host,
            port=args.target_port,
            db=target_db,
            username=args.target_username,
            password=args.target_password,
        )

        source_size = source.dbsize()
        copied = 0
        for key in source.scan_iter(count=500):
            payload = source.dump(key)
            if payload is None:
                continue
            ttl_ms = source.pttl(key)
            target.restore(key, max(ttl_ms, 0), payload, replace=True)
            copied += 1

        if copied != source_size:
            raise RuntimeError(
                f"source DB {source_db} changed during copy: "
                f"expected {source_size} keys, copied {copied}"
            )
        result[f"{source_db}->{target_db}"] = {
            "source_keys": source_size,
            "copied_keys": copied,
        }

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
