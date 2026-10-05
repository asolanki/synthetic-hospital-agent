"""Upload prepared archive.tar.gz.part-* files through the authenticated cf CLI.

Parts must already form a verified gzip stream and be smaller than the REST
endpoint's 300 MB limit. Existing matching parts are reused; conflicting parts
are rejected. manifest.json is published last after every part is verified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def cf(*args: str):
    process = subprocess.run(["cf", *args], capture_output=True, text=True, timeout=600)
    if process.returncode:
        raise RuntimeError(f"cf exited {process.returncode}: {process.stderr}\n{process.stdout}")
    return json.loads(process.stdout)


def remote_objects(bucket: str, prefix: str) -> dict:
    objects = {}
    after = None
    while True:
        args = ["r2", "objects", "list", "--bucket-name", bucket,
                "--prefix", prefix, "--per-page", "1000"]
        if after is not None:
            args.extend(["--start-after", after])
        page = cf(*args)
        objects.update((item["key"], item) for item in page)
        if len(page) < 1000:
            return objects
        after = max(item["key"] for item in page)


def hashes(path: Path, archive_hash=None) -> dict:
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            sha256.update(block)
            md5.update(block)
            if archive_hash is not None:
                archive_hash.update(block)
            size += len(block)
    return {"bytes": size, "sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def verify(info: dict, metadata: dict) -> None:
    if int(metadata["size"]) != info["bytes"] or metadata["etag"].strip('"') != info["md5"]:
        raise ValueError(f"Remote size/ETag mismatch for {info['key']}; refusing to overwrite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts_dir", type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    prefix = args.prefix.rstrip("/") + "/"
    paths = sorted(args.parts_dir.glob("archive.tar.gz.part-*"))
    if not paths:
        parser.error("No prepared archive parts found")
    archive_hash = hashlib.sha256()
    parts = []
    for path in paths:
        info = {"file": path.name, "key": prefix + path.name, **hashes(path, archive_hash)}
        if not 0 < info["bytes"] < 300_000_000:
            parser.error(f"Part violates the REST upload-size limit: {path}")
        parts.append(info)
    existing = remote_objects(args.bucket, prefix)

    def upload(info: dict) -> None:
        if info["key"] in existing:
            verify(info, existing[info["key"]])
        else:
            metadata = cf(
                "r2", "objects", "put", info["key"], "--bucket-name", args.bucket,
                "--file", str(args.parts_dir / info["file"]),
                "--content-type", "application/octet-stream",
                "--content-length", str(info["bytes"]),
            )
            verify(info, metadata)
        print(json.dumps({"verified_key": info["key"], "bytes": info["bytes"]}), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(upload, parts))
    remote = remote_objects(args.bucket, prefix)
    for info in parts:
        verify(info, remote[info["key"]])
    manifest = {
        "format": "ordered chunks of one tar.gz stream",
        "bucket": args.bucket,
        "prefix": prefix,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": archive_hash.hexdigest(),
        "archive_bytes": sum(info["bytes"] for info in parts),
        "parts": parts,
        "verification": "Every part's server-reported size and ETag match local bytes and MD5; SHA-256 hashes permit independent restoration checks.",
        "restore": "Download all listed parts, verify their SHA-256 hashes, concatenate in manifest order, then extract the tar.gz stream.",
    }
    manifest_path = args.parts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    info = {"key": prefix + "manifest.json", **hashes(manifest_path)}
    metadata = cf("r2", "objects", "put", info["key"], "--bucket-name", args.bucket,
                  "--file", str(manifest_path), "--content-type", "application/json")
    verify(info, metadata)
    print(json.dumps({"archive": f"r2://{args.bucket}/{prefix}",
                      "parts": len(parts), "bytes": manifest["archive_bytes"],
                      "manifest_sha256": info["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
