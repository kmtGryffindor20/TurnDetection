#!/usr/bin/env python3
"""Re-encode existing turn clips to H.264 to reclaim disk space.

The pipeline originally wrote clips with OpenCV's ``mp4v`` encoder at its
default quality, which produces roughly 55 Mbps / ~140 MB for a 20 s 720p60
clip. This script replaces each clip in place with an H.264 version (default
CRF 23), typically 8-10x smaller, while keeping the original file name so the
Excel catalog paths stay valid.

Usage:
    python compress_clips.py dataset_turns_3class
    python compress_clips.py dataset_turns_3class --dry-run
    python compress_clips.py dataset_turns_3class --keep-original
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _probe(path: Path, ffprobe: str | None) -> dict[str, str]:
    if not ffprobe:
        return {}
    result = _run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,bit_rate",
            "-of", "default=noprint_wrappers=1", str(path),
        ],
        timeout=30,
    )
    if result.returncode != 0:
        return {}
    data = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key:
            data[key.strip()] = value.strip()
    return data


def _mb(num_bytes: int) -> float:
    return num_bytes / 1_000_000


def _encode(src: Path, dst: Path, ffmpeg: str, crf: int, preset: str) -> None:
    result = _run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src), "-an",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst),
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(
            detail[-1] if detail else f"ffmpeg exited with {result.returncode}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        nargs="?",
        default="dataset_turns_3class",
        help="folder containing *.mp4 clips (default: dataset_turns_3class)",
    )
    parser.add_argument(
        "--crf", type=int, default=23,
        help="x264 CRF; lower = better/larger (default: 23)",
    )
    parser.add_argument(
        "--preset", default="veryfast",
        help="x264 preset (default: veryfast)",
    )
    parser.add_argument(
        "--min-bitrate", type=float, default=20.0,
        help="skip H.264/HEVC clips at or below this Mbps (default: 20)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be done",
    )
    parser.add_argument(
        "--keep-original", action="store_true",
        help="write <name>.h264.mp4 instead of replacing the original",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="overwrite an existing .h264.mp4 in --keep-original mode",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 2
    encoders = _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-encoders"])
    if "libx264" not in encoders.stdout:
        print("error: this ffmpeg has no libx264 encoder", file=sys.stderr)
        return 2
    ffprobe = shutil.which("ffprobe")

    clips = sorted(
        p for p in root.rglob("*.mp4")
        if not p.name.endswith(".tmp.mp4")
        and ".transcode_cache" not in p.parts
    )
    if not clips:
        print(f"No .mp4 files under {root}")
        return 0

    before_total = sum(p.stat().st_size for p in clips)
    after_total = 0
    new_total = 0
    encoded = skipped = failed = 0

    for index, path in enumerate(clips, 1):
        info = _probe(path, ffprobe)
        codec = info.get("codec_name", "unknown")
        try:
            bitrate = int(info.get("bit_rate", "")) / 1_000_000
        except (TypeError, ValueError):
            bitrate = None

        if (
            codec in ("h264", "hevc")
            and bitrate is not None
            and bitrate <= args.min_bitrate
        ):
            after_total += path.stat().st_size
            skipped += 1
            print(f"[{index}/{len(clips)}] skip {path} ({codec}, {bitrate:.1f} Mbps)")
            continue

        if args.keep_original:
            target = path.with_name(path.stem + ".h264.mp4")
            if target.exists() and not args.overwrite:
                after_total += path.stat().st_size
                skipped += 1
                print(f"[{index}/{len(clips)}] skip {path} ({target.name} exists)")
                continue
        else:
            target = path.with_name(path.stem + ".tmp.mp4")

        old_size = path.stat().st_size
        bitrate_text = f"{bitrate:.1f} Mbps" if bitrate is not None else "unknown bitrate"
        print(
            f"[{index}/{len(clips)}] {path} ({codec}, {bitrate_text}) "
            f"-> H.264 CRF {args.crf}"
        )

        if args.dry_run:
            after_total += old_size
            encoded += 1
            continue

        try:
            _encode(path, target, ffmpeg, args.crf, args.preset)
            new_size = target.stat().st_size
            if new_size <= 0 or new_size >= old_size:
                target.unlink(missing_ok=True)
                after_total += old_size
                skipped += 1
                print(
                    f"    kept original (new size {_mb(new_size):.1f} MB "
                    f">= {_mb(old_size):.1f} MB)"
                )
                continue
            if args.keep_original:
                after_total += old_size
                new_total += new_size
            else:
                os.replace(target, path)
                after_total += new_size
            encoded += 1
            print(
                f"    {_mb(old_size):.1f} MB -> {_mb(new_size):.1f} MB "
                f"({old_size / new_size:.1f}x smaller)"
            )
        except Exception as exc:  # noqa: BLE001 - continue with other clips
            target.unlink(missing_ok=True)
            after_total += old_size
            failed += 1
            print(f"    FAILED: {exc}", file=sys.stderr)

    if args.dry_run:
        print(f"\nDry run: {encoded} clip(s) would be encoded, {skipped} skipped.")
        return 0

    if args.keep_original:
        print(
            f"\nWrote {encoded} H.264 copy(ies), {_mb(new_total):.1f} MB "
            f"(skipped {skipped}, failed {failed}); originals kept."
        )
        return 1 if failed else 0

    saved = before_total - after_total
    ratio = before_total / after_total if after_total else 0.0
    print(
        f"\nEncoded {encoded}, skipped {skipped}, failed {failed} | "
        f"{_mb(before_total):.1f} MB -> {_mb(after_total):.1f} MB "
        f"(saved {_mb(saved):.1f} MB, {ratio:.1f}x)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
