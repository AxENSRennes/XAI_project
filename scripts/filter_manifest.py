from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.manifests import read_manifest, write_jsonl


def can_open_audio(path: str) -> bool:
    try:
        with sf.SoundFile(path, "r") as _:
            return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-audio", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.input)
    filtered = []
    missing = 0
    unreadable = 0

    for row in rows:
        audio_path = row.get("audio_path") or row.get("path")
        if not audio_path or not Path(audio_path).exists():
            missing += 1
            continue
        if args.check_audio and not can_open_audio(audio_path):
            unreadable += 1
            continue
        filtered.append(row)

    write_jsonl(args.output, filtered)
    print(
        f"kept={len(filtered)} missing={missing} unreadable={unreadable} output={args.output}"
    )


if __name__ == "__main__":
    main()
