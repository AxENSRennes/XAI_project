from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


FILES = {
    "audiocaps-train-counterfactual.json": "https://raw.githubusercontent.com/ali-vosoughi/counterfactual-audio/main/audiocaps-train-counterfactual.json",
    "clotho-development-counterfactual.json": "https://raw.githubusercontent.com/ali-vosoughi/counterfactual-audio/main/clotho-development-counterfactual.json",
    "clotho-validation-counterfactual.json": "https://raw.githubusercontent.com/ali-vosoughi/counterfactual-audio/main/clotho-validation-counterfactual.json",
    "macs-counterfactual.json": "https://raw.githubusercontent.com/ali-vosoughi/counterfactual-audio/main/macs-counterfactual.json",
    "prompts_1.json": "https://raw.githubusercontent.com/ali-vosoughi/counterfactual-audio/main/prompts_1.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        destination = output_dir / filename
        urllib.request.urlretrieve(url, destination)
        print(f"downloaded {filename} -> {destination}")


if __name__ == "__main__":
    main()

