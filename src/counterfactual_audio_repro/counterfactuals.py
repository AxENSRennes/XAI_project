from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

PROMPT_SCAFFOLD = [
    "Audio signal is a kind of wave that is directional from the source to microphone. Then, microphone records the sound. Each sound may have multiple sound sources. Humans detect these sounds and describe it based on its sources. What are the sound source or sources in the given caption?",
    "Alter these sources with something very different and generate a caption of the same length and in one line and without itemization.",
]


REPLACEMENTS = {
    "dog": "car engine",
    "dogs": "car engines",
    "cat": "siren",
    "cats": "sirens",
    "bird": "jackhammer",
    "birds": "jackhammers",
    "people": "cars",
    "person": "truck",
    "crowd": "line",
    "man": "train",
    "woman": "motorcycle",
    "child": "bus",
    "children": "trucks",
    "adult": "vehicle",
    "adults": "cars",
    "talking": "honking",
    "speaking": "revving",
    "walking": "passing",
    "footsteps": "rain",
    "clapping": "chirping",
    "car": "dogs",
    "cars": "dogs",
    "train": "airplane",
    "horn": "alarm",
    "gun": "piano",
    "gunshots": "fireworks",
    "firework": "gunshot",
    "fireworks": "gunshots",
    "indoors": "outdoors",
    "indoor": "outdoor",
    "street": "forest",
    "engine": "dog",
    "barking": "revving",
    "bark": "engine revving",
    "laughing": "drilling",
    "music": "siren",
    "speech": "traffic",
    "talk": "honk",
    "voices": "engines",
    "voice": "engine",
    "rain": "applause",
    "wind": "machinery",
}


@dataclass
class CounterfactualExample:
    caption: str
    counterfactual_caption: str
    sources: list[str]
    replacements: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "caption": self.caption,
                "counterfactual_caption": self.counterfactual_caption,
                "sources": self.sources,
                "replacements": self.replacements,
            },
            ensure_ascii=True,
        )


def _rewrite_token(text: str, source: str, target: str) -> str:
    pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.isupper():
            return target.upper()
        if token[0].isupper():
            return target.capitalize()
        return target

    return pattern.sub(replace, text)


def generate_counterfactual(caption: str) -> CounterfactualExample:
    sources: list[str] = []
    applied: dict[str, str] = {}
    rewritten = caption.strip()
    lowered = rewritten.lower()

    for source, target in REPLACEMENTS.items():
        if re.search(rf"\b{re.escape(source)}\b", lowered):
            rewritten = _rewrite_token(rewritten, source, target)
            sources.append(source)
            applied[source] = target
            lowered = rewritten.lower()
            if len(applied) >= 4:
                break

    if not applied:
        rewritten = f"An unrelated mechanical scene replaces: {caption.strip()}"
        sources.append("implicit sound source")
        applied["implicit sound source"] = "mechanical scene"

    return CounterfactualExample(
        caption=caption.strip(),
        counterfactual_caption=rewritten,
        sources=sources,
        replacements=applied,
    )


def generate_from_lines(lines: list[str]) -> list[CounterfactualExample]:
    return [generate_counterfactual(line) for line in lines if line.strip()]


def load_text_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()

