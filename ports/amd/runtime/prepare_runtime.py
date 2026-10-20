"""Adapt only actual Engine profile identities to native HIP name/gfx facts."""
from __future__ import annotations

import argparse
from pathlib import Path
import re


def transform_model_instance(text: str) -> str:
    pattern = re.compile(
        r"(?<![\w:])context_cost_hardware_class\(\s*"
        r"(?P<props>(?:device\.)?props)\.name\s*,\s*"
        r"(?P=props)\.major\s*,\s*(?P=props)\.minor\s*\)"
    )
    text, count = pattern.subn(
        lambda match: "::ninfer::amd::context_cost_hardware_class("
        + match["props"] + ".name, " + match["props"] + ".gcnArchName)", text
    )
    if count != 3:
        raise ValueError(f"expected three original Engine hardware-class consumers, found {count}")
    include = '#include "runtime/amd_hardware_class.h"\n'
    if include not in text:
        text = include + text
    return text


def write_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_view.resolve(strict=True)
    output = args.output.resolve()
    relative = Path("src/runtime/engine/model_instance.cpp")
    rewritten = transform_model_instance((source / relative).read_text(encoding="utf-8"))
    write_changed(output / relative, rewritten)
    write_changed(output / "src/runtime/amd_hardware_class.h",
                  Path(__file__).with_name("amd_hardware_class.h").read_text(encoding="utf-8"))
    print("Native AMD runtime hardware identity: three actual Engine consumers")


if __name__ == "__main__":
    main()
