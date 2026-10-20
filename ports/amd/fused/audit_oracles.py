"""Compare all retained original fusion test sources against the common HIP view.

Only GDN gating's hardware-residency test namespace may differ. Its primal output
oracle and every numerical tolerance, boundary, guard and workspace test remain
byte-identical to the original common HIP source after removing the overlay mark.
"""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("--base-view", required=True, type=Path)
parser.add_argument("--fused-view", required=True, type=Path)
parser.add_argument("--report", required=True, type=Path)
args = parser.parse_args()
base, fused = args.base_view.resolve(), args.fused_view.resolve()
marker = "// Native gfx1201 fusion physical-policy overlay.\n"
results = []
for source in sorted((fused / "tests").rglob("*")):
    if not source.is_file() or source.suffix not in (".h", ".cuh", ".cpp", ".cu"):
        continue
    relative = source.relative_to(fused).as_posix()
    actual = source.read_text(encoding="utf-8").removeprefix(marker)
    expected = (base / relative).read_text(encoding="utf-8")
    physical = relative == "tests/ops/test_gdn_gating_proj.cpp"
    if physical:
        actual = actual.replace('#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"\n', "")
        # The physical policy namespace contains no output-value reference or
        # numerical acceptance condition. Preserve the entire rest verbatim.
        pattern = r"namespace sm_count_routes \{.*?\} // namespace sm_count_routes"
        actual, count_a = re.subn(pattern, "", actual, flags=re.DOTALL)
        expected, count_e = re.subn(pattern, "", expected, flags=re.DOTALL)
        if (count_a, count_e) != (1, 1): raise ValueError("GDN physical-policy boundary changed")
    results.append({"source": relative, "unchanged_numerical_oracle": actual == expected,
                    "native_hardware_policy_namespace": physical})
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
failures = [item["source"] for item in results if not item["unchanged_numerical_oracle"]]
print(json.dumps({"test_sources": len(results), "numerical_oracle_differences": failures,
                  "hardware_policy_changes": [item["source"] for item in results if item["native_hardware_policy_namespace"]]}))
raise SystemExit(bool(failures) or not results)
