#!/usr/bin/env python3
from pathlib import Path
import re
import subprocess
import sys

OUT = Path("kirsch_comparison_memory_ledger.txt")
PNG = Path("kirsch_comparison_placeholder.png")


def run_test_binary() -> str:
    binary = Path("./test_memory_ledger")
    if not binary.exists():
        raise SystemExit("test_memory_ledger 不存在，请先运行 make test_circle 或 make test")
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    return result.stdout


def summarize(text: str) -> str:
    matches = re.findall(r"网格尺寸: (\d+x\d+x\d+).*?Total resident \(est\.\)\s+(\d+) bytes", text, re.S)
    lines = ["memory-ledger summary"]
    if matches:
        for grid, resident in matches:
            lines.append(f"{grid}: resident_bytes={resident}")
    else:
        lines.append("no parsable ledger entries found")
    return "\n".join(lines) + "\n"


def write_placeholder_png() -> None:
    # 1x1 transparent PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    PNG.write_bytes(png_bytes)


if __name__ == "__main__":
    try:
        output = run_test_binary()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stdout)
        sys.stderr.write(exc.stderr)
        raise
    OUT.write_text(summarize(output), encoding="utf-8")
    write_placeholder_png()
    print(f"wrote {OUT}")
    print(f"wrote {PNG}")
