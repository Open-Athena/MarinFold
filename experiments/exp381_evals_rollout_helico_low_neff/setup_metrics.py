"""Build the official TMscore executable at the frozen source revision."""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISION = "23f14ffcf8647f2dc1c0e01043db3a7f3a661c28"


def main() -> None:
    """Build and record source/compiler/binary identities for structural scoring."""
    source = HERE / "scratch/USalign"
    if not source.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/pylelab/USalign.git", str(source)],
            check=True,
        )
    subprocess.run(["git", "checkout", "--detach", REVISION], cwd=source, check=True)
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source
    ):
        raise ValueError("TMscore source has local changes")
    binary = HERE / "scratch/TMscore"
    subprocess.run(
        ["g++", "-O3", "-o", str(binary), str(source / "TMscore.cpp")], check=True
    )
    manifest = dict(
        repository="https://github.com/pylelab/USalign",
        revision=REVISION,
        compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[
            0
        ],
        flags=["-O3"],
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.iterdir())
            if p.suffix in {".h", ".cpp"}
        },
    )
    (HERE / "data/tmscore_build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
