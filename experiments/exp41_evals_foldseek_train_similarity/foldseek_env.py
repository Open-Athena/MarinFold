# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Locate (or install) the Foldseek binary and run it.

Foldseek is a standalone C++ binary, not a Python package, so the rest
of this experiment shells out to it. This module is the single place
that knows how to find it:

1. ``$FOLDSEEK_BIN`` — an explicit path the user set.
2. ``foldseek`` already on ``$PATH`` (e.g. a conda install).
3. A static binary cached under ``$MARINFOLD_FOLDSEEK_DIR`` (default
   ``~/.cache/marinfold/foldseek``), downloaded on first use from
   https://mmseqs.com/foldseek.

``ensure_foldseek()`` returns an absolute path, installing into the
cache only if nothing is found and ``auto_install`` is left on. Set
``MARINFOLD_FOLDSEEK_NO_INSTALL=1`` to forbid the download (e.g. in CI),
in which case a missing binary raises rather than reaching out to the
network.

The static-binary selection follows Foldseek's own quick-start
(https://github.com/steineggerlab/foldseek#quick-start): AVX2 vs SSE2
on x86-64 Linux, an ARM64 build, and a universal macOS build.
"""

import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

_DOWNLOAD_BASE = "https://mmseqs.com/foldseek"


def _cache_dir() -> Path:
    override = os.environ.get("MARINFOLD_FOLDSEEK_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "marinfold" / "foldseek"


def _static_tarball_name() -> str:
    """Pick the right prebuilt Foldseek tarball for this machine."""
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        return "foldseek-osx-universal.tar.gz"
    if system != "Linux":
        raise RuntimeError(
            f"No known Foldseek static build for platform {system!r}; "
            f"install foldseek manually and set $FOLDSEEK_BIN."
        )
    if machine in ("aarch64", "arm64"):
        return "foldseek-linux-arm64.tar.gz"
    if machine in ("x86_64", "amd64"):
        # AVX2 builds are much faster; fall back to SSE2 on older CPUs.
        try:
            flags = Path("/proc/cpuinfo").read_text()
        except OSError:
            flags = ""
        return "foldseek-linux-avx2.tar.gz" if "avx2" in flags else "foldseek-linux-sse2.tar.gz"
    raise RuntimeError(
        f"No known Foldseek static build for machine {machine!r}; "
        f"install foldseek manually and set $FOLDSEEK_BIN."
    )


def _cached_binary() -> Path:
    """Path where the static binary lands after extraction (may not exist)."""
    # The tarball extracts a top-level ``foldseek/`` dir with bin/foldseek.
    return _cache_dir() / "foldseek" / "bin" / "foldseek"


def install_foldseek() -> Path:
    """Download + extract the static Foldseek binary into the cache.

    Returns the path to the extracted ``foldseek`` executable. Idempotent:
    if the cached binary already exists it is returned without re-downloading.
    """
    binary = _cached_binary()
    if binary.exists():
        return binary

    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    tarball_name = _static_tarball_name()
    url = f"{_DOWNLOAD_BASE}/{tarball_name}"
    tar_path = cache / tarball_name

    print(f"[foldseek_env] downloading {url}")
    urllib.request.urlretrieve(url, tar_path)
    print(f"[foldseek_env] extracting {tar_path.name} into {cache}")
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(cache, filter="data")  # safe-extraction filter (py>=3.11.4)
        except TypeError:
            tf.extractall(cache)
    tar_path.unlink(missing_ok=True)

    if not binary.exists():
        raise RuntimeError(
            f"extracted {tarball_name} but {binary} is missing; "
            f"the Foldseek tarball layout may have changed."
        )
    binary.chmod(0o755)
    return binary


def ensure_foldseek(auto_install: bool = True) -> str:
    """Return an absolute path to a usable Foldseek binary.

    Resolution order: ``$FOLDSEEK_BIN`` → ``$PATH`` → cached static binary
    → (optionally) download a static binary. Raises if nothing is found and
    installation is disabled.
    """
    env_bin = os.environ.get("FOLDSEEK_BIN")
    if env_bin:
        p = Path(env_bin).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"$FOLDSEEK_BIN={env_bin!r} does not exist")
        return str(p.resolve())

    on_path = shutil.which("foldseek")
    if on_path:
        return on_path

    cached = _cached_binary()
    if cached.exists():
        return str(cached.resolve())

    if not auto_install or os.environ.get("MARINFOLD_FOLDSEEK_NO_INSTALL"):
        raise FileNotFoundError(
            "foldseek not found on $PATH, in $FOLDSEEK_BIN, or in the cache, "
            "and auto-install is disabled. Install it (conda install -c "
            "bioconda foldseek) or unset MARINFOLD_FOLDSEEK_NO_INSTALL."
        )
    return str(install_foldseek().resolve())


def run_foldseek(
    args: list[str], *, auto_install: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    """Run ``foldseek <args>``, raising on a non-zero exit.

    Extra ``kwargs`` pass through to ``subprocess.run`` (e.g. ``cwd``,
    ``capture_output``, ``text``). Exceptions propagate by design.
    """
    binary = ensure_foldseek(auto_install=auto_install)
    kwargs.setdefault("check", True)
    return subprocess.run([binary, *args], **kwargs)


def foldseek_version(auto_install: bool = True) -> str:
    """Return the installed Foldseek version string (e.g. a commit hash)."""
    proc = run_foldseek(
        ["version"], auto_install=auto_install, capture_output=True, text=True
    )
    return proc.stdout.strip()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "command",
        choices=["which", "install", "version"],
        help="which: print resolved path; install: force a static download; version: print version",
    )
    args = ap.parse_args()

    if args.command == "install":
        print(install_foldseek())
    elif args.command == "which":
        print(ensure_foldseek())
    else:
        print(foldseek_version())


if __name__ == "__main__":
    main()
