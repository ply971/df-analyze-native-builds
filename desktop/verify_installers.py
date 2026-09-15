"""Install and exercise release artifacts on disposable GitHub-hosted runners."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = "df-analyze-desktop"


def run(args, **kwargs):
    print("+ " + " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, timeout=900, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "installer-artifacts")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("RUNNER_TEMP"):
        raise SystemExit("Run this installation test only on a disposable GitHub Actions runner.")
    artifacts = args.artifacts.resolve()
    checksums = sorted(artifacts.glob("*.sha256"))
    if not checksums:
        raise SystemExit("No installer checksum files were downloaded.")
    for checksum in checksums:
        artifact = checksum.with_suffix("")
        expected, name = checksum.read_text(encoding="utf-8").strip().split(maxsplit=1)
        if name != artifact.name:
            raise ValueError(f"Unexpected checksum filename: {name}")
        with artifact.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {artifact.name}")
        print(f"[install-test] SHA-256 OK: {artifact.name}", flush=True)

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    staging = Path(os.environ["RUNNER_TEMP"]) / "df-analyze installed test"
    staging.mkdir()
    sample = staging / "sample.json"
    shutil.copy2(ROOT / "data" / "small_classifier_data.json", sample)
    env = os.environ.copy()
    for variable in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "DF_ANALYZE_CACHE", "MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
        env.pop(variable, None)
    # Exercise default user cache paths with a normal, unprivileged app process.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[variable] = "1"

    def check(executable: Path, label: str) -> None:
        command = [executable, "--self-test", "--data", sample, "--target", "target", "--outdir", staging / label]
        if sys.platform == "linux":
            command = ["xvfb-run", "-a", "-s", "-screen 0 1440x1000x24", *command]
        run(command, cwd=staging, env=env)
        print(f"[install-test] PASSED: {label}", flush=True)

    if sys.platform == "win32":
        installer = artifacts / f"{APP}-{version}-windows-x86_64-setup.exe"
        installed = staging / "Installed application"
        run([installer, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", f"/DIR={installed}", f"/LOG={ROOT / 'installer.log'}"])
        check(installed / f"{APP}.exe", "windows-setup")
        run([installed / "unins000.exe", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"])
        if (installed / f"{APP}.exe").exists():
            raise RuntimeError("Windows uninstaller left the application executable behind.")
        print("[install-test] Windows uninstall OK", flush=True)
    elif sys.platform == "darwin":
        installer = artifacts / f"{APP}-{version}-macos-arm64.dmg"
        mount = staging / "Mounted disk"
        installed = Path("/Applications/df-analyze Desktop.app")
        if installed.exists():
            raise RuntimeError("Refusing to overwrite an existing Mac installation.")
        run(["hdiutil", "attach", installer, "-readonly", "-nobrowse", "-mountpoint", mount])
        try:
            run(["sudo", "ditto", mount / installed.name, installed])
        finally:
            run(["hdiutil", "detach", mount])
        run(["codesign", "--verify", "--deep", "--strict", installed])
        check(installed / "Contents" / "MacOS" / APP, "macos-dmg")
    elif sys.platform == "linux":
        installer = artifacts / f"{APP}-{version}-linux-amd64.deb"
        run(["sudo", "apt-get", "update"])
        run(["sudo", "apt-get", "install", "-y", installer, "xvfb", "xauth", "libgl1-mesa-dri"])
        desktop_entry = Path(f"/usr/share/applications/{APP}.desktop")
        if not desktop_entry.is_file():
            raise RuntimeError("The Linux application menu entry was not installed.")
        check(Path("/opt") / APP / APP, "linux-deb")
        portable = artifacts / f"{APP}-{version}-linux-x86_64.tar.gz"
        with tarfile.open(portable) as archive:
            archive.extractall(staging / "Portable app", filter="data")
        check(staging / "Portable app" / APP / APP, "linux-portable")
        run(["sudo", "apt-get", "remove", "-y", APP])
        if desktop_entry.exists() or (Path("/opt") / APP / APP).exists():
            raise RuntimeError("Debian uninstall left the app or menu entry behind.")
        print("[install-test] Linux uninstall OK", flush=True)
    else:
        raise SystemExit(f"Unsupported runner: {sys.platform}")


if __name__ == "__main__":
    main()
