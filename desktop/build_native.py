"""Build and smoke-test native Linux and Apple Silicon desktop installers.

Run with Python 3.13 on the target OS; uv must be on PATH. The build environment
is separate from the developer's .venv and the Windows distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = "df-analyze-desktop"
MAC_APP = "df-analyze Desktop.app"
PYTHON_VERSION = "3.13.11"


def run(args: list[str | Path], *, cwd: Path = ROOT, env=None, timeout=3600) -> None:
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True, timeout=timeout)


def native_target() -> str:
    machine = platform.machine().lower()
    if sys.platform == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if sys.platform == "darwin" and machine == "arm64":
        return "macos-arm64"
    raise SystemExit(
        "Build on Linux x86-64 or an Apple Silicon Mac. "
        "Windows cannot cross-compile these installers; Intel Macs are not supported "
        "by the project's current Dear PyGui and PyTorch versions."
    )


def write_requirements(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    # The desktop runs its models through PyTorch. It never imports the web UI
    # or TensorFlow; avoid installing those independent frameworks in its build.
    omitted = {"tensorflow", "streamlit"}
    requirements = []
    for dependency in project["project"]["dependencies"]:
        name = re.match(r"[\w.-]+", dependency).group().lower().replace("_", "-")
        if name not in omitted:
            requirements.append(dependency)
    for name in ("pyinstaller", "pyinstaller-hooks-contrib"):
        package = next(p for p in lock["package"] if p["name"] == name)
        requirements.append(f"{name}=={package['version']}")
    requirements_path = directory / "requirements.txt"
    requirements_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    constraints = directory / "constraints.txt"
    run([
        "uv", "export", "--locked", "--no-dev", "--no-emit-project", "--no-hashes",
        "--format", "requirements.txt", "--output-file", constraints, "--quiet",
    ])
    return requirements_path, constraints


def prepare_environment(target: str, work: Path) -> Path:
    requirements, constraints = write_requirements(work)
    venv = work / "venv"
    python = venv / "bin" / "python"
    if not python.exists():
        run(["uv", "venv", "--python", PYTHON_VERSION, venv])
    args = [
        "uv", "pip", "install", "--python", str(python),
        "--only-binary", "torch,torchvision,torchaudio,dearpygui,numpy,scipy,"
        "scikit-learn,pandas,pyarrow,llvmlite,numba,catboost,lightgbm",
        "-r", str(requirements), "-c", str(constraints),
    ]
    if target.startswith("linux"):
        # CUDA is not required for desktop analysis. This also avoids downloading
        # several GB of NVIDIA runtimes on the Linux build machine.
        args += ["--torch-backend", "cpu"]
    run(args)
    return python


def smoke_test(bundle: Path, target: str) -> None:
    # Relocate outside the checkout, including a space in the installation path,
    # so source files, the working directory and the build venv cannot hide missing
    # bundle contents. Test as an ordinary user, not through sudo.
    with tempfile.TemporaryDirectory(prefix="df-analyze smoke ") as temporary:
        staging = Path(temporary)
        installed = staging / bundle.name
        if target.startswith("macos"):
            run(["ditto", bundle, installed])
            executable = installed / "Contents" / "MacOS" / APP
        else:
            shutil.copytree(bundle, installed, symlinks=True)
            executable = installed / APP
        sample = staging / "sample.json"
        shutil.copy2(ROOT / "data" / "small_classifier_data.json", sample)
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        env["DF_ANALYZE_CACHE"] = str(staging / "cache")
        env["MPLCONFIGDIR"] = str(staging / "matplotlib")
        env["NUMBA_CACHE_DIR"] = str(staging / "numba")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[key] = "1"
        command = [
            str(executable), "--self-test", "--data", str(sample),
            "--target", "target", "--outdir", str(staging / "analysis"),
        ]
        if target.startswith("linux") and not env.get("DISPLAY"):
            command = ["xvfb-run", "-a", "-s", "-screen 0 1440x1000x24", *command]
        run(command, cwd=staging, env=env, timeout=900)


def build_deb(bundle: Path, output: Path, version: str) -> Path:
    artifact = output / f"{APP}-{version}-linux-amd64.deb"
    with tempfile.TemporaryDirectory(prefix="df-analyze-deb-") as temporary:
        staging = Path(temporary)
        shutil.copytree(bundle, staging / "opt" / APP, symlinks=True)
        applications = staging / "usr" / "share" / "applications"
        applications.mkdir(parents=True)
        (applications / f"{APP}.desktop").write_text(
            "[Desktop Entry]\nType=Application\nName=df-analyze Desktop\n"
            "Comment=Explore datasets and run machine learning experiments\n"
            f"Exec=/opt/{APP}/{APP}\nIcon=applications-science\n"
            "Terminal=false\nCategories=Science;Education;\n", encoding="utf-8",
        )
        size = sum(p.stat().st_size for p in staging.rglob("*")
                   if p.is_file() and not p.is_symlink()) // 1024
        control = staging / "DEBIAN"
        control.mkdir()
        (control / "control").write_text(
            f"Package: {APP}\nVersion: {version}\nArchitecture: amd64\n"
            "Maintainer: df-analyze maintainers\nSection: science\nPriority: optional\n"
            f"Installed-Size: {size}\n"
            "Depends: libc6 (>= 2.35), libstdc++6, libgl1, libx11-6, libxext6, "
            "libxrandr2, libxinerama1, libxcursor1, libxi6, libgomp1\n"
            "Description: Desktop interface for df-analyze\n"
            " Explore tabular datasets and run machine learning experiments.\n",
            encoding="utf-8",
        )
        run(["dpkg-deb", "--root-owner-group", "-Zxz", "--build", staging, artifact])
    run(["dpkg-deb", "--info", artifact])
    return artifact


def build_dmg(bundle: Path, output: Path, version: str) -> Path:
    artifact = output / f"{APP}-{version}-macos-arm64.dmg"
    run(["codesign", "--verify", "--deep", "--strict", bundle])
    with tempfile.TemporaryDirectory(prefix="df-analyze-dmg-") as temporary:
        staging = Path(temporary)
        run(["ditto", bundle, staging / MAC_APP])
        (staging / "Applications").symlink_to("/Applications", target_is_directory=True)
        (staging / "INSTALL.txt").write_text(
            "Drag df-analyze Desktop.app to Applications.\n"
            "Requires macOS 14 or newer on Apple Silicon (M-series).\n"
            "This build is not Apple-notarized. macOS may require approval in\n"
            "System Settings > Privacy & Security the first time you open it.\n",
            encoding="utf-8",
        )
        run([
            "hdiutil", "create", "-volname", "df-analyze Desktop", "-srcfolder",
            staging, "-ov", "-format", "UDZO", artifact,
        ])
    run(["hdiutil", "verify", artifact])
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements-only", type=Path,
                        help="Export build inputs without installing or building anything")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only install the isolated native build environment")
    args = parser.parse_args()
    if args.requirements_only is not None:
        write_requirements(args.requirements_only.resolve())
        return
    target = native_target()
    required_tools = ["uv", "dpkg-deb", "xvfb-run"] if target.startswith("linux") else [
        "uv", "ditto", "codesign", "hdiutil",
    ]
    missing = [tool for tool in required_tools if not shutil.which(tool)]
    if missing:
        raise SystemExit("Install required build tools first: " + ", ".join(missing))
    work = ROOT / "build" / "native" / target
    python = prepare_environment(target, work)
    if args.prepare_only:
        return
    dist = ROOT / "dist" / "native" / target
    run([
        python, "-m", "PyInstaller", ROOT / "desktop" / "build.spec", "--noconfirm",
        "--clean", "--distpath", dist, "--workpath", work / "pyinstaller",
    ])
    bundle = dist / (MAC_APP if target.startswith("macos") else APP)
    smoke_test(bundle, target)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    output = ROOT / "dist" / "installer" / "native"
    output.mkdir(parents=True, exist_ok=True)
    if target.startswith("macos"):
        artifacts = [build_dmg(bundle, output, version)]
    else:
        artifacts = [build_deb(bundle, output, version)]
        portable = output / f"{APP}-{version}-linux-x86_64.tar.gz"
        with tarfile.open(portable, "w:gz") as archive:
            archive.add(bundle, arcname=APP)
        artifacts.append(portable)
    for artifact in artifacts:
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifact.with_name(artifact.name + ".sha256").write_text(
            f"{digest}  {artifact.name}\n", encoding="utf-8",
        )
        print(f"Installer ready: {artifact}", flush=True)


if __name__ == "__main__":
    main()
