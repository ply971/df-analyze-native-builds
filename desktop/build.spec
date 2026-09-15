# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for the df-analyze desktop GUI.
#
# Build with (from the repo root, inside the project venv):
#   .venv\Scripts\pyinstaller.exe desktop\build.spec --noconfirm
#
# See desktop/README.md for the full story. Short version: df-analyze's
# dependency stack (torch, transformers, catboost, lightgbm, sklearn, numba,
# pytorch_tabular, ...) is ~1.6GB, and several of these packages are known to
# need explicit help from PyInstaller's static import analysis (native
# shared libraries, dynamic/lazy imports, importlib.metadata probing).
# RISKY_COLLECT_ALL / RISKY_COPY_METADATA below are a well-evidenced starting
# point, not a guaranteed-final list -- if a fresh build hits a new
# ModuleNotFoundError or importlib.metadata.PackageNotFoundError, that's the
# normal iterate-and-extend loop, not a sign something is fundamentally
# wrong. Windows keeps its diagnostic console; macOS uses a windowed app
# bundle. Every native installer must pass the relocated self-test.
import os
import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).resolve().parent
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

RISKY_COLLECT_ALL = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "tokenizers",
    "catboost",
    "lightgbm",
    "numba",
    "llvmlite",
    "sklearn",
    "scipy",
    "pytorch_lightning",
    "lightning",
    "lightning_fabric",
    "pytorch_tabular",
    "optuna",
    "dearpygui",
    "ipykernel",
    "jupyter_client",
    "IPython",
    "matplotlib_inline",
]
RISKY_COPY_METADATA = [
    *RISKY_COLLECT_ALL,
    "tqdm",
    "filelock",
    "regex",
    "requests",
    "packaging",
    "numpy",
    "pyyaml",
]

datas = []
binaries = []
hiddenimports = []
# Keep the application's complete Python package tree available. Some pipeline
# imports use the legacy src.df_analyze name, which can otherwise cause module
# graph analysis to omit siblings needed by the GUI and bundled kernel.
for source_root in (ROOT / "src", ROOT):
    package = source_root / ("df_analyze" if source_root.name == "src" else "desktop")
    for source_file in sorted(package.rglob("*.py")):
        relative = source_file.relative_to(source_root)
        datas.append((str(source_file), str(relative.parent)))
        module = ".".join(relative.with_suffix("").parts)
        hiddenimports.append(module.removesuffix(".__init__"))

for pkg in RISKY_COLLECT_ALL:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

for pkg in RISKY_COPY_METADATA:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass  # not every entry above resolves as an installed distribution name

a = Analysis(
    [str(ROOT / "desktop" / "entry.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "desktop" / "frozen_runtime.py")],
    # These frameworks are not used by the desktop application. In particular,
    # transformers probes TensorFlow even though our embedding engine uses torch.
    excludes=["tensorflow", "keras", "streamlit"],
    noarchive=False,
    optimize=0,
)
print("Application GUI modules in archive:", [entry for entry in a.pure if "df_analyze.gui" in entry[0]])
print("Application GUI source files:", [entry for entry in a.datas if "df_analyze/gui" in entry[0].replace("\\", "/")])
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [("u", None, "OPTION")],
    exclude_binaries=True,
    name="df-analyze-desktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=sys.platform != "darwin",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=os.environ.get("DF_ANALYZE_CODESIGN_IDENTITY"),
    entitlements_file=os.environ.get("DF_ANALYZE_ENTITLEMENTS_FILE"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="df-analyze-desktop",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="df-analyze Desktop.app",
        bundle_identifier="org.df-analyze.desktop",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "df-analyze Desktop",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "LSMinimumSystemVersion": "14.0",
        },
    )
