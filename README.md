# df-analyze Desktop installers

Private build repository for the Windows, Linux and Apple Silicon desktop installers.

[View native builds and downloads](https://github.com/ply971/df-analyze-native-builds/actions/workflows/native-installers.yml)

## Downloads

Open a successful workflow run and download its installer artifacts. Each artifact
contains an installer and its SHA-256 checksum. Actions downloads are retained for
14 days. No release is published automatically.

| Artifact | Installer | Platform |
| --- | --- | --- |
| `df-analyze-windows-x86_64` | `.exe` setup wizard | 64-bit Windows |
| `df-analyze-linux-x86_64` | `.deb` and portable `.tar.gz` | Ubuntu 22.04 or newer, x86-64 |
| `df-analyze-macos-arm64` | `.dmg` | macOS 14 or newer, Apple Silicon |

Intel Macs and other Linux distributions require separate compatibility testing.

## Install

- **Windows:** open the setup executable and follow the wizard.
- **Linux:** install the DEB using `sudo apt install ./df-analyze-desktop-4.1.0-linux-amd64.deb`,
  or extract the portable archive and run `df-analyze-desktop/df-analyze-desktop`.
- **Mac:** open the DMG and drag **df-analyze Desktop.app** to **Applications**.
  The app is ad-hoc signed, not Apple-notarized. macOS may require first-launch
  approval in **System Settings > Privacy & Security**.

Python and the notebook kernel are bundled. Embedding model weights are downloaded
when needed; an internet connection is required for their initial download.

## Verification

The workflow builds on each native OS and relocates the app outside the checkout
before testing. Installers are created only after the packaged app passes:

- launch and chart rendering;
- CSV, Parquet, Excel and JSON import/export;
- small sklearn, CatBoost, LightGBM, Numba, PyTorch and text/image computations;
- a real analysis with saved performance tables;
- notebook execution, persistent variables, inline plotting and export.

See [desktop build instructions](desktop/README.md) for local builds and requirements.
These automated checks do not certify every OS version, graphics driver or dataset.

## Source

Based on [stfxecutables/df-analyze](https://github.com/stfxecutables/df-analyze).
The original project documentation is preserved in [UPSTREAM_README.md](UPSTREAM_README.md).
