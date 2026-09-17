# df-analyze Desktop installers

Native build repository for the Windows, Linux and Apple Silicon desktop installers.

[Download the latest release](https://github.com/ply971/df-analyze-native-builds/releases/latest)

## Downloads

**Published release:** [df-analyze Desktop 4.1.0](https://github.com/ply971/df-analyze-native-builds/releases/tag/v4.1.0).
All three passed [independent installation checks on fresh runners](https://github.com/ply971/df-analyze-native-builds/actions/runs/34999228336)
on September 15, 2026. See [installation instructions](INSTALL.md).

Download your installer from the release's **Assets** section. The release also
includes SHA-256 checksums, installation instructions, and the build report.
Release assets are not subject to the 14-day Actions artifact retention period.

| Platform | Installer | Requirements |
| --- | --- | --- |
| Windows | [.exe setup wizard](https://github.com/ply971/df-analyze-native-builds/releases/download/v4.1.0/df-analyze-desktop-4.1.0-windows-x86_64-setup.exe) | x86-64 |
| Linux | [.deb installer](https://github.com/ply971/df-analyze-native-builds/releases/download/v4.1.0/df-analyze-desktop-4.1.0-linux-amd64.deb) | Ubuntu 22.04 or newer, x86-64 |
| Mac | [.dmg disk image](https://github.com/ply971/df-analyze-native-builds/releases/download/v4.1.0/df-analyze-desktop-4.1.0-macos-arm64.dmg) | macOS 14 or newer, Apple Silicon |

The additional Linux portable archive is available in the build's Actions
artifacts, which retain their separate 14-day expiry.

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

Before uploading the installers, the build workflow installs each setup and
repeats those checks. It also tests the Linux portable archive and Windows/Linux
uninstallers. Analysis runs as an ordinary user, including the `/opt` Linux install.

The separate **Install and test finished setup files** workflow accepts a build
run ID. It downloads the finished artifacts onto fresh native runners, checks
their hashes, installs them, and repeats the app checks. It also verifies the
Linux portable archive and Windows/Linux uninstallers.

See [desktop build instructions](desktop/README.md) for local builds and requirements.
These automated checks do not certify every OS version, graphics driver or dataset.

## Source

Based on [stfxecutables/df-analyze](https://github.com/stfxecutables/df-analyze).
The original project documentation is preserved in [UPSTREAM_README.md](UPSTREAM_README.md).
