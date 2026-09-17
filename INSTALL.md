# df-analyze Desktop 4.1.0

## Choose your setup

| Platform | Setup file | Installation |
| --- | --- | --- |
| Windows x86-64 | `df-analyze-desktop-4.1.0-windows-x86_64-setup.exe` | Open the setup wizard. It installs for your user account. |
| Ubuntu 22.04 or newer, x86-64 | `df-analyze-desktop-4.1.0-linux-amd64.deb` | Run the command below, then open **df-analyze Desktop** from the applications menu. |
| macOS 14 or newer, Apple Silicon | `df-analyze-desktop-4.1.0-macos-arm64.dmg` | Open the disk image and drag **df-analyze Desktop.app** to **Applications**. |

For Ubuntu, open a terminal in this folder and run:

```bash
sudo apt install ./df-analyze-desktop-4.1.0-linux-amd64.deb
```

Python, the analysis libraries, and the notebook kernel are included. Embedding
model weights require an internet connection for their initial download.

## Compatibility and signing

- The Mac installer supports M-series Macs. Intel Macs are not included.
- Other Linux distributions and ARM Linux require separate builds or testing.
- The Mac app is ad-hoc signed and is not Apple-notarized; first-launch approval
  may be required in macOS Privacy & Security settings.
- The Windows setup is unsigned and may display a Windows security prompt.
- Each installer has a matching `.sha256` file for integrity verification.

## Source and verification

- [Published installers](https://github.com/ply971/df-analyze-native-builds/releases/tag/v4.1.0)
- [Source repository](https://github.com/ply971/df-analyze-native-builds)
- [Native build and installation checks](https://github.com/ply971/df-analyze-native-builds/actions/runs/34997497438)
- [Independent installation checks on fresh runners](https://github.com/ply971/df-analyze-native-builds/actions/runs/34999228336)
- App source commit: `6909bb15e03f7350aa0cf84fc727ef1ed4392ae3`.

All three platforms passed both workflows on September 15, 2026.

The workflow tests chart rendering, data import/export, model calculations, a
real analysis, and notebook execution, plotting, and export. It also installs
the finished setups and checks Windows/Linux uninstallers. The Linux portable
archive is available in the workflow's Linux artifact and has its own test.

Release downloads are not subject to the 14-day Actions artifact expiry.
The local setup files are independent copies. These checks cover the listed targets and test cases;
they do not certify every hardware configuration or dataset.
