# Desktop installer verification

Version **4.1.0**, app source commit `6909bb15e03f7350aa0cf84fc727ef1ed4392ae3`.
Verified September 15, 2026.

| Native runner | Installer | Build and installed app | Independent fresh-runner test |
| --- | --- | --- | --- |
| Windows Server 2022, x86-64 | Inno Setup `.exe` | Passed | Passed |
| Ubuntu 22.04, x86-64 | `.deb` and portable `.tar.gz` | Passed | Passed |
| macOS 14, Apple Silicon | `.dmg` | Passed | Passed |

- [Build, relocated-app tests, packaging and installation tests](https://github.com/ply971/df-analyze-native-builds/actions/runs/34997497438)
- [Independent download and installation tests on fresh runners](https://github.com/ply971/df-analyze-native-builds/actions/runs/34999228336)

## Checks performed

- Installer SHA-256 hashes matched before installation.
- The native GUI opened and rendered its charts.
- CSV, Parquet, Excel and JSON data import/export succeeded.
- Small sklearn, CatBoost, LightGBM, Numba, PyTorch and text/image model calculations ran.
- A real analysis produced saved performance tables.
- The bundled notebook kernel executed cells, retained variables, rendered an
  inline plot and exported an `.ipynb` file.
- The installed Linux app ran from `/opt` as an ordinary user. The portable
  archive was extracted into a separate folder and passed the same app checks.
- Windows and Linux uninstallers removed the app; the Linux menu entry was removed.
- macOS code-signature and disk-image integrity checks passed.

Python and notebook dependencies are bundled. No separate Python installation
is required by the app. Initial pretrained embedding-weight downloads need internet.

## Limits

These are checks of the stated runners and cases, not a guarantee for every OS
version, graphics driver, hardware configuration or dataset. The macOS package
targets macOS 14+ on Apple Silicon; Intel Macs are not included. Other Linux
distributions and Linux ARM need separate compatibility work.

The Mac app is ad-hoc signed, not Apple-notarized. The Windows installer is
unsigned. Warning-free public distribution still requires the appropriate
developer signing credentials and Apple notarization.
