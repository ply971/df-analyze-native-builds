# df-analyze desktop

Run from the repository root with the project environment:

```powershell
.venv\Scripts\python.exe desktop/app.py
```

Alternatively, `uv run python desktop/app.py` resolves the project environment.
The interface uses Dear PyGui; analysis remains in the existing df-analyze pipeline.
Dataset inspection and configuration do not require loading the full ML stack.

Start with a CSV, Parquet, JSON or XLSX dataset. Review the preview and missing-value
summary, select the target and task, assign column roles, and choose a preset or
individual models. Each launched experiment receives its own timestamped folder.
The Results view reads the saved performance tables; lower is better for MAE,
MSQE and MDAE, while higher is better for the other available metrics.

The Visual workflow page provides draggable nodes and editable connections.
Enable **Use canvas models for the next run** to execute its learner choices.
Rebuild from model settings after switching task or applying a preset if you want
those choices reflected on the canvas. Incomplete workflows cannot run.
Export experiment settings to keep the graph and its layout for another session.

## Python notebook

The desktop and web notebooks share a persistent Python kernel and a field-journal
starter with research notes, dataset exploration, and a styled distribution plot.
The web notebook separates mint code cards from lavender notes; the desktop uses
matching accents, cell titles, and a dedicated observations panel. Markdown notes
show headings, callouts, and simple tables in the desktop observations panel.

Use **Insert app dataset and settings** to capture the current configuration:
`raw_df` holds the original table and `df` applies duplicate removal and the
preparation recipe. Insert and run a new setup cell after changing app settings.
The analysis helpers add data quality, target distribution, or saved results code.

Cells can be duplicated, reordered, and restored with **Undo delete** (the last
20 deletions). **Run through here** executes setup cells through the selected cell;
**Run from here** executes it and the remaining cells. Both skip notes and stop
at the first Python error. Notebook exports preserve imported metadata and
attachments. Save to `.ipynb` to keep your work across app sessions.

## Development checks

```powershell
.venv\Scripts\python.exe -m pytest test/test_gui_common.py test/test_gui_cli_integration.py test/test_desktop_runner.py test/test_desktop_log_stream.py
.venv\Scripts\python.exe -m pyright desktop/app.py desktop/workflow_canvas.py desktop/dpg_context.py src/df_analyze/gui_common.py src/df_analyze/gui_workflow.py
```

`dpg_context.py` corrects Dear PyGui's generated context-manager return types at
the library boundary. Widget argument checking remains enabled.

Additional interaction checks are in `test/test_desktop_studio.py`,
`test/test_streamlit_studio.py`, and `test/test_gui_workflow.py`. The browser
canvas event handlers have dependency-free Node tests:
`node --test test/workflow_canvas.test.mjs`.

## Packaging

An existing PyInstaller specification is available:

```powershell
.venv\Scripts\pyinstaller.exe desktop\build.spec --noconfirm
```

The ML dependencies make this a large build. Source-level tests do not certify
the resulting executable. Verify each new package with a real small analysis:

```powershell
dist\df-analyze-desktop\df-analyze-desktop.exe --self-test --data data\small_classifier_data.json --target target --outdir desktop-smoke-results
```

The self-test briefly opens a real viewport, renders the charts, and runs an
analysis. It needs a graphical session (or Xvfb on Linux).

### Native Linux and macOS installers

`build_native.py` builds on the target operating system. It creates an isolated
Python 3.13.15 environment, uses the versions in `uv.lock`, freezes the app,
copies it outside the checkout into a path containing spaces, and runs its
chart-and-analysis self-test. It creates installers only if that test succeeds.
Outputs go into `dist/installer/native/`, with SHA-256 checksum files.

| Build host | Output | Target |
| --- | --- | --- |
| Ubuntu 22.04 x86-64 | `.deb` and portable `.tar.gz` | Ubuntu 22.04 or newer; other glibc distributions require testing |
| macOS 14 Apple Silicon | `.dmg` containing an `.app` | macOS 14 or newer on M-series Macs |

Intel Macs are not supported by the current dependency versions. These are
separate native builds, not conversions of the Windows executable.

On Ubuntu, install uv and the build/graphics tools, then run:

```bash
sudo apt-get update
sudo apt-get install -y binutils xz-utils xvfb xauth libgl1 libgl1-mesa-dri libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 libgomp1
uv run --no-project --python 3.13.15 python desktop/build_native.py
```

On an Apple Silicon Mac with uv and Apple's command-line developer tools:

```bash
brew install libomp
uv run --no-project --python 3.13.15 python desktop/build_native.py
```

The GitHub Actions workflow `.github/workflows/native-installers.yml` runs these
same builds on Ubuntu and macOS. Run **Build Linux and macOS installers** from
Actions, or push the `packaging/native-installers` branch. Download the two
installer artifacts from the successful workflow. It does not publish releases.

Install the Linux `.deb` with `sudo apt install ./df-analyze-desktop-4.1.0-linux-amd64.deb`.
For the portable archive, extract it and run `df-analyze-desktop/df-analyze-desktop`.
The portable build still needs the listed system graphics libraries.
On macOS, open the DMG and drag **df-analyze Desktop.app** into **Applications**.

The default macOS build uses PyInstaller's ad-hoc signing and is not notarized
by Apple. macOS may require first-launch approval in **System Settings > Privacy
& Security**. A distribution certificate can be provided with
`DF_ANALYZE_CODESIGN_IDENTITY` and an optional `DF_ANALYZE_ENTITLEMENTS_FILE`;
Apple notarization remains a separate distribution step.

The desktop build includes the analysis and embedding libraries. Linux uses
CPU PyTorch to avoid requiring CUDA. TensorFlow and the separate Streamlit web UI
are excluded because the desktop does not use them. Embedding model weights
are downloaded when needed. As with the existing frozen Windows application,
notebook execution requires a separate Python environment with ipykernel and
the project's dependencies; the frozen analysis executable is not a notebook
Python interpreter.

Frozen apps keep their analysis cache and GANDALF logs in user storage so they
can run from `/opt` or `/Applications` without modifying their installation.
