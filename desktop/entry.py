"""Dispatch packaged GUI, analysis workers and the bundled notebook kernel."""

import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def main() -> None:
    multiprocessing.freeze_support()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--self-test":
        import df_analyze

        print("[self-test] Package file:", df_analyze.__file__, flush=True)
        print("[self-test] Package paths:", list(df_analyze.__path__), flush=True)
        for directory in df_analyze.__path__:
            print("[self-test] GUI module files:", list(Path(directory).glob("gui*.py")), flush=True)
    if mode in {"--notebook-kernel", "--analysis-worker", "--embedding-worker"}:
        del sys.argv[1]
        import torch  # noqa: F401 -- import before transformers
        from joblib import parallel_config

        # Frozen executables are not general-purpose Python interpreters.
        # Use threads for model parallelism inside these isolated processes.
        with parallel_config(backend="threading"):
            if mode == "--notebook-kernel":
                from ipykernel.kernelapp import IPKernelApp

                IPKernelApp.launch_instance()
            elif mode == "--analysis-worker":
                from df_analyze._main import main as analyze

                analyze()
            else:
                from df_analyze.embedding.main import main as embed

                embed()
        return
    from desktop.app import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
