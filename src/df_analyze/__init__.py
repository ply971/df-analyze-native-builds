import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if not getattr(sys, "frozen", False):
    sys.path.append(str(SRC))


# from df_analyze._main import main
# def main() -> int:
#     print("Hello from rye-learn!")
#     return 0
