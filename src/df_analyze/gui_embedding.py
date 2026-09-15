"""No-code configuration for df-embed; importing this never loads a model."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EmbeddingConfig:
    data_path: str = ""
    modality: str = "nlp"
    name: str = ""
    outpath: str = ""
    limit_samples: int = 0
    batch_size: int = 2
    action: str = "Create embeddings"

    def argv(self) -> list[str]:
        if self.modality not in ("nlp", "vision"):
            raise ValueError("Choose text (nlp) or image (vision) embeddings.")
        if self.action not in ACTIONS:
            raise ValueError("Choose an embedding action.")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Embedding batch size must be positive.")
        if type(self.limit_samples) is not int or self.limit_samples < 0:
            raise ValueError("Sample limit must be zero (all) or a positive integer.")
        args = ["--modality", self.modality]
        if self.action != ACTIONS[0]:
            return [*args, "--download", *(["--force-download"] if self.action == ACTIONS[2] else [])]
        path = Path(self.data_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".parquet":
            raise ValueError("Choose an existing Parquet embedding input.")
        output = self.output_path()
        if output.suffix.lower() != ".parquet":
            raise ValueError("Embedding output must be a Parquet file.")
        if output == path:
            raise ValueError("Choose an output different from your input dataset.")
        if not output.parent.is_dir():
            raise ValueError("Choose an existing folder for the embedding output.")
        args += ["--data", str(path), "--out", str(output), "--batch-size", str(self.batch_size)]
        if self.name.strip():
            args += ["--name", self.name.strip()]
        if self.limit_samples:
            args += ["--limit-samples", str(self.limit_samples)]
        return args

    def output_path(self) -> Path:
        return (Path(self.outpath).expanduser() if self.outpath.strip() else Path(self.data_path).expanduser().with_name("embedded.parquet")).resolve()


ACTIONS = ["Create embeddings", "Download models", "Replace downloaded models"]
INPUT_HELP = (
    "Text input: a Parquet table with text and label (classification) or target "
    "(regression) columns. Image input: image bytes and label or target columns. "
    "The output is a numeric Parquet dataset you can load into the analysis app. "
    "Download models once before offline use. Replacing models downloads them again."
)
