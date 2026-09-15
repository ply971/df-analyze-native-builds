"""Small real computations that certify the packaged native libraries load."""

from pathlib import Path


def check_data_and_models(output: Path) -> None:
    import torch  # import before transformers
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from lightgbm import LGBMRegressor
    from numba import njit
    from sklearn.ensemble import RandomForestClassifier
    from transformers import SiglipConfig, SiglipModel, XLMRobertaConfig, XLMRobertaModel

    from df_analyze import gui_common as gc

    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "c"]})
    for suffix in ("csv", "parquet", "xlsx", "json"):
        path = output / f"roundtrip.{suffix}"
        gc.save_dataset(frame, path)
        pd.testing.assert_frame_equal(frame, gc.load_full_dataset(path))
    print("[self-test] CSV, Parquet, XLSX and JSON import/export OK", flush=True)

    rng = np.random.default_rng(17)
    features = rng.normal(size=(40, 4))
    labels = (features[:, 0] > 0).astype(int)
    models = [
        RandomForestClassifier(n_estimators=3, n_jobs=1, random_state=17),
        CatBoostClassifier(iterations=3, thread_count=1, allow_writing_files=False, verbose=False),
        LGBMRegressor(n_estimators=3, n_jobs=1, min_child_samples=2, verbosity=-1),
    ]
    for model in models:
        model.fit(features, labels)
        if not np.isfinite(model.predict(features)).all():
            raise RuntimeError(f"Non-finite predictions from {type(model).__name__}")

    @njit
    def square_sum(values):
        return (values * values).sum()

    if not np.isclose(square_sum(features), np.square(features).sum()):
        raise RuntimeError("Numba calculation failed")
    tensor = torch.tensor(features, dtype=torch.float32, requires_grad=True)
    tensor.square().sum().backward()
    if not torch.isfinite(tensor.grad).all():
        raise RuntimeError("PyTorch gradient calculation failed")

    # Tiny models with random weights exercise the same text/image model code
    # as embeddings, without downloading multi-GB production model weights.
    common = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2)
    text_model = XLMRobertaModel(XLMRobertaConfig(vocab_size=64, **common)).eval()
    ids = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]])
    vision_model = SiglipModel(SiglipConfig(
        text_config=dict(vocab_size=64, max_position_embeddings=8, **common),
        vision_config=dict(image_size=32, patch_size=16, **common),
    )).eval()
    with torch.no_grad():
        assert torch.isfinite(text_model(ids).last_hidden_state).all()
        assert torch.isfinite(vision_model(input_ids=ids, pixel_values=torch.zeros(2, 3, 32, 32)).logits_per_image).all()
    print("[self-test] sklearn, CatBoost, LightGBM, Numba, torch and text/image backends OK", flush=True)
