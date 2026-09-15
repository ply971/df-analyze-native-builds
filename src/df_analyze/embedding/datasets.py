from __future__ import annotations

import os
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import (
    Optional,
    Union,
)

import numpy as np
import pandas as pd
from pandas import DataFrame, RangeIndex, Series
from PIL import Image
from PIL.Image import Image as ImageObject
from tqdm import tqdm

from df_analyze.embedding.cli import EmbeddingModality, EmbeddingOptions


class EmbeddingDataset:
    def __init__(
        self,
        datapath: Optional[Path],
        name: Optional[str],
    ) -> None:
        self.datapath = self.validate_datapath(datapath)
        self.name = name or self.datapath.name
        self._df = None

    def __str__(self) -> str:
        cls = f"{self.__class__.__name__}"
        return f"{cls}('{self.name}' @ {self.datapath})"

    __repr__ = __str__

    def validate_datapath(self, path: Optional[Path]) -> Path:
        if path is None:
            raise ValueError("Got `None` for datapath. Embedding data cannot be None.")
        if not path.exists():
            raise FileNotFoundError(
                f"No data found at {path} (realpath: {os.path.realpath(path)})"
            )
        if path.suffix != ".parquet":
            raise ValueError(
                f"Data must be in .parquet format, but got: {path}. Please refer to df-analyze documentation"
            )
        return path

    def load_raw(self) -> DataFrame:
        try:
            return pd.read_parquet(self.datapath).reset_index(drop=True)
        except Exception as e:
            raise RuntimeError(
                f"Could not load .parquet data at {self.datapath}. Details above."
            ) from e

    @property
    def is_cls(self) -> bool:
        return "label" in self.load().columns.tolist()

    def X(self, limit: Optional[int] = None) -> Series:
        df = self.load(limit=limit)
        modality = EmbeddingDataset.get_modality(df)
        if modality is EmbeddingModality.NLP:
            return df["text"]
        elif modality is EmbeddingModality.Vision:
            return df["image"]
        else:
            raise NotImplementedError(f"Unknown modality: {modality}")

    def y(self, limit: Optional[int] = None) -> Series:
        df = self.load(limit=limit)
        if self.is_cls:
            return df["label"]
        else:
            return df["target"]

    def load(self, limit: Optional[int] = None) -> DataFrame:
        if self._df is not None:
            return self._df
        df = self.load_raw()
        if limit is not None:
            df = df.iloc[:limit]
        self.validate_cols(df)
        self.validate_data(df)
        self._df = df
        return self._df

    def validate_cols(self, df: DataFrame) -> None:
        """Check that columns are correctly named and with the correct type"""
        df = df.reset_index(drop=True)  # just in case
        VALID_COLS = [
            sorted(["image", "label"]),
            sorted(["image", "target"]),
            sorted(["text", "label"]),
            sorted(["text", "target"]),
        ]

        cols = sorted(df.columns.tolist())
        if len(cols) > 2:  # drop junk
            keeps = ["image", "text", "label", "target"]
            drops = list(set(cols).difference(keeps))
            df.drop(columns=drops, inplace=True)
            cols = sorted(df.columns.tolist())

        if cols not in VALID_COLS:
            raise ValueError(
                "Malformed data. Data must have only two columns, i.e. be one of:"
                f"{VALID_COLS}. Got: {cols}"
            )
        if "image" in cols:
            if not df["image"].apply(lambda x: isinstance(x, bytes)).all():
                raise TypeError(
                    "Found column 'image' in data, but not all rows are `bytes` type"
                )
        if "text" in cols:
            if not df["text"].apply(lambda x: isinstance(x, str)).all():
                ix = df["text"].apply(lambda x: not isinstance(x, str))
                strange = df["text"][ix]
                raise TypeError(
                    "Found column 'text' in data, but not all rows are `str` type.\n"
                    f"Non-string values:\n{strange}"
                )
        if "label" in cols:
            try:
                labels = df["label"].astype(np.int64, errors="raise")
            except Exception as e:
                raise TypeError(
                    "Found column 'label' in data, but encountered error (above) "
                    f"when attempting to coerce to np.int64. Series:\n{df['label']}"
                ) from e
            if not (labels.apply(str) == df["label"].apply(str)).all():
                raise TypeError(
                    "Found column 'label' in data, but labels change values after "
                    "coercion to np.int64. This likely means labels are stored in "
                    "a floating point or other format. Convert them to an integer format "
                    "to resolve this error."
                )
        if "target" in cols:
            try:
                target = df["target"].astype(float, errors="raise")
            except Exception as e:
                raise TypeError(
                    "Found column 'target' in data, but encountered error (above) "
                    "when attempting to coerce to float."
                ) from e
            if not (target == df["target"]).all():
                raise TypeError(
                    "Found column 'target' in data, but targets change values after "
                    "coercion to float. This likely means you have either NaN values "
                    "or other inappropriate data types (e.g. object, string) in your "
                    "target column. Convert these values to floating point or remove "
                    "them in order to resolve this error."
                )

    @staticmethod
    def get_modality(validated: DataFrame) -> EmbeddingModality:
        VISION_COLS = [
            sorted(["image", "label"]),
            sorted(["image", "target"]),
        ]
        NLP_COLS = [
            sorted(["text", "label"]),
            sorted(["text", "target"]),
        ]
        valids = ["image", "text", "label", "target"]
        cols = sorted(validated.columns.tolist())
        cols = sorted(set(cols).intersection(valids))

        if cols in VISION_COLS:
            return EmbeddingModality.Vision
        elif cols in NLP_COLS:
            return EmbeddingModality.NLP
        else:
            raise RuntimeError(f"Impossible! Columns: {cols}")

    @abstractmethod
    def validate_data(self, df: DataFrame) -> None:
        """
        To be performed AFTER .load(), this checks some basic sanity stuff
        like whether all images have a reasonable shape, such as image size and
        channels, or text length, and also number of samples.

        We do NOT validate e.g. the number of samples per class, this is handled
        later by df-analyze tabular analysis functions.
        """
        modality = EmbeddingDataset.get_modality(df)
        if modality is EmbeddingModality.Vision:
            are_PIL = df["image"].apply(lambda x: isinstance(x, ImageObject))
            if not are_PIL.all():
                raise ValueError(
                    f"Some rows are not PIL Images:\n{df['image'][~are_PIL]}"
                )
            pass
            # for ix, im in enumerate(df["image"]):
            #     if len(im.size) != 3:
            #         plt.imshow(im)
            #         plt.show()
            #         raise ValueError(f"Invalid image shape: {im.size} at index {ix}")
            #     if 3 not in im.size:
            #         raise ValueError(
            #             f"Possible one-channel image: {im.size} at index {ix}"
            #         )
        elif modality is EmbeddingModality.NLP:
            # unclear what to do here besides running the tokenizer on all inputs
            pass
        else:
            raise NotImplementedError(f"No validation logic impemented for {modality}")

    @abstractmethod
    def embed(self) -> DataFrame:
        pass


class VisionDataset(EmbeddingDataset):
    def __init__(
        self,
        datapath: Optional[Path],
        name: Optional[str],
    ) -> None:
        super().__init__(datapath=datapath, name=name)
        self.images_converted: bool = False

    @staticmethod
    def random(tempdir: str, load_limit: Optional[int] = None) -> VisionDataset:
        """
        Notes
        -----
        We must pass in an existing `tempdir` since otherwise it would be destroyed
        if created within this function, and then subsequent EmbeddingOptions
        validation would fail.
        """
        N = np.random.randint(20, 500)
        all_bytes = DataFrame(index=RangeIndex(N), columns=["image"])
        for i in range(N):
            W = np.random.randint(32, 512)
            H = np.random.randint(32, 512)
            sigma = np.random.uniform(0.1, 10)
            img = Image.effect_noise(size=(W, H), sigma=sigma).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG")
            byts = buf.getvalue()
            all_bytes.loc[i] = byts
        labels = Series(
            data=np.random.randint(0, 2, N), name="label", index=all_bytes.index
        )
        try:
            file = Path(tempdir) / "data.parquet"
            df = pd.concat([all_bytes, labels], axis=1)
            df.to_parquet(file)
            randname = f"random__{np.random.randint(0, 999, 1).item():03d}"
            ds = VisionDataset(datapath=file, name=randname)
            # must call before tempdir is destroyed
            ds.load(limit=load_limit)
        except Exception as e:
            raise RuntimeError("Problem during creation of temporary datafile") from e
        return ds

    def load(self, limit: Optional[int] = None) -> DataFrame:
        if self.images_converted and (self._df is not None):
            return self._df

        # https://github.com/python-pillow/Pillow/issues/4987#issuecomment-710994934
        #
        # "
        # To protect against potential DOS attacks caused by “decompression
        # bombs” (i.e. malicious files which decompress into a huge amount of
        # data and are designed to crash or cause disruption by using up a lot of
        # memory), Pillow will issue a DecompressionBombWarning if the number of
        # pixels in an image is over a certain limit, PIL.Image.MAX_IMAGE_PIXELS.
        #
        # This threshold can be changed by setting PIL.Image.MAX_IMAGE_PIXELS. It
        # can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
        #
        # If desired, the warning can be turned into an error with
        # warnings.simplefilter('error', Image.DecompressionBombWarning) or
        # suppressed entirely with warnings.simplefilter('ignore',
        # Image.DecompressionBombWarning). See also the logging documentation to
        # have warnings output to the logging facility instead of stderr.
        #
        # If the number of pixels is greater than twice
        # PIL.Image.MAX_IMAGE_PIXELS, then a DecompressionBombError will be
        # raised instead. So:
        #
        #   from PIL import Image
        #   Image.MAX_IMAGE_PIXELS = None   # disables the warning
        #   Image.open(...)   # whatever operation you now run should work
        # "
        Image.MAX_IMAGE_PIXELS = None  # disables the warning

        df = self.load_raw()
        df.drop_duplicates(inplace=True)
        self.validate_cols(df)
        if limit is not None:
            df = df.iloc[:limit].copy()

        # we do this in a loop because we want to not explode
        # memory with a new .apply column
        for ix, byts in tqdm(
            enumerate(df["image"]),
            desc="Converting images to RGB",
            total=len(df),
            disable=len(df) < 10000,
        ):
            img = Image.open(BytesIO(byts)).convert("RGB")
            del byts
            idx = df.index[ix]
            df.loc[idx, "image"] = img  # type: ignore
        # im = df["image"].apply(lambda b: Image.open(BytesIO(b)).convert("RGB"))  # type: ignore
        # df = pd.concat([im, df.drop(columns="image")], axis=1)
        self.validate_data(df)

        self._df = df
        self.images_converted = True
        return self._df


class NLPDataset(EmbeddingDataset):
    def __init__(
        self,
        datapath: Optional[Path],
        name: Optional[str],
    ) -> None:
        super().__init__(datapath=datapath, name=name)


def dataset_from_opts(opts: EmbeddingOptions) -> Union[VisionDataset, NLPDataset]:
    if opts.modality is EmbeddingModality.NLP:
        return NLPDataset(name=opts.name, datapath=opts.datapath)
    elif opts.modality is EmbeddingModality.Vision:
        return VisionDataset(datapath=opts.datapath, name=opts.name)
    else:
        raise NotImplementedError(f"No dataset defined for {opts.modality}")
