from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
SRC = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
sys.path.append(str(SRC))  # isort: skip
# fmt: on


import logging
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import List, Optional, TypeVar, Union
from warnings import warn

from pandas import DataFrame

from df_analyze.analysis.univariate.associate import target_associations
from df_analyze.analysis.univariate.predict.predict import univariate_predictions
from df_analyze.cli.cli import ProgramOptions, get_options
from df_analyze.hypertune import evaluate_tuned
from df_analyze.multitarget import _eval_results_for_target
from df_analyze.nonsense import silence_spam
from df_analyze.preprocessing.cleaning import sanitize_names
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import prepare_data
from df_analyze.preprocessing.targets import as_target_list
from df_analyze.selection.filter import FilterSelected, filter_select_features
from df_analyze.selection.multitarget import (
    aggregate_filter_selected,
    aggregate_model_selected,
)
from df_analyze.selection.models import model_select_features

RESULTS_DIR = Path(__file__).parent / "results"

T = TypeVar("T")

# https://github.com/Lightning-AI/pytorch-lightning/issues/3431#issuecomment-2130390858
logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")


class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        return "available:" not in record.getMessage()


logger.addFilter(IgnorePLFilter())


def listify(item: Union[T, list[T], tuple[T, ...]]) -> List[T]:
    if isinstance(item, list):
        return item
    if isinstance(item, tuple):
        return [i for i in item]
    return [item]


def sort_df(df: DataFrame) -> DataFrame:
    """Auto-detect if classification or regression based on columns and sort"""
    cols = [c.lower() for c in df.columns]
    is_regression = False
    sort_col = None
    for col in cols:
        if ("mae" in col) and ("sd" not in col):
            is_regression = True
            sort_col = col
            break
    if sort_col is None:
        return df
    ascending = is_regression
    return df.sort_values(by=sort_col, ascending=ascending)


def print_sorted(df: DataFrame) -> None:
    """Auto-detect if classification or regression based on columns"""
    cols = [c.lower() for c in df.columns]
    is_regression = None
    sort_col = None
    for col in cols:
        if "mae" in col:
            is_regression = True
            break
        if "acc" in col:
            is_regression = False
            break
    if is_regression is None:
        print(df.to_markdown(tablefmt="simple", floatfmt="0.3f"))
        return
    sort_col = "mae" if is_regression else "acc"
    ascending = is_regression
    table = df.sort_values(by=sort_col, ascending=ascending).to_markdown(
        tablefmt="simple", floatfmt="0.3f"
    )
    print(table)


def log_options(options: ProgramOptions) -> None:
    opts = deepcopy(options.__dict__)

    print("Will run analyses with options:")
    pprint(opts, indent=2, depth=2, compact=False)


def _adaptive_error_base_dir(
    prog_dirs,
    fold_idx: Optional[int],
    target_name: Optional[str] = None,
) -> Optional[Path]:
    if prog_dirs.results is None:
        return None
    base_dir = prog_dirs.results / "adaptive_error"
    if fold_idx is not None:
        base_dir = base_dir / f"test{fold_idx:02d}"
    if target_name is not None:
        base_dir = base_dir / str(prog_dirs._safe_target_name(target_name))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def main() -> None:
    # TODO:
    # get arguments
    # run analyses based on args
    options = get_options()
    options.to_json()
    # if options.verbosity.value > 0:
    #     log_options(options)

    # print(options.drops)
    # sys.exit(0)
    is_cls = options.is_classification
    prog_dirs = options.program_dirs
    targets = as_target_list(options.targets)
    target_spec: Union[str, list[str]] = targets[0] if len(targets) == 1 else targets
    categoricals = options.categoricals
    ordinals = options.ordinals
    drops = options.drops
    grouper = options.grouper
    test_size = options.test_val_size
    method = options.tests_method
    seed = options.seed
    # joblib_cache = options.program_dirs.joblib_cache
    # if joblib_cache is not None:
    #     memory = Memory(location=joblib_cache)

    df = options.load_df()
    merges = options.merged_df()
    if merges is not None:
        merged_df, ix_train, ix_tests = merges
    else:
        merged_df, ix_train, ix_tests = (None, None, None)

    df, renames = sanitize_names(df, target_spec)
    if merged_df is not None:
        # We already check column names are identical across test dfs, so
        # we do not need to use renaming info twice
        merged_df = sanitize_names(merged_df, target_spec)[0]
    prog_dirs.save_renames(renames)

    # Likewise, below variables are just list[str], and so we don't need to do
    # anything for the merged_df
    categoricals = renames.rename_columns(categoricals)
    ordinals = renames.rename_columns(ordinals)
    drops = renames.rename_columns(drops)

    if merged_df is not None:
        merged_df, inspection = inspect_data(
            merged_df, target_spec, grouper, categoricals, ordinals, drops, _warn=True
        )
    else:
        df, inspection = inspect_data(
            df, target_spec, grouper, categoricals, ordinals, drops, _warn=True
        )
    prog_dirs.save_inspect_reports(inspection)
    prog_dirs.save_inspect_tables(inspection)

    raw_df = merged_df if merged_df is not None else df
    prepared = prepare_data(
        raw_df, target_spec, grouper, inspection, is_cls, ix_train, ix_tests, method,
        nan_handling=options.nan_handling,
        robust=options.norm.value == "robust",
    )
    prog_dirs.save_prepared_raw(prepared)
    prog_dirs.save_prep_report(prepared.to_markdown())

    # describe prepared features
    if isinstance(prepared.y, DataFrame):
        target_cols = prepared.target_cols
        first_target = target_cols[0]
        desc_cont, desc_cat, desc_target = prepared.for_target(
            first_target
        ).describe_features()
    else:
        desc_cont, desc_cat, desc_target = prepared.describe_features()
    prog_dirs.save_feature_descriptions(desc_cont, desc_cat, desc_target)

    prep_splits = prepared.get_splits(test_size=test_size, seed=seed)
    for fold_idx, (prep_train, prep_test) in enumerate(prep_splits):
        # prep_train, prep_test = prepared.split()
        if merged_df is None:
            fold_idx = None

        if not isinstance(prep_train.y, DataFrame):
            associations = target_associations(prep_train)
            prog_dirs.save_univariate_assocs(associations, fold_idx)
            prog_dirs.save_assoc_report(associations.to_markdown(), fold_idx)

            if options.no_preds:
                predictions = None
            else:
                predictions = univariate_predictions(prep_train, is_cls)
                prog_dirs.save_univariate_preds(predictions, fold_idx)
                prog_dirs.save_pred_report(predictions.to_markdown(), fold_idx)

            # select features via filter methods first
            assoc_filtered, pred_filtered = filter_select_features(
                prep_train, associations, predictions, options
            )
            prog_dirs.save_filter_report(assoc_filtered, fold_idx)
            prog_dirs.save_filter_report(pred_filtered, fold_idx)

            # TODO: make embedded and wrapper selection mutually exclusive. Only two
            # phases of feature selection: filter selection, and model-based
            # selection, wher model-based selection means either embedded or wrapper
            # (stepup, stepdown) methods.
            selected = model_select_features(prep_train, options)
            prog_dirs.save_model_selection_reports(selected, fold_idx)
            prog_dirs.save_model_selection_data(selected, fold_idx)
        else:
            target_cols = prep_train.target_cols
            per_target_assoc = {}
            per_target_pred = {}
            per_target_selected = {}
            has_pred_for_any_target = False

            for target_name in target_cols:
                prep_train_t = prep_train.for_target(target_name)
                associations_t = target_associations(prep_train_t)
                prog_dirs.save_univariate_assocs(
                    associations_t, fold_idx=fold_idx, target=target_name
                )
                prog_dirs.save_assoc_report(
                    associations_t.to_markdown(), fold_idx=fold_idx, target=target_name
                )

                if options.no_preds:
                    predictions_t = None
                else:
                    predictions_t = univariate_predictions(prep_train_t, is_cls)
                    prog_dirs.save_univariate_preds(
                        predictions_t, fold_idx=fold_idx, target=target_name
                    )
                    prog_dirs.save_pred_report(
                        predictions_t.to_markdown(),
                        fold_idx=fold_idx,
                        target=target_name,
                    )

                assoc_t, pred_t = filter_select_features(
                    prep_train_t, associations_t, predictions_t, options
                )
                selected_t = model_select_features(prep_train_t, options)
                per_target_assoc[target_name] = assoc_t
                per_target_pred[target_name] = pred_t
                per_target_selected[target_name] = selected_t
                if pred_t is not None:
                    has_pred_for_any_target = True

            mt_top_k = options.mt_top_k
            if isinstance(mt_top_k, int) and mt_top_k <= 0:
                mt_top_k = None

            assoc_filtered = aggregate_filter_selected(
                [per_target_assoc[target_name] for target_name in target_cols],
                method="association",
                is_cls=is_cls,
                strategy=options.mt_agg_strategy,
                top_k=mt_top_k,
                target_names=target_cols,
            )
            if options.no_preds or not has_pred_for_any_target:
                pred_filtered = None
            else:
                pred_parts = []
                for target_name in target_cols:
                    pred_t = per_target_pred[target_name]
                    if pred_t is None:
                        pred_parts.append(
                            FilterSelected(
                                selected=[],
                                cont_scores=None,
                                cat_scores=None,
                                method="prediction",
                                is_classification=is_cls,
                            )
                        )
                    else:
                        pred_parts.append(pred_t)
                pred_filtered = aggregate_filter_selected(
                    pred_parts,
                    method="prediction",
                    is_cls=is_cls,
                    strategy=options.mt_agg_strategy,
                    top_k=mt_top_k,
                    target_names=target_cols,
                )
            selected = aggregate_model_selected(
                [per_target_selected[target_name] for target_name in target_cols],
                is_cls=is_cls,
                strategy=options.mt_agg_strategy,
                top_k=mt_top_k,
                target_names=target_cols,
            )

            prog_dirs.save_filter_report(assoc_filtered, fold_idx)
            prog_dirs.save_filter_report(pred_filtered, fold_idx)
            prog_dirs.save_model_selection_reports(selected, fold_idx)
            prog_dirs.save_model_selection_data(selected, fold_idx)

        silence_spam()
        eval_results = evaluate_tuned(
            prepared=prepared,
            prep_train=prep_train,
            prep_test=prep_test,
            assoc_filtered=assoc_filtered,
            pred_filtered=pred_filtered,
            model_selected=selected,
            options=options,
        )
        prog_dirs.save_eval_report(eval_results, fold_idx)
        prog_dirs.save_eval_tables(eval_results, fold_idx)
        prog_dirs.save_eval_data(eval_results, fold_idx)
        if options.adaptive_error:
            from df_analyze.analysis.adaptive_error.runner import (
                run_adaptive_error_analysis,
            )
            base_dir = _adaptive_error_base_dir(prog_dirs, fold_idx=fold_idx)

            if isinstance(prep_train.y, DataFrame):
                target_cols = prep_train.target_cols

                for target_name in target_cols:
                    prep_train_t = prep_train.for_target(target_name)
                    prep_test_t = prep_test.for_target(target_name)
                    eval_results_t = _eval_results_for_target(
                        eval_results=eval_results,
                        prep_train_t=prep_train_t,
                        prep_test_t=prep_test_t,
                        target=target_name,
                    )

                    target_base_dir = _adaptive_error_base_dir(
                        prog_dirs, fold_idx=fold_idx, target_name=target_name
                    )

                    run_adaptive_error_analysis(
                        prep_train=prep_train_t,
                        prep_test=prep_test_t,
                        eval_results=eval_results_t,
                        options=options,
                        prog_dirs=prog_dirs,
                        no_preds=options.no_preds,
                        base_dir=target_base_dir,
                    )
            else:
                run_adaptive_error_analysis(
                    prep_train=prep_train,
                    prep_test=prep_test,
                    eval_results=eval_results,
                    options=options,
                    prog_dirs=prog_dirs,
                    no_preds=options.no_preds,
                    base_dir=base_dir,
                )
        try:
            print(eval_results.to_markdown())
        except ValueError as e:
            warn(
                f"Got error when attempting to print final report:\n{e}\n"
                f"Details:\n{traceback.format_exc()}"
            )
    # TODO: Assemble final summary tables


if __name__ == "__main__":
    main()
