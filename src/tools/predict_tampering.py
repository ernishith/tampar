import argparse
import sys
from pathlib import Path
from typing import List, Optional

from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import pandas as pd
from joblib import dump, load

from src.tampering.compare import METRICS, CompareType
from src.tampering.evaluate import evaluate
from src.tampering.predictor import TamperingClassificator

SPLIT_STRING = "___"


def load_results(path: Path, exclude_base: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.reset_index(inplace=True)

    # Filter out base/base_adv folders if requested
    if exclude_base:
        before_count = len(df)
        # Check if 'view' column contains 'base' (case-insensitive)
        df = df[~df["view"].str.contains("base", case=False, na=False)]
        after_count = len(df)
        print(
            f"Excluded base folder: {before_count - after_count} rows removed ({before_count} -> {after_count})"
        )

    df["id"] = (
        df["view"]
        + SPLIT_STRING
        + df["sideface_name"]
        + SPLIT_STRING
        + df["gt_keypoints"].astype(str)
    )
    return df


def create_pivot(df: pd.DataFrame) -> pd.DataFrame:
    # Debug: Check for duplicates
    print(f"DEBUG: Total rows in CSV: {len(df)}")
    print(f"DEBUG: Unique IDs: {df['id'].nunique()}")
    duplicates = df[df.duplicated(subset=["id", "compare_type"], keep=False)]
    if len(duplicates) > 0:
        print(f"DEBUG: Found {len(duplicates)} duplicate rows!")
        print(
            f"DEBUG: Sample duplicates:\n{duplicates[['id', 'compare_type', 'msssim']].head(10)}"
        )

    df_pivot = df.pivot_table(
        index="id",
        columns="compare_type",
        values=METRICS,
        aggfunc=lambda x: ",".join(map(str, x)),
    )
    df_pivot.columns = [
        "score_{}_{}".format(col, method) for col, method in df_pivot.columns
    ]
    df_pivot = df_pivot.reset_index()

    df_final = pd.merge(
        df[["tampered", "tampering", "dataset_split", "gt_keypoints", "id"]],
        df_pivot,
        on="id",
    )
    df_final["tampering"] = df_final["tampering"].fillna("")
    df_final.fillna(-1, inplace=True)
    return df_final


def get_data_splits(df_input: pd.DataFrame, gt_keypoints: bool = False):
    data_gt = df_input[df_input["gt_keypoints"] == True]
    data_pred = df_input[df_input["gt_keypoints"] == False]

    if gt_keypoints:
        # Support both "validation" and "adversarial_validation" splits
        data_train = data_gt[data_gt["dataset_split"].str.contains("validation")]
        data_test = data_gt[data_gt["dataset_split"].str.contains("test")]
    else:
        # Support both "validation" and "adversarial_validation" splits
        data_train = data_pred[data_pred["dataset_split"].str.contains("validation")]
        data_test = data_pred[data_pred["dataset_split"].str.contains("test")]
    return data_train, data_test


def train_predictor(
    df_final: pd.DataFrame,
    gt_keypoints: bool = False,
    predictor_type: str = "simple_threshold",
    mode: str = "validation",
    balance_dataset: bool = False,
    save_model: Optional[str] = None,
    load_model: Optional[str] = None,
) -> pd.DataFrame:
    SCORES = [n for n in df_final.columns if n.startswith("score")]
    data_train, data_test = get_data_splits(df_final, gt_keypoints=gt_keypoints)

    # Debug: Check how many samples we have
    print(f"DEBUG: Total samples in df_final: {len(df_final)}")
    print(f"DEBUG: Training samples: {len(data_train)}")
    print(f"DEBUG: Test samples: {len(data_test)}")
    print(f"DEBUG: Unique dataset_split values: {df_final['dataset_split'].unique()}")
    print(f"DEBUG: gt_keypoints setting: {gt_keypoints}")

    results_performance = []
    for compare_types in [[t] for t in CompareType.SELECTION()] + [
        CompareType.SELECTION()
    ]:
        scores = [s for s in SCORES if s.split("_")[-1] in compare_types]
        scores = [s for s in scores if "_".join(s.split("_")[1:-1]) in METRICS]
        if len(scores) == 0:
            continue
        predictor = TamperingClassificator(predictor_type)
        X_train = data_train[scores].to_numpy().astype(float)
        y_train = data_train["tampered"].to_numpy().astype(int)
        if balance_dataset:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        ids_train = data_train["id"].to_numpy()
        predictor.set_data(X_train, y_train, ids_train)
        predictor.feature_names = [s.replace("score_", "") for s in scores]

        if load_model:
            # Load a pre-trained model and evaluate on test data
            model = load(load_model)
            X_test = data_test[scores].to_numpy().astype(float)
            y_test = data_test["tampered"].to_numpy().astype(int)
            test_metrics = evaluate(model, X_test, y_test)
            result_dict = {
                "predictor": predictor_type,
                "compare_types": ", ".join(compare_types),
                "scores": ", ".join(
                    set(["_".join(s.split("_")[1:-1]) for s in scores])
                ),
                **test_metrics,
            }
        else:
            # Default: cross-validation for both validation and test modes
            (
                train_metrics_summary,
                val_metrics_summary,
                models,
            ) = predictor.validate_model(5)
            result_dict = {
                "predictor": predictor_type,
                "compare_types": ", ".join(compare_types),
                "scores": ", ".join(
                    set(["_".join(s.split("_")[1:-1]) for s in scores])
                ),
                **val_metrics_summary,
            }

        # Add feature importance for tree-based models
        if load_model:
            trained_model = model
        else:
            trained_model = models[0]
        if hasattr(trained_model, "feature_importances_"):
            result_dict["feature_importance"] = {
                name: value
                for name, value in zip(
                    predictor.feature_names,
                    trained_model.feature_importances_,
                )
                if value > 0
            }
        else:
            result_dict["feature_importance"] = "N/A (ensemble model)"
        results_performance.append(result_dict)

        # Save model if explicitly requested
        if save_model and not load_model:
            predictor.test_split_size = 0
            final_model, train_metrics, _ = predictor.train()
            compare_suffix = "_".join(compare_types)
            model_path = f"{save_model}_{predictor_type}_{compare_suffix}.joblib"
            dump(final_model, model_path)
            print(f"Model saved to: {model_path}")

    df_results_ = pd.DataFrame(results_performance)
    return df_results_


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train tampering predictor using SimSAC similarity scores"
    )
    p.add_argument(
        "--mode",
        type=str,
        default="validation",
        help="Input run type(either 'validation' or 'test' or 'train')",
    )
    p.add_argument(
        "--gt_keypoints",
        action="store_true",  # default is false
        help="ground truth keypoints available",
    )
    p.add_argument(
        "--balance_dataset",
        action="store_true",  # default is false
        help="enable using smote for balancing dataset",
    )
    p.add_argument(
        "--simscores_csv",
        type=str,
        default=None,
        help="Path to simscores CSV file. If not provided, uses data/misc/simscores_validation.csv",
    )
    p.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Save trained model to this path prefix (e.g., --save_model mymodel will save mymodel_rf_plain.joblib)",
    )
    p.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load a pre-trained model from this path for inference instead of cross-validation",
    )
    p.add_argument(
        "--predictor_type",
        type=str,
        default="all",
        choices=[
            "simple_threshold",
            "decision_tree",
            "random_forest",
            "xgboost",
            "ensemble",
            "all",
        ],
        help='Type of predictor to train. Use "all" to compare all classifiers (default: all)',
    )
    p.add_argument(
        "--output",
        type=str,
        default="tampering_results.csv",
        help="Output CSV file for results (default: tampering_results.csv)",
    )
    p.add_argument(
        "--exclude_base",
        action="store_true",
        default=False,
        help="Exclude base and base_adv folders from analysis (default: False)",
    )
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]  # common pattern for CLI entry points
    args = build_parser().parse_args(argv)
    mode = args.mode
    predictor = args.predictor_type
    gt_keypoints = args.gt_keypoints
    balance_dataset = args.balance_dataset
    print(
        f"Mode: {mode}, gt_keypoints: {gt_keypoints}, predictor: {predictor}, balance_dataset: {balance_dataset}"
    )
    if mode not in ("validation", "train", "test"):
        raise ValueError("mode must be either 'validation', 'train', or 'test'")

    # Determine input path
    if args.simscores_csv:
        simscores_path = Path(args.simscores_csv)
    else:
        file_type = "test" if mode == "test" else "validation"
        simscores_path = ROOT / "data" / "misc" / f"simscores_{file_type}.csv"

    if not simscores_path.exists():
        print(f"Error: SimScores CSV not found at {simscores_path}")
        sys.exit(1)

    print(f"Loading SimScores from: {simscores_path}")
    print(f"Exclude base folder: {args.exclude_base}")
    df = load_results(simscores_path, exclude_base=args.exclude_base)
    df_final = create_pivot(df)
    # Determine which predictors to run
    if args.predictor_type == "all":
        predictor_types = [
            "simple_threshold",
            "decision_tree",
            "random_forest",
            "xgboost",
            "ensemble",
        ]
    else:
        predictor_types = [args.predictor_type]
    # Run each predictor and collect results
    all_results = []
    for predictor_type in predictor_types:
        print(f"\n{'='*70}")
        print(f"Training with predictor: {predictor_type.upper()}")
        print(f"{'='*70}")

        try:
            df_results = train_predictor(
                df_final,
                gt_keypoints=gt_keypoints,
                predictor_type=predictor_type,
                mode=mode,
                balance_dataset=balance_dataset,
                save_model=args.save_model,
                load_model=args.load_model,
            )
            all_results.append(df_results)
        except Exception as e:
            print(f"Error with {predictor_type}: {e}")
            continue

    # Combine all results
    if len(all_results) > 0:
        df_combined = pd.concat(all_results, ignore_index=True)
        df_combined.to_csv(args.output, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {args.output}")
        print(f"{'='*70}")

        # Print summary comparison
        if len(all_results) > 1:
            print("\n" + "=" * 70)
            print("CLASSIFIER COMPARISON SUMMARY")
            print("=" * 70)
            # Show best accuracy for each predictor
            # Use correct column names from evaluate.py
            agg_dict = {"accuracy": "max"}

            # Check which metric columns exist
            if "f1_binary" in df_combined.columns:
                agg_dict["f1_binary"] = "max"
            if "precision_binary" in df_combined.columns:
                agg_dict["precision_binary"] = "max"
            if "recall_binary" in df_combined.columns:
                agg_dict["recall_binary"] = "max"
            if "roc_auc" in df_combined.columns:
                agg_dict["roc_auc"] = "max"

            summary = df_combined.groupby("predictor").agg(agg_dict).round(4)
            print(summary.to_string())
            print("\n")
    else:
        print("No results generated.")


if __name__ == "__main__":
    raise SystemExit(main())
