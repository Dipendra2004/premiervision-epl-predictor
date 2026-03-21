import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_no_leakage_model import build_feature_table, time_based_split


DATA_PATH = "epl_final.csv"
DRAW_CLASS = 1


def add_balance_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add matchup-balance indicators that often correlate with draws.

    These are derived from pre-match rolling features already created in the
    leakage-free pipeline.
    """
    out = X.copy()

    def add_abs_gap(col_a: str, col_b: str, new_name: str):
        if col_a in out.columns and col_b in out.columns:
            out[new_name] = (out[col_a] - out[col_b]).abs()

    add_abs_gap("home_form_points_avg_5", "away_form_points_avg_5", "abs_form_gap")
    add_abs_gap("home_FTHG_avg_5", "away_FTAG_avg_5", "abs_attack_gap")
    add_abs_gap("home_FTAG_avg_5", "away_FTHG_avg_5", "abs_defense_gap")
    add_abs_gap("home_HomeShotsOnTarget_avg_5", "away_AwayShotsOnTarget_avg_5", "abs_sot_gap")

    if "home_form_points_avg_5" in out.columns and "away_form_points_avg_5" in out.columns:
        out["form_sum"] = out["home_form_points_avg_5"] + out["away_form_points_avg_5"]

    return out


def split_train_validation_time(X_train: pd.DataFrame, y_train: pd.Series, val_ratio: float = 0.2):
    cut = int(len(X_train) * (1 - val_ratio))
    X_sub_train = X_train.iloc[:cut].reset_index(drop=True)
    y_sub_train = y_train.iloc[:cut].reset_index(drop=True)
    X_val = X_train.iloc[cut:].reset_index(drop=True)
    y_val = y_train.iloc[cut:].reset_index(drop=True)
    return X_sub_train, X_val, y_sub_train, y_val


def get_model_columns(X: pd.DataFrame):
    datetime_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)]
    X_model = X.drop(columns=datetime_cols, errors="ignore")
    categorical_cols = [c for c in X_model.columns if X_model[c].dtype == "object"]
    numeric_cols = [c for c in X_model.columns if c not in categorical_cols]
    return X_model, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    draw_f1 = f1_score((y_true == DRAW_CLASS).astype(int), (y_pred == DRAW_CLASS).astype(int), zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n=== {title} ===")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision (w)    : {precision:.4f}")
    print(f"Recall (w)       : {recall:.4f}")
    print(f"F1-score (w)     : {f1:.4f}")
    print(f"F1-score (macro) : {macro_f1:.4f}")
    print(f"Draw F1          : {draw_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Home Win", "Draw", "Away Win"], zero_division=0))

    return {
        "accuracy": acc,
        "precision_w": precision,
        "recall_w": recall,
        "f1_w": f1,
        "f1_macro": macro_f1,
        "f1_draw": draw_f1,
    }


def tune_multiclass_draw_threshold(X_sub_train, y_sub_train, X_val, y_val):
    X_sub_model, numeric_cols, categorical_cols = get_model_columns(X_sub_train)
    X_val_model = X_val.drop(columns=[c for c in X_val.columns if np.issubdtype(X_val[c].dtype, np.datetime64)], errors="ignore")

    best = None
    best_score = -1.0

    draw_weights = [1.5, 2.0, 2.5, 3.0, 3.5]
    thresholds = np.arange(0.22, 0.46, 0.02)

    for w_draw in draw_weights:
        class_weight = {0: 1.0, 1: w_draw, 2: 1.0}
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", LogisticRegression(max_iter=4000, class_weight=class_weight)),
            ]
        )
        pipeline.fit(X_sub_model, y_sub_train)
        probs = pipeline.predict_proba(X_val_model)

        for th in thresholds:
            raw_pred = np.argmax(probs, axis=1)
            pred = np.where(probs[:, DRAW_CLASS] >= th, DRAW_CLASS, raw_pred)

            draw_f1 = f1_score((y_val == DRAW_CLASS).astype(int), (pred == DRAW_CLASS).astype(int), zero_division=0)
            macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)

            score = 0.7 * draw_f1 + 0.3 * macro_f1
            if score > best_score:
                best_score = score
                best = {
                    "pipeline": pipeline,
                    "draw_weight": w_draw,
                    "draw_threshold": float(th),
                    "draw_f1_val": float(draw_f1),
                    "macro_f1_val": float(macro_f1),
                }

    return best


def train_two_stage_model(X_sub_train, y_sub_train, X_val, y_val):
    X_sub_model, numeric_cols, categorical_cols = get_model_columns(X_sub_train)
    X_val_model = X_val.drop(columns=[c for c in X_val.columns if np.issubdtype(X_val[c].dtype, np.datetime64)], errors="ignore")

    preprocess = build_preprocessor(numeric_cols, categorical_cols)

    # Stage 1: Draw vs Non-Draw
    y_draw_train = (y_sub_train == DRAW_CLASS).astype(int)
    stage1 = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=3000, class_weight={0: 1.0, 1: 2.8})),
        ]
    )
    stage1.fit(X_sub_model, y_draw_train)

    # Stage 2: Home vs Away on non-draw training rows only.
    non_draw_idx = y_sub_train != DRAW_CLASS
    y_ha_train = (y_sub_train[non_draw_idx] == 2).astype(int)  # 0=home, 1=away
    stage2 = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    stage2.fit(X_sub_model.loc[non_draw_idx], y_ha_train)

    p_draw = stage1.predict_proba(X_val_model)[:, 1]
    p_away_given_non_draw = stage2.predict_proba(X_val_model)[:, 1]

    best = None
    best_score = -1.0
    for th in np.arange(0.22, 0.46, 0.02):
        pred = []
        for p_draw_i, p_away in zip(p_draw, p_away_given_non_draw):
            if p_draw_i >= th:
                pred.append(DRAW_CLASS)
            else:
                pred.append(2 if p_away >= 0.5 else 0)

        pred = np.asarray(pred)
        draw_f1 = f1_score((y_val == DRAW_CLASS).astype(int), (pred == DRAW_CLASS).astype(int), zero_division=0)
        macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        score = 0.7 * draw_f1 + 0.3 * macro_f1

        if score > best_score:
            best_score = score
            best = {
                "stage1": stage1,
                "stage2": stage2,
                "draw_threshold": float(th),
                "draw_f1_val": float(draw_f1),
                "macro_f1_val": float(macro_f1),
            }

    return best


def predict_two_stage(best_two_stage: dict, X: pd.DataFrame) -> np.ndarray:
    X_model = X.drop(columns=[c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)], errors="ignore")
    p_draw = best_two_stage["stage1"].predict_proba(X_model)[:, 1]
    p_away_given_non_draw = best_two_stage["stage2"].predict_proba(X_model)[:, 1]
    th = best_two_stage["draw_threshold"]

    pred = []
    for p_draw_i, p_away in zip(p_draw, p_away_given_non_draw):
        if p_draw_i >= th:
            pred.append(DRAW_CLASS)
        else:
            pred.append(2 if p_away >= 0.5 else 0)
    return np.asarray(pred)


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, _ = build_feature_table(df)
    X = add_balance_features(X)

    X_train, X_test, y_train, y_test = time_based_split(X, y, test_size=0.2)
    X_sub_train, X_val, y_sub_train, y_val = split_train_validation_time(X_train, y_train, val_ratio=0.2)

    print("=== Improved Leakage-Free Pipeline ===")
    print(f"Samples: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Sub-train: {len(X_sub_train)} | Validation: {len(X_val)}")

    # Model A: weighted multiclass with tuned draw threshold.
    best_mc = tune_multiclass_draw_threshold(X_sub_train, y_sub_train, X_val, y_val)
    print("\nBest multiclass settings:")
    print(
        {
            "draw_weight": best_mc["draw_weight"],
            "draw_threshold": best_mc["draw_threshold"],
            "val_draw_f1": round(best_mc["draw_f1_val"], 4),
            "val_macro_f1": round(best_mc["macro_f1_val"], 4),
        }
    )

    # Retrain Model A on full train set using selected draw weight.
    X_train_model, numeric_cols, categorical_cols = get_model_columns(X_train)
    X_test_model = X_test.drop(columns=[c for c in X_test.columns if np.issubdtype(X_test[c].dtype, np.datetime64)], errors="ignore")

    model_a = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight={0: 1.0, 1: best_mc["draw_weight"], 2: 1.0},
                ),
            ),
        ]
    )
    model_a.fit(X_train_model, y_train)
    probs_a = model_a.predict_proba(X_test_model)
    pred_a = np.where(probs_a[:, DRAW_CLASS] >= best_mc["draw_threshold"], DRAW_CLASS, np.argmax(probs_a, axis=1))
    metrics_a = evaluate_predictions(y_test.to_numpy(), pred_a, "Model A: Weighted Multiclass + Draw Threshold")

    # Model B: two-stage draw-first approach.
    best_two_stage = train_two_stage_model(X_sub_train, y_sub_train, X_val, y_val)
    print("\nBest two-stage settings:")
    print(
        {
            "draw_threshold": best_two_stage["draw_threshold"],
            "val_draw_f1": round(best_two_stage["draw_f1_val"], 4),
            "val_macro_f1": round(best_two_stage["macro_f1_val"], 4),
        }
    )

    # Refit two-stage models on full train set.
    X_train_model_full = X_train.drop(columns=[c for c in X_train.columns if np.issubdtype(X_train[c].dtype, np.datetime64)], errors="ignore")
    X_test_model_full = X_test.drop(columns=[c for c in X_test.columns if np.issubdtype(X_test[c].dtype, np.datetime64)], errors="ignore")

    y_draw_train_full = (y_train == DRAW_CLASS).astype(int)
    stage1_full = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", LogisticRegression(max_iter=3000, class_weight={0: 1.0, 1: 2.8})),
        ]
    )
    stage1_full.fit(X_train_model_full, y_draw_train_full)

    non_draw_idx_full = y_train != DRAW_CLASS
    y_ha_train_full = (y_train[non_draw_idx_full] == 2).astype(int)
    stage2_full = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    stage2_full.fit(X_train_model_full.loc[non_draw_idx_full], y_ha_train_full)

    p_draw_full = stage1_full.predict_proba(X_test_model_full)[:, 1]
    p_away_full = stage2_full.predict_proba(X_test_model_full)[:, 1]
    th_b = best_two_stage["draw_threshold"]

    pred_b = []
    for p_draw_i, p_away in zip(p_draw_full, p_away_full):
        if p_draw_i >= th_b:
            pred_b.append(DRAW_CLASS)
        else:
            pred_b.append(2 if p_away >= 0.5 else 0)
    pred_b = np.asarray(pred_b)
    metrics_b = evaluate_predictions(y_test.to_numpy(), pred_b, "Model B: Two-Stage Draw-First")

    print("\n=== Summary (Test Set) ===")
    print(
        {
            "model_a_draw_f1": round(metrics_a["f1_draw"], 4),
            "model_a_macro_f1": round(metrics_a["f1_macro"], 4),
            "model_b_draw_f1": round(metrics_b["f1_draw"], 4),
            "model_b_macro_f1": round(metrics_b["f1_macro"], 4),
        }
    )


if __name__ == "__main__":
    main()
