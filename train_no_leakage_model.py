import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "epl_final.csv"


# Features that leak target information because they are known only after kickoff/end.
LEAKAGE_COLUMNS = [
    "FullTimeHomeGoals",
    "FullTimeAwayGoals",
    "FullTimeResult",
    "HalfTimeHomeGoals",
    "HalfTimeAwayGoals",
    "HalfTimeResult",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
]


def add_rolling_team_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Create pre-match features using only past matches (shift(1) avoids leakage)."""
    out = df.copy()

    # Normalize common column names used in many EPL datasets.
    rename_map = {
        "FullTimeHomeGoals": "FTHG",
        "FullTimeAwayGoals": "FTAG",
        "FullTimeResult": "FTR",
        "HomeYellowCards": "HY",
        "AwayYellowCards": "AY",
        "HomeRedCards": "HR",
        "AwayRedCards": "AR",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    # Parse/sort by date so train/test respects time.
    if "MatchDate" in out.columns:
        out["MatchDate"] = pd.to_datetime(out["MatchDate"], errors="coerce")
    else:
        out["MatchDate"] = pd.NaT

    out = out.sort_values(["MatchDate", "Season"], na_position="last").reset_index(drop=True)

    # Current-match result encoded from home perspective.
    if "FTR" not in out.columns:
        raise ValueError("Target column not found. Expected 'FTR' or 'FullTimeResult'.")

    out["target"] = out["FTR"].map({"H": 0, "D": 1, "A": 2})

    # Points earned in this match (for form), then shifted later to keep pre-match only.
    out["home_points"] = out["FTR"].map({"H": 3, "D": 1, "A": 0})
    out["away_points"] = out["FTR"].map({"H": 0, "D": 1, "A": 3})

    home_stats = ["HomeShots", "HomeShotsOnTarget", "HomeCorners", "HomeFouls", "HY", "HR", "FTHG", "FTAG"]
    away_stats = ["AwayShots", "AwayShotsOnTarget", "AwayCorners", "AwayFouls", "AY", "AR", "FTAG", "FTHG"]

    # Create rolling means from previous home/away matches for each team.
    for col in home_stats:
        if col in out.columns:
            out[f"home_{col}_avg_{window}"] = (
                out.groupby("HomeTeam")[col]
                .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            )

    for col in away_stats:
        if col in out.columns:
            out[f"away_{col}_avg_{window}"] = (
                out.groupby("AwayTeam")[col]
                .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            )

    out[f"home_form_points_avg_{window}"] = (
        out.groupby("HomeTeam")["home_points"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    )
    out[f"away_form_points_avg_{window}"] = (
        out.groupby("AwayTeam")["away_points"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    )

    # Season-progress features (available pre-match).
    out["home_match_number"] = out.groupby(["Season", "HomeTeam"]).cumcount() + 1
    out["away_match_number"] = out.groupby(["Season", "AwayTeam"]).cumcount() + 1

    out["form_points_diff"] = out[f"home_form_points_avg_{window}"] - out[f"away_form_points_avg_{window}"]

    return out


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df_feat = add_rolling_team_features(df)

    # Drop known leakage columns from candidate features.
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df_feat.columns]

    candidate_features = [
        c
        for c in df_feat.columns
        if c not in drop_cols + ["target", "home_points", "away_points"]
    ]

    # Keep only columns we can know pre-match.
    pre_match_features = [
        c
        for c in candidate_features
        if c
        in {
            "Season",
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "home_match_number",
            "away_match_number",
            "form_points_diff",
        }
        or c.startswith("home_")
        or c.startswith("away_")
    ]

    # Drop rows with missing target labels.
    df_feat = df_feat.dropna(subset=["target"]).copy()

    X = df_feat[pre_match_features]
    y = df_feat["target"].astype(int)

    return X, y, drop_cols


def time_based_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    # Respect chronology to avoid future-to-past leakage.
    if "MatchDate" in X.columns:
        sort_idx = X["MatchDate"].fillna(pd.Timestamp.max).sort_values().index
        X = X.loc[sort_idx].reset_index(drop=True)
        y = y.loc[sort_idx].reset_index(drop=True)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    datetime_cols = [c for c in X_train.columns if np.issubdtype(X_train[c].dtype, np.datetime64)]

    X_train_model = X_train.drop(columns=datetime_cols, errors="ignore")
    X_test_model = X_test.drop(columns=datetime_cols, errors="ignore")

    categorical_cols = [c for c in X_train_model.columns if X_train_model[c].dtype == "object"]
    numeric_cols = [c for c in X_train_model.columns if c not in categorical_cols]

    preprocess = ColumnTransformer(
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

    model = LogisticRegression(max_iter=3000, multi_class="multinomial")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipeline.fit(X_train_model, y_train)
    y_pred = pipeline.predict(X_test_model)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\n=== Evaluation Metrics (No Leakage) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Home Win", "Draw", "Away Win"],
            zero_division=0,
        )
    )

    return pipeline


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, dropped_leakage = build_feature_table(df)

    print("=== Leakage Features Removed ===")
    print(dropped_leakage)
    print(f"\nTotal samples after feature prep: {len(X)}")

    X_train, X_test, y_train, y_test = time_based_split(X, y, test_size=0.2)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
