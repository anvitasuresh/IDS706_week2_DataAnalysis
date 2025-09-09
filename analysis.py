import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------- Load & Inspect ---------------------------


def load_data(csv_path: str) -> pd.DataFrame:
    print("Loading data for Student Dropout and Academic Success…")
    df = pd.read_csv(csv_path)
    return df


def inspect(df: pd.DataFrame):
    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== INFO ===")
    df.info()

    print("\n=== DESCRIBE ===")
    print(df.describe())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\n=== DUPLICATES ===")
    print(df.duplicated().sum())


# --------------------------- Data Cleaning ---------------------------


def clean_data(
    df: pd.DataFrame, target: str = "Target", outlier_z: float | None = 3.5
) -> pd.DataFrame:

    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    # Fix misspelling seen in column name
    if "Nacionality" in df.columns and "Nationality" not in df.columns:
        df.rename(columns={"Nacionality": "Nationality"}, inplace=True)

    # Change to numeric columns
    numeric_like_cols = [
        "Application order",
        "Age at enrollment",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]
    for col in numeric_like_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing values
    print("\n=== Missing values before ===")
    print(df.isna().sum().sort_values(ascending=False).head(25))

    # Imputing missing vals in columns
    num_cols = [
        c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_cols = [c for c in df.columns if c != target and df[c].dtype == "object"]

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isna().any():
            mode_vals = df[c].mode(dropna=True)
            if not mode_vals.empty:
                df[c] = df[c].fillna(mode_vals.iloc[0])

    # Dropping duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} duplicate rows.")

    # Remove numeric outliers via z-score
    if outlier_z is not None and len(num_cols) > 0:
        from scipy import stats  # local import to keep top imports tidy

        z = np.abs(stats.zscore(df[num_cols], nan_policy="omit"))
        keep = (np.isnan(z) | (z < outlier_z)).all(axis=1)
        kept = int(keep.sum())
        print(f"Outlier filter: kept {kept}/{len(df)} rows (z < {outlier_z}).")
        df = df.loc[keep].copy()

    # Encode target to numeric
    if df[target].dtype == "object":
        mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}

        if set(df[target].unique()).issubset(set(mapping.keys())):
            df[target] = df[target].map(mapping)

    # Report missing
    print("\n=== Missing values after ===")
    print(df.isna().sum().sort_values(ascending=False).head(25))

    print(f"\nCleaned shape: {df.shape}")
    print(f"\n{target} distribution:")
    print(df[target].value_counts(dropna=False))

    return df


# --------------------------- Data Transformation ---------------------------


def engineer_features(df: pd.DataFrame, target: str = "Target") -> pd.DataFrame:

    df = df.copy()

    # Approved/Enrolled ratios per semester
    if {
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (enrolled)",
    }.issubset(df.columns):
        denom = df["Curricular units 1st sem (enrolled)"].replace({0: np.nan})
        df["first_sem_pass_rate"] = df["Curricular units 1st sem (approved)"] / denom

    if {
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (enrolled)",
    }.issubset(df.columns):
        denom2 = df["Curricular units 2nd sem (enrolled)"].replace({0: np.nan})
        df["second_sem_pass_rate"] = df["Curricular units 2nd sem (approved)"] / denom2

    # Total approved & average grade across semesters
    if {
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
    }.issubset(df.columns):
        df["total_approved"] = (
            df["Curricular units 1st sem (approved)"]
            + df["Curricular units 2nd sem (approved)"]
        )

    if {
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
    }.issubset(df.columns):
        df["avg_grade"] = df[
            ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]
        ].mean(axis=1)

    # Fill any new NaN created
    for col in ["first_sem_pass_rate", "second_sem_pass_rate"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


# --------------------------- Machine Learning ---------------------------


def train_model(df):

    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc * 100, 2), "%")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (RF)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Feature importance
    importances = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color="skyblue")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance for Predicting Student Outcome")
    plt.tight_layout()
    plt.show()

    return model


# 5) ----------------- Visualization & Storytelling -----------------
def make_visuals(df: pd.DataFrame, target: str = "Target"):
    Path("outputs").mkdir(exist_ok=True)

    # (a) Target distribution
    plt.figure(figsize=(6, 4))
    order = df[target].value_counts().index
    sns.countplot(data=df, x=target, order=order)
    plt.title("Target Distribution (0=Dropout, 1=Enrolled, 2=Graduate)")
    plt.tight_layout()
    plt.savefig("outputs/target_distribution.png", dpi=150)
    print("Saved: outputs/target_distribution.png")

    # (b) Top 10 correlations with target
    num_df = df.select_dtypes(include=[np.number])
    if target in num_df.columns:
        corr = num_df.corr(numeric_only=True)[target].drop(labels=[target])
        top10 = corr.abs().nlargest(10).index
        plt.figure(figsize=(8, 5))
        corr[top10].plot(kind="bar")
        plt.title("Top 10 Feature Correlations with Target")
        plt.ylabel("Correlation")
        plt.tight_layout()
        plt.savefig("outputs/top10_correlations.png", dpi=150)
        print("Saved: outputs/top10_correlations.png")

    # (c) Age at enrollment vs Target
    if "Age at enrollment" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target, y="Age at enrollment")
        plt.title("Age at Enrollment vs Target")
        plt.tight_layout()
        plt.savefig("outputs/age_vs_target.png", dpi=150)
        print("Saved: outputs/age_vs_target.png")

    # (d) Tuition fees vs Target
    if "Tuition fees up to date" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Tuition fees up to date", hue=target)
        plt.title("Tuition Fees Status vs Target")
        plt.tight_layout()
        plt.savefig("outputs/tuition_vs_target.png", dpi=150)
        print("Saved: outputs/tuition_vs_target.png")

    # (e) Scholarship vs Target
    if "Scholarship holder" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Scholarship holder", hue=target)
        plt.title("Scholarship Holder vs Target")
        plt.tight_layout()
        plt.savefig("outputs/scholarship_vs_target.png", dpi=150)
        print("Saved: outputs/scholarship_vs_target.png")

    # (f) Correlation heatmap
    if len(num_df.columns) > 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(num_df.corr(numeric_only=True), cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap of Numeric Features")
        plt.tight_layout()
        plt.savefig("outputs/correlation_heatmap.png", dpi=150)
        print("Saved: outputs/correlation_heatmap.png")


# ---------- Using Polars ----------


def load_data_polars(csv_path: str) -> pl.DataFrame:
    return pl.read_csv(csv_path)


def clean_and_engineer_polars(df_pl: pl.DataFrame) -> pl.DataFrame:

    # Rename 'Nacionality' to 'Nationality'
    if "Nacionality" in df_pl.columns and "Nationality" not in df_pl.columns:
        df_pl = df_pl.rename({"Nacionality": "Nationality"})

    # Target mapping (only if it's text)
    if "Target" in df_pl.columns and df_pl.schema["Target"] == pl.Utf8:
        mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
        df_pl = df_pl.with_columns(
            pl.col("Target").map_elements(
                lambda x: mapping.get(x, None), return_dtype=pl.Int64
            )
        )

    # Feature engineering
    df_pl = df_pl.with_columns(
        [
            # Pass rates per semester
            (
                pl.col("Curricular units 1st sem (approved)")
                / pl.col("Curricular units 1st sem (enrolled)").replace(0, None)
            )
            .fill_null(0.0)
            .alias("first_sem_pass_rate"),
            (
                pl.col("Curricular units 2nd sem (approved)")
                / pl.col("Curricular units 2nd sem (enrolled)").replace(0, None)
            )
            .fill_null(0.0)
            .alias("second_sem_pass_rate"),
            # Totals
            (
                pl.col("Curricular units 1st sem (approved)")
                + pl.col("Curricular units 2nd sem (approved)")
            ).alias("total_approved"),
            # Average grade
            (
                (
                    pl.col("Curricular units 1st sem (grade)")
                    + pl.col("Curricular units 2nd sem (grade)")
                )
                / 2
            ).alias("avg_grade"),
        ]
    )

    # Drop duplicates
    df_pl = df_pl.unique()

    return df_pl


# ---------- Comparing Polars vs Pandas ----------


def polars_vs_pandas(csv_path: str = "dataset.csv"):

    # pandas

    t0 = time.time()
    df_pd = pd.read_csv(csv_path)
    t_pd_load = time.time() - t0

    t0 = time.time()
    df_pd = clean_data(df_pd)
    df_pd = engineer_features(df_pd)
    t_pd_clean = time.time() - t0

    t0 = time.time()
    _ = df_pd.groupby("Target")[["first_sem_pass_rate", "second_sem_pass_rate"]].mean(
        numeric_only=True
    )
    t_pd_groupby = time.time() - t0

    # polars
    t0 = time.time()
    df_pl = load_data_polars(csv_path)
    t_pl_load = time.time() - t0

    t0 = time.time()
    df_pl = clean_and_engineer_polars(df_pl)
    t_pl_clean = time.time() - t0

    t0 = time.time()
    _ = df_pl.group_by("Target").agg(
        [
            pl.col("first_sem_pass_rate").mean().alias("first_sem_pass_rate"),
            pl.col("second_sem_pass_rate").mean().alias("second_sem_pass_rate"),
        ]
    )
    t_pl_groupby = time.time() - t0

    # Summary
    total_pd = t_pd_load + t_pd_clean + t_pd_groupby
    total_pl = t_pl_load + t_pl_clean + t_pl_groupby

    print(f"Load:    pandas {t_pd_load:.4f}s | polars {t_pl_load:.4f}s")
    print(f"Clean:   pandas {t_pd_clean:.4f}s | polars {t_pl_clean:.4f}s")
    print(f"GroupBy: pandas {t_pd_groupby:.4f}s | polars {t_pl_groupby:.4f}s")
    print(f"TOTAL:   pandas {total_pd:.4f}s | polars {total_pl:.4f}s")
    print(
        "Result:  "
        + (
            "Polars faster this run"
            if total_pl < total_pd
            else "Pandas faster this run"
        )
    )


# ----------------- Main function -----------------


def main():
    df_raw = load_data("dataset.csv")
    df_clean = clean_data(df_raw, target="Target", outlier_z=3.5)
    df_feat = engineer_features(df_clean, target="Target")

    cat_cols = [
        c for c in df_feat.columns if df_feat[c].dtype == "object" and c != "Target"
    ]
    if cat_cols:
        df_feat = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)

    df_feat = df_feat.dropna(subset=["Target"])

    make_visuals(df_feat, target="Target")
    _ = train_model(df_feat)

    # ----- Performance comparison: Pandas vs Polars -----
    print("\n=== Polars vs Pandas ===")
    polars_vs_pandas(csv_path="dataset.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
