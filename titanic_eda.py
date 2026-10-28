"""Beginner-friendly exploratory analysis for Kaggle's Titanic train.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = {"Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"}
AGE_BINS = [0, 12, 17, 35, 59, float("inf")]
AGE_LABELS = ["Child (0–12)", "Teen (13–17)", "Adult (18–35)", "Middle age (36–59)", "Senior (60+)"]
PALETTE = {0: "#C56B73", 1: "#3F8F83"}


def load_data(path: str | Path = "data/train.csv") -> pd.DataFrame:
    """Read and validate the Kaggle Titanic training CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download train.csv from the Kaggle Titanic competition "
            "and place it at data/train.csv. See README.md for instructions."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Expected Kaggle Titanic columns are missing: {', '.join(sorted(missing))}")
    df = df.copy()
    df["Survival status"] = df["Survived"].map({0: "Did not survive", 1: "Survived"})
    df["Age group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    return df


def inspect_data(df: pd.DataFrame) -> dict[str, object]:
    """Return core data quality and shape summaries."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str),
        "missing": df.isna().sum().sort_values(ascending=False),
        "duplicates": int(df.duplicated().sum()),
        "survival_count": df["Survived"].value_counts().sort_index(),
    }


def group_survival_rates(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Calculate the fraction surviving within each requested group."""
    return (
        df.groupby(by, observed=False)["Survived"]
        .agg(survival_rate="mean", passengers="size", known_outcomes="count")
        .reset_index()
        .assign(survival_rate=lambda table: table["survival_rate"] * 100)
    )


def observations(df: pd.DataFrame) -> dict[str, str]:
    """Generate concise observations grounded in the current dataset values."""
    sex = group_survival_rates(df, "Sex").set_index("Sex")["survival_rate"]
    pclass = group_survival_rates(df, "Pclass").set_index("Pclass")["survival_rate"]
    age = group_survival_rates(df, "Age group").set_index("Age group")["survival_rate"].dropna()
    sex_gap = sex.get("female", float("nan")) - sex.get("male", float("nan"))
    class_gap = pclass.get(1, float("nan")) - pclass.get(3, float("nan"))
    youngest = age.idxmax() if not age.empty else "the available age groups"
    youngest_rate = age.max() if not age.empty else float("nan")
    return {
        "sex": f"In this sample, {sex.get('female', float('nan')):.1f}% of women survived versus {sex.get('male', float('nan')):.1f}% of men (a {sex_gap:.1f}-percentage-point difference).",
        "class": f"Survival was {pclass.get(1, float('nan')):.1f}% in 1st class and {pclass.get(3, float('nan')):.1f}% in 3rd class ({class_gap:.1f} percentage points apart).",
        "age": f"Among passengers with a recorded age, the highest survival rate was in {youngest} ({youngest_rate:.1f}%). Ages are missing for {df['Age'].isna().sum()} passengers, so this comparison excludes them.",
        "age_distribution": "Each histogram group is normalized separately (relative frequency); bar heights compare age distributions rather than raw group sizes.",
    }


def create_charts(df: pd.DataFrame, output_dir: str | Path = "outputs") -> list[Path]:
    """Save a survival overview, demographic survival bars and a normalized histogram."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    saved: list[Path] = []

    overall = df["Survived"].value_counts(normalize=True).reindex([0, 1]).mul(100)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(["Did not survive", "Survived"], overall.values, color=[PALETTE[0], PALETTE[1]])
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set(title="Passenger Survival in the Training Set", ylabel="Passengers (%)", xlabel="Outcome", ylim=(0, max(overall.values) * 1.2))
    saved.append(_save(fig, output_dir / "overall_survival.png"))

    for column, labels, title, filename in [
        ("Sex", None, "Survival Rate by Sex", "survival_by_sex.png"),
        ("Pclass", [1, 2, 3], "Survival Rate by Passenger Class", "survival_by_class.png"),
        ("Age group", AGE_LABELS, "Survival Rate by Age Group", "survival_by_age_group.png"),
    ]:
        rates = group_survival_rates(df, column)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        sns.barplot(data=rates, x=column, y="survival_rate", order=labels, color="#4C9A91", ax=ax)
        ax.set(title=title, xlabel=column, ylabel="Survived within group (%)", ylim=(0, 100))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)
        saved.append(_save(fig, output_dir / filename))

    age_df = df.dropna(subset=["Age"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.histplot(
        data=age_df, x="Age", hue="Survived", bins=20, stat="probability", common_norm=False,
        multiple="layer", element="step", fill=True, alpha=0.32, palette=PALETTE, ax=ax,
    )
    ax.set(title="Age Distribution by Survival Outcome", xlabel="Age (years)", ylabel="Relative frequency within outcome")
    saved.append(_save(fig, output_dir / "age_distribution_by_outcome.png"))
    return saved


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run(path: str | Path = "data/train.csv", output_dir: str | Path = "outputs"):
    df = load_data(path)
    stats = inspect_data(df)
    charts = create_charts(df, output_dir)
    return df, stats, observations(df), charts


if __name__ == "__main__":
    data, stats, notes, files = run()
    print(f"Rows: {stats['shape'][0]:,} | columns (including derived fields): {stats['shape'][1]}")
    print(f"Duplicate rows: {stats['duplicates']}")
    print("Missing values:\n" + stats["missing"].to_string())
    print("\nObservations:")
    for key, note in notes.items():
        print(f"- {note}")
    print(f"\nSaved {len(files)} charts to {Path('outputs').resolve()}")
