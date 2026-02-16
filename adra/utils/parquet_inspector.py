"""
Parquet inspection utilities for loading, summarizing, and visualizing Parquet files.

Basic usage
-----------
As a script (inspect a file from the command line):

    python -m adra.utils.parquet_inspector path/to/file.parquet

  Export preview to JSON (first N rows, default 5):

    python -m adra.utils.parquet_inspector path/to/file.parquet -o preview.json
    python -m adra.utils.parquet_inspector path/to/file.parquet --save-json preview.json --rows 10

  Other options:

    python -m adra.utils.parquet_inspector path/to/file.parquet --rows 10 --visualize

As a library:

    from adra.utils.parquet_inspector import load_parquet, inspect_parquet

    # Load only (optionally subset columns):
    df = load_parquet("data.parquet")
    df = load_parquet("data.parquet", columns=["col_a", "col_b"])

    # Load and print shape, dtypes, stats, and a preview:
    df = inspect_parquet("data.parquet", n_rows=5)
    # With pairplot and JSON export:
    df = inspect_parquet("data.parquet", n_rows=10, visualize=True, save_json_path="out.json")
"""
import os
from typing import Optional, Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_parquet(path: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load a Parquet file into a pandas DataFrame.

    Args:
        path: Path to the parquet file on disk.
        columns: Optionally restrict the loaded columns to this subset.

    Returns:
        A ``pd.DataFrame`` containing the parquet data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path, columns=columns)
    return df


def inspect_parquet(
    path: str,
    n_rows: int = 5,
    visualize: bool = False,
    save_json_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load and inspect a parquet file.

    The function prints a high-level summary including the DataFrame shape, column
    dtypes, basic statistics (for numeric columns) and a preview of the first
    *n_rows* rows.

    Optionally, a pairplot of the *numeric* columns is shown to give a quick
    visual overview of the data distribution.

    Args:
        path: Parquet file path.
        n_rows: Number of rows to print for preview (default: 5).
        visualize: If ``True`` and the environment supports a display, a seaborn
            pairplot is rendered for numeric columns. For wide tables (>10
            numeric columns) only the first 10 are plotted to avoid overly busy
            figures.

    Returns:
        The loaded ``pd.DataFrame`` so that callers can further manipulate it if
        desired.
    """
    df = load_parquet(path)

    print("=" * 80)
    print(f"Loaded parquet file: {path}")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")
    print("-" * 80)
    print("Column dtypes:")
    print(df.dtypes)
    print("-" * 80)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        print("Basic numeric statistics (first 10 numeric cols):")
        print(df[numeric_cols[:10]].describe())
        print("-" * 80)

    print(f"Data preview (first {n_rows} rows):")
    print(df.head(n_rows))
    print("=" * 80)

    # Save preview to JSON if requested
    if save_json_path is not None:
        try:
            df.head(n_rows).to_json(save_json_path, orient="records", lines=True, force_ascii=False)
            print(f"Saved first {n_rows} rows to {save_json_path}")
        except Exception as e:
            print(f"[WARN] Could not save JSON preview to {save_json_path}: {e}")

    # Visualization section
    if visualize and len(numeric_cols) > 1:
        cols_to_plot = numeric_cols[:10]  # limit to at most 10 to keep the plot readable
        print(f"Generating pairplot for {len(cols_to_plot)} numeric columns …")
        sns.pairplot(df[cols_to_plot].sample(min(1000, len(df))), corner=True)
        plt.suptitle("Parquet numeric column pairplot", y=1.02)
        plt.tight_layout()
        plt.show()

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a Parquet file – prints schema, stats and a data preview."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the Parquet file to inspect",
    )
    parser.add_argument(
        "--rows",
        "-n",
        type=int,
        default=5,
        help="Number of rows to show in the preview (default: 5)",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Generate a Seaborn pairplot for numeric columns",
    )
    parser.add_argument(
        "--save-json",
        "-o",
        type=str,
        help="Optional path to save the first N rows to a JSON file",
    )

    parsed_args = parser.parse_args()
    inspect_parquet(
        parsed_args.path,
        n_rows=parsed_args.rows,
        visualize=parsed_args.visualize,
        save_json_path=parsed_args.save_json,
    ) 