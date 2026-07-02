#!/usr/bin/env python3
"""
Compute the Pearson correlation between each wavelength and the binary class label,
then overlay the wavelengths selected by iGA-BOSS.

Expected input structure:
- ROOT / PIPELINE / "PLS" / "X_df.csv"
- ROOT / PIPELINE / "GA-iPLS_BOSS" / "wavelengths_*.txt" or "wavelengths.txt"
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[0]

PIPELINE = "10_SG1_MSC"

DATA_CSV = ROOT / PIPELINE / "PLS" / "X_df.csv"
WL_DIR = ROOT / PIPELINE / "GA-iPLS_BOSS"

OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)

OUT_PATH = OUT_DIR / f"pearson_class_correlation_iGA_BOSS_{PIPELINE}.pdf"
OUT_TABLE = OUT_DIR / f"pearson_class_correlation_iGA_BOSS_{PIPELINE}.csv"
OUT_SELECTED_TABLE = OUT_DIR / f"selected_wavelength_correlations_{PIPELINE}.csv"

METADATA_COLS = ["Date", "Class", "Stress_weight", "Position"]

CLASS_MAPPING = {
    "Controlled (C)": 0,
    "Stressed Water (W)": 1,
}


# ---------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------

def load_data():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    missing = [col for col in METADATA_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing metadata columns in {DATA_CSV}: {missing}")

    classes = df["Class"].astype(str).str.strip()
    y_series = classes.map(CLASS_MAPPING)

    if y_series.isna().any():
        unmapped = sorted({label for label in classes[y_series.isna()]})
        raise ValueError(
            f"Unmapped class labels found: {unmapped}. "
            f"Current CLASS_MAPPING is {CLASS_MAPPING}."
        )

    y = y_series.astype(int).to_numpy()

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"Only one class found after mapping: {np.unique(y)}. "
            "Check CLASS_MAPPING."
        )

    candidate_cols = [col for col in df.columns if col not in METADATA_COLS]

    wavelength_cols = []
    wavelengths = []

    for col in candidate_cols:
        try:
            wavelengths.append(float(col))
            wavelength_cols.append(col)
        except ValueError:
            continue

    if not wavelength_cols:
        raise ValueError("No wavelength columns detected in the input file.")

    X = df[wavelength_cols].apply(pd.to_numeric, errors="coerce")
    wavelengths = np.asarray(wavelengths, dtype=float)

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    X = X.iloc[:, order]

    if X.isna().any().any():
        X = X.fillna(X.mean())

    return X, y, wavelengths


def read_selected_wavelengths():
    if not WL_DIR.exists():
        raise FileNotFoundError(f"Selected wavelength directory not found: {WL_DIR}")

    files = sorted(WL_DIR.glob("wavelengths_*.txt"))

    if not files:
        fallback = WL_DIR / "wavelengths.txt"
        if fallback.exists():
            files = [fallback]

    if not files:
        raise FileNotFoundError(
            f"No wavelength file found in {WL_DIR}. "
            "Expected wavelengths_*.txt or wavelengths.txt."
        )

    selected_file = files[0]

    wavelengths = []
    for line in selected_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            wavelengths.append(float(line))

    if not wavelengths:
        raise ValueError(f"No wavelengths found in {selected_file}")

    return np.asarray(wavelengths, dtype=float)


def compute_correlations(X: pd.DataFrame, y: np.ndarray):
    y_series = pd.Series(y, index=X.index, name="binary_class")
    r = X.corrwith(y_series, method="pearson")
    return r.to_numpy(dtype=float)


def match_selected(wavelengths: np.ndarray, selected_raw: np.ndarray):
    matched = []

    for wl in selected_raw:
        idx = int(np.argmin(np.abs(wavelengths - wl)))
        matched.append(wavelengths[idx])

    return np.unique(np.asarray(matched, dtype=float))


def save_tables(wavelengths, r_values, selected_wavelengths):
    selected_mask = np.isclose(
        wavelengths[:, None],
        selected_wavelengths[None, :],
        rtol=0.0,
        atol=1e-6,
    ).any(axis=1)

    table = pd.DataFrame({
        "Wavelength_nm": wavelengths,
        "Pearson_r": r_values,
        "Abs_Pearson_r": np.abs(r_values),
        "Selected_by_iGA_BOSS": selected_mask,
    })

    table.to_csv(OUT_TABLE, index=False)

    selected_table = table[table["Selected_by_iGA_BOSS"]].copy()
    selected_table.to_csv(OUT_SELECTED_TABLE, index=False)


def plot_profile(wavelengths, r_values, selected_wavelengths):
    abs_r = np.abs(r_values)

    max_abs = np.nanmax(abs_r)
    p90 = np.nanpercentile(abs_r, 90)
    spectrum_color = "blue"
    percentile_color = "green"
    selected_color = "red"

    sns.set_context("paper", font_scale=1.6)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 4.8))

    spectrum_line = ax.plot(
        wavelengths,
        abs_r,
        linewidth=1.5,
        label="Full-spectrum |Pearson r| with class label",
        color=spectrum_color,
        zorder=2,
    )[0]

    percentile_line = ax.axhline(
        p90,
        linestyle="--",
        linewidth=1.0,
        label="90th percentile of |r|",
        color=percentile_color,
        zorder=1,
    )

    for wl in selected_wavelengths:
        ax.axvline(
            wl,
            linewidth=0.8,
            color=selected_color,
            alpha=0.45,
            zorder=0,
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("|Pearson r| with class label")
    ax.set_xlim(float(np.nanmin(wavelengths)), float(np.nanmax(wavelengths)))

    if np.isfinite(max_abs) and max_abs > 0:
        ax.set_ylim(0, max_abs * 1.15)
    else:
        ax.set_ylim(0, 1)

    selected_handle = Line2D([0], [0], color=selected_color, lw=1.5, alpha=0.75, label="Selected wavelengths (iGA-BOSS)")
    ax.legend(handles=[spectrum_line, percentile_line, selected_handle], title="", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(OUT_PATH), dpi=300)
    plt.close(fig)


def main():
    X, y, wavelengths = load_data()
    selected_raw = read_selected_wavelengths()
    selected_wavelengths = match_selected(wavelengths, selected_raw)

    r_values = compute_correlations(X, y)

    save_tables(wavelengths, r_values, selected_wavelengths)
    plot_profile(wavelengths, r_values, selected_wavelengths)

    print(f"Saved figure: {OUT_PATH}")
    print(f"Saved full correlation table: {OUT_TABLE}")
    print(f"Saved selected wavelength table: {OUT_SELECTED_TABLE}")


if __name__ == "__main__":
    main()