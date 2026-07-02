#!/usr/bin/env python3
"""Generate a LaTeX results table from metrics and elapsed_time files.

Usage:
    python3 scripts/generate_table.py

This script scans these folders under the repo root:
  - 10_SG_MSC
  - 10_SG_SVN
  - 10_SG1_MSC
  - 10_SG1_SVN

It collects `metrics_*.txt` and `elapsed_time.txt` files for common algorithms
and writes `results_table_generated.tex` at the repo root.
"""
from pathlib import Path
import ast
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT_TEX = ROOT / "results_table_generated.tex"

PREPROCS = {
    "10_SG_MSC": "SG + MSC",
    "10_SG_SVN": "SG + SNV",
    "10_SG1_MSC": "SG1 + MSC",
    "10_SG1_SVN": "SG1 + SNV",
}

ALG_FOLDERS = [
    ("PLS", "PLS-DA"),
    ("SVM", "SVM"),
    ("ELM", "ELM"),
    ("XGBoost", "XGBoost"),
    ("RandomForest", "RandomForest"),
    ("CARS", "CARS"),
    ("GA-iPLS", "GA-iPLS"),
    ("BOSS", "BOSS"),
    ("GA-iPLS_BOSS", "iGA-BOSS"),
]

METR_RE = re.compile(r"metrics_(\d+)\.txt")

def read_metrics(path: Path):
    try:
        text = path.read_text(encoding="utf-8").strip()
        return ast.literal_eval(text)
    except Exception:
        return None

def read_elapsed(path: Path):
    try:
        text = path.read_text(encoding="utf-8").strip()
        m = re.search(r"Elapsed time:\s*(\S+)", text)
        return m.group(1) if m else "--"
    except Exception:
        return "--"

def fmt(x, nd=2):
    if x is None:
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def acc_per_wav(acc, wav):
    try:
        return f"{float(acc)/float(wav):.5f}"
    except Exception:
        return "--"

def gather():
    rows = []
    for folder, display in PREPROCS.items():
        base = ROOT / folder
        for alg_folder, alg_label in ALG_FOLDERS:
            alg_path = base / alg_folder
            if not alg_path.exists():
                # try variants with hyphens or underscores
                alg_path = base / alg_folder.replace('-', '_')
                if not alg_path.exists():
                    continue

            # find metrics file (pick the first metrics_*.txt)
            metrics_file = None
            for f in alg_path.glob('metrics_*.txt'):
                metrics_file = f
                break

            if metrics_file is None:
                continue

            m = METR_RE.search(metrics_file.name)
            wav = int(m.group(1)) if m else 2101

            metrics = read_metrics(metrics_file)
            elapsed = read_elapsed(alg_path / 'elapsed_time.txt')

            acc = metrics.get('accuracy') if metrics else None
            rec = metrics.get('recall') if metrics else None
            prec = metrics.get('precision') if metrics else None
            f1 = metrics.get('f1') if metrics else None
            spec = metrics.get('specificity') if metrics else None
            nl = metrics.get('nLV') if metrics else metrics.get('LV') if metrics else None

            # For full-spectrum classifiers (PLS, SVM) we keep LV as -- if None
            nl_display = '--' if nl is None else str(nl)

            rows.append({
                'preproc': display,
                'algo': alg_label,
                'wav': wav,
                'lv': nl_display,
                'acc': acc,
                'rec': rec,
                'prec': prec,
                'f1': f1,
                'spec': spec,
                'acc_wav': acc_per_wav(acc, wav) if acc is not None else '--',
                'time': elapsed,
            })

    # Keep ordering consistent with example: group by preprocessing, and order algs as in ALG_FOLDERS
    ordered = []
    for folder, display in PREPROCS.items():
        for _, alg_label in ALG_FOLDERS:
            for r in rows:
                if r['preproc'] == display and r['algo'] == alg_label:
                    ordered.append(r)
    return ordered

LATEX_HEADER = r"""
\begin{table}[th!]
\centering
\footnotesize
\setlength\tabcolsep{2.3pt}
\renewcommand{\arraystretch}{1.2}

\begin{tabular}{ccccccccccc}
\hline
\textbf{Preproc.} & \textbf{Algo.} & \textbf{\#Wav.} & \textbf{\#LV} & \textbf{Acc.} & \textbf{Rec.} & \textbf{Prec.} & \textbf{F1} & \textbf{Spec.} &
\begin{tabular}{c}\textbf{Acc./}\\\textbf{Wav.}\end{tabular} &
\begin{tabular}{c}\textbf{Comput.}\\\textbf{Time}\end{tabular} \\
\hline
"""

LATEX_FOOT = r"""
\hline
\end{tabular}
\caption{Test set classification performance across preprocessing pipelines and wavelength-selection strategies. 
In the table, PLS-DA, SVM, ELM, XGBoost, and Random Forest are classification models based on the full spectrum, whereas the other rows represent PLS-DA models in which the wavelengths are selected using the CARS, GA-iPLS, BOSS, and iGA-BOSS methods, respectively.
Metrics include accuracy (Acc.), recall (Rec.), precision (Prec.), F1-score (F1), specificity (Spec.), accuracy per wavelength, and computational time.}
\label{tab:results}
\end{table}
"""

def write_table(rows):
    lines = [LATEX_HEADER.strip()]
    last_pre = None
    for r in rows:
        pre = r['preproc']
        pre_display = pre if pre != last_pre else '         '
        acc_s = fmt(r['acc'], 2) if r['acc'] is not None else '--'
        rec_s = fmt(r['rec'], 2) if r['rec'] is not None else '--'
        prec_s = fmt(r['prec'], 2) if r['prec'] is not None else '--'
        f1_s = fmt(r['f1'], 2) if r['f1'] is not None else '--'
        spec_s = fmt(r['spec'], 2) if r['spec'] is not None else '--'

        line = f"{pre_display} & {r['algo']:<12} & {r['wav']:>4} & {r['lv']:>2} & {acc_s} & {rec_s} & {prec_s} & {f1_s} & {spec_s} & {r['acc_wav']} & {r['time']} \\\\"
        lines.append(line)
        last_pre = pre

    lines.append(LATEX_FOOT.strip())
    OUT_TEX.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote LaTeX table to: {OUT_TEX}")

def main():
    rows = gather()
    if not rows:
        print("No metrics found. Are you running the script from the repository root?", file=sys.stderr)
        sys.exit(1)
    write_table(rows)

if __name__ == '__main__':
    main()
