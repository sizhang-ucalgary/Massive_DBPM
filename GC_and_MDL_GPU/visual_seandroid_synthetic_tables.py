import csv
import os
from collections import defaultdict

# Configuration
CSV_SEANDROID = 'Results_SEAndroid.csv'
CSV_SYNTHETIC = 'Results_Synthetic.csv'
OUTPUT_TEX    = 'Tables_Results.tex'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(val, digits=4):
    """Format a numeric string to fixed decimal places."""
    try:
        return f"{float(val):.{digits}f}"
    except (ValueError, TypeError):
        return str(val)

def fmt_int(val):
    """Format domain count as a comma-separated integer."""
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)

def strip_dot(val):
    """Strip trailing dots from values like '0.3.'."""
    return str(val).rstrip('.')

def load_csv(path):
    """Return list of row-dicts from a CSV, skipping blank lines."""
    with open(path, newline='') as f:
        return [r for r in csv.DictReader(f) if any(v.strip() for v in r.values())]

# ---------------------------------------------------------------------------
# Table body builders
# ---------------------------------------------------------------------------

def build_seandroid_body(rows):
    """
    Tabular body for tab:results-seandroid.
    Partial: one GC row per ps value (pn=0.0), sorted by ps.
    Noise:   one MDL row per pn value (ps=0.1), sorted by pn.
    """
    lines = []

    partial_rows = sorted([r for r in rows if r['type'] == 'Partial'],
                         key=lambda r: float(strip_dot(r['ps'])))
    noise_rows   = sorted([r for r in rows if r['type'] == 'Noise'],
                         key=lambda r: (float(strip_dot(r['ps'])), float(strip_dot(r['pn']))))

    def data_cols(r):
        return (strip_dot(r['ps']), strip_dot(r['pn']), r['method'],
                fmt(r['accuracy']), fmt(r['precision']),
                fmt(r['recall']),   fmt(r['f1']),
                fmt_int(r['domains']), fmt(r['time'], 2))

    # Partial block
    for i, r in enumerate(partial_rows):
        ps, pn, method, acc, prec, rec, f1, dom, t = data_cols(r)
        type_col = f"\\multirow{{{len(partial_rows)}}}{{*}}{{Partial}}" if i == 0 else ""
        lines.append(f"        {type_col} & {ps} & {pn} & {method} "
                     f"& {acc} & {prec} & {rec} & {f1} & {dom} & {t} \\\\")

    lines.append("        \\midrule")

    # Noise block
    for i, r in enumerate(noise_rows):
        ps, pn, method, acc, prec, rec, f1, dom, t = data_cols(r)
        type_col = f"\\multirow{{{len(noise_rows)}}}{{*}}{{Noise}}" if i == 0 else ""
        lines.append(f"        {type_col} & {ps} & {pn} & {method} "
                     f"& {acc} & {prec} & {rec} & {f1} & {dom} & {t} \\\\")

    return "\n".join(lines)


def build_synthetic_body(rows):
    """
    Tabular body for tab:results-synthetic.
    Partial: 3 methods (GC, DT, MLP) per (ps, pn=0.0), separated by cmidrule.
    Noise:   3 methods (MDL, DT, MLP) per (ps, pn),    separated by cmidrule.
    """
    lines = []
    METHOD_ORDER = {'Partial': ['GC', 'DT', 'MLP'], 'Noise': ['MDL', 'DT', 'MLP']}

    for scenario in ['Partial', 'Noise']:
        scen_rows = [r for r in rows if r['type'] == scenario]
        morder    = METHOD_ORDER[scenario]

        # Collect ordered (ps, pn) groups
        groups, order = defaultdict(list), []
        for r in scen_rows:
            key = (strip_dot(r['ps']), strip_dot(r['pn']))
            if key not in groups:
                order.append(key)
            groups[key].append(r)

        # Sort each group by preferred method order
        for key in order:
            groups[key].sort(key=lambda r: morder.index(r['method'])
                             if r['method'] in morder else 99)

        n_total = sum(len(groups[k]) for k in order)
        first_of_scenario = True

        for g_idx, (ps, pn) in enumerate(order):
            group      = groups[(ps, pn)]
            n_in_group = len(group)

            for row_idx, r in enumerate(group):
                acc    = fmt(r['accuracy']);   prec = fmt(r['precision'])
                rec    = fmt(r['recall']);     f1   = fmt(r['f1'])
                dom    = fmt_int(r['domains']); t   = fmt(r['time'], 2)
                method = r['method']

                # Type column: big multirow for first row of entire scenario
                if first_of_scenario and row_idx == 0:
                    col_type = f"\\multirow{{{n_total}}}{{*}}{{{scenario}}}"
                    first_of_scenario = False
                else:
                    col_type = ""

                # ps / pn columns: multirow for first row of each (ps, pn) group
                col_ps = f"\\multirow{{{n_in_group}}}{{*}}{{{ps}}}" if row_idx == 0 else ""
                col_pn = f"\\multirow{{{n_in_group}}}{{*}}{{{pn}}}" if row_idx == 0 else ""

                if row_idx == 0:
                    lines.append(f"        {col_type} & {col_ps} & {col_pn}")
                    lines.append(f"               & {method} "
                                 f"& {acc} & {prec} & {rec} & {f1} & {dom} & {t} \\\\")
                else:
                    lines.append(f"         &  &  & {method} "
                                 f"& {acc} & {prec} & {rec} & {f1} & {dom} & {t} \\\\")

            # Separator between (ps, pn) groups, not after the last
            if g_idx < len(order) - 1:
                lines.append("         \\cmidrule{2-10}")

        # Separator between Partial and Noise
        if scenario == 'Partial':
            lines.append("        \\midrule")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Document generator  (same pattern as visual_skewness_tikzpicture.py)
# ---------------------------------------------------------------------------

def generate_tex(rows_sea, rows_syn):
    """Build the full TeX document string with top/bottom placement and labels."""
    lines = []

    # ── Preamble ──
    lines.append(r'\documentclass{article}')
    lines.append(r'\usepackage{booktabs, multirow, array, caption, graphicx}')
    lines.append(r'\usepackage[margin=1in]{geometry}') # Ensures tables fit on page
    lines.append(r'\begin{document}')
    lines.append(r'')

    # ── Table 1: SEAndroid (Top) ──
    lines.append(r'\begin{table}[t]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{SEAndroid Dataset Results}')
    lines.append(r'    \label{tab:results-seandroid}')
    lines.append(r'    \resizebox{\textwidth}{!}{%') # Optional: scales table to fit width
    lines.append(r'    \begin{tabular}{@{}ccc|lccccccc@{}}')
    lines.append(r'        \toprule')
    lines.append(r'        \textbf{Type} & \textbf{$p_s$} & \textbf{$p_n$}'
                 r' & \textbf{Method} & \textbf{Acc} & \textbf{Prec}'
                 r' & \textbf{Rec} & \textbf{F1} & \textbf{Domains} & \textbf{Time (s)} \\')
    lines.append(r'        \midrule')
    lines.append(build_seandroid_body(rows_sea))
    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}%')
    lines.append(r'    }')
    lines.append(r'\end{table}')
    
    lines.append(r'')
    lines.append(r'\vfill') # Pushes Table 2 to the bottom
    lines.append(r'')

    # ── Table 2: Synthetic (Bottom) ──
    lines.append(r'\begin{table}[b]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{Synthetic Dataset Results}')
    lines.append(r'    \label{tab:results-synthetic}')
    lines.append(r'    \resizebox{\textwidth}{!}{%') # Optional: scales table to fit width
    lines.append(r'    \begin{tabular}{@{}ccc|lccccccc@{}}')
    lines.append(r'        \toprule')
    lines.append(r'        \textbf{Type} & \textbf{$p_s$} & \textbf{$p_n$}'
                 r' & \textbf{Method} & \textbf{Acc} & \textbf{Prec}'
                 r' & \textbf{Rec} & \textbf{F1} & \textbf{Domains} & \textbf{Time (s)} \\')
    lines.append(r'        \midrule')
    lines.append(build_synthetic_body(rows_syn))
    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}%')
    lines.append(r'    }')
    lines.append(r'\end{table}')

    lines.append(r'')
    lines.append(r'\end{document}')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for path in [CSV_SEANDROID, CSV_SYNTHETIC]:
        if not os.path.exists(path):
            print(f'Error: {path} not found in the current directory.')
            return

    rows_sea = load_csv(CSV_SEANDROID)
    rows_syn = load_csv(CSV_SYNTHETIC)

    tex_content = generate_tex(rows_sea, rows_syn)

    with open(OUTPUT_TEX, 'w') as f:
        f.write(tex_content)

    print(f'[OK] Written to: {OUTPUT_TEX}')
    print(f'     Table 1 (SEAndroid):  {len(rows_sea)} rows')
    print(f'     Table 2 (Synthetic):  {len(rows_syn)} rows')
    print(f'')
    print(f'Compile with:  pdflatex {OUTPUT_TEX}')


if __name__ == '__main__':
    main()
