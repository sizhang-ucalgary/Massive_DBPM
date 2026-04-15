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
    """Return list of row-dicts from a CSV, normalized."""
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if any(v.strip() for v in r.values())]
    
    # Normalize 'complexity' to 'domains'
    for r in rows:
        if 'complexity' in r:
            r['domains'] = r['complexity']
        # Normalize types for consistency
        if r['type'] == 'Clean': 
            r['type'] = 'Partial'
        if r['type'] == 'Noisy': 
            r['type'] = 'Noise'
            
    return rows

def aggregate_rows(rows):
    """Group rows by (type, ps, pn, method) and calculate mean/std stats."""
    keys = ['type', 'ps', 'pn', 'method']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'time', 'domains']
    
    groups = defaultdict(list)
    for r in rows:
        key = (r['type'], strip_dot(r['ps']), strip_dot(r['pn']), r['method'])
        groups[key].append(r)
        
    agg_results = []
    for key, g_rows in groups.items():
        res = {
            'type': key[0],
            'ps': key[1],
            'pn': key[2],
            'method': key[3]
        }
        
        for m in metrics:
            vals = [float(r[m]) for r in g_rows]
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            res[m] = (mean, std)
            
        agg_results.append(res)
        
    return agg_results

def fmt_metric(stats, digits=4):
    """Format metric as just mean (as requested)."""
    mean, std = stats
    return f"{mean:.{digits}f}"

def fmt_dom_stats(stats):
    r"""Format domain as mean \pm std where std is an integer."""
    mean, std = stats
    m_int = int(round(mean))
    s_int = int(round(std))
    if s_int == 0:
        return f"{m_int:,}"
    return f"${m_int:,} \\pm {s_int}$"

# ---------------------------------------------------------------------------
# Table body builders
# ---------------------------------------------------------------------------

def build_seandroid_body(all_rows):
    """
    Tabular body for tab:results-seandroid.
    Aggregates multiple runs per instance.
    """
    rows = aggregate_rows(all_rows)
    lines = []

    partial_rows = sorted([r for r in rows if r['type'] == 'Partial'],
                         key=lambda r: float(r['ps']))
    noise_rows   = sorted([r for r in rows if r['type'] == 'Noise'],
                         key=lambda r: (float(r['ps']), float(r['pn'])))

    def row_to_cols(r):
        return (r['ps'], r['pn'], r['method'],
                fmt_metric(r['accuracy']), fmt_metric(r['precision']),
                fmt_metric(r['recall']),   fmt_metric(r['f1']),
                fmt_dom_stats(r['domains']), fmt_metric(r['time'], 2))

    # Partial block
    for i, r in enumerate(partial_rows):
        ps, pn, method, acc, prec, rec, f1, dom, t = row_to_cols(r)
        type_col = f"\\multirow{{{len(partial_rows)}}}{{*}}{{Partial}}" if i == 0 else ""
        lines.append(f"        {type_col} & {ps} & {pn} & {method} "
                     f"& {acc} & {prec} & {rec} & {f1} & {dom} & {t} \\\\")

    lines.append("        \\midrule")

    # Noise block
    for i, r in enumerate(noise_rows):
        ps, pn, method, acc, prec, rec, f1, dom, t = row_to_cols(r)
        type_col = f"\\multirow{{{len(noise_rows)}}}{{*}}{{Noise}}" if i == 0 else ""
        lines.append(f"        {type_col} & {ps} & {pn} & {method} "
                     f"& {acc} & {prec} & {rec} & {f1} & {dom} & {t} \\\\")

    return "\n".join(lines)


def build_synthetic_body(all_rows):
    """
    Tabular body for tab:results-synthetic.
    Aggregates multiple runs per instance.
    """
    rows = aggregate_rows(all_rows)
    lines = []
    METHOD_ORDER = {'Partial': ['GC', 'DT', 'MLP'], 'Noise': ['MDL', 'DT', 'MLP']}

    for scenario in ['Partial', 'Noise']:
        scen_rows = [r for r in rows if r['type'] == scenario]
        morder    = METHOD_ORDER[scenario]

        # Group by (ps, pn)
        groups, order = defaultdict(list), []
        for r in scen_rows:
            key = (r['ps'], r['pn'])
            if key not in groups:
                order.append(key)
            groups[key].append(r)

        # Sort each group by method order
        for key in order:
            groups[key].sort(key=lambda r: morder.index(r['method'])
                             if r['method'] in morder else 99)

        n_total = sum(len(groups[k]) for k in order)
        first_of_scenario = True

        for g_idx, (ps, pn) in enumerate(order):
            group      = groups[(ps, pn)]
            n_in_group = len(group)

            for row_idx, r in enumerate(group):
                acc    = fmt_metric(r['accuracy']);   prec = fmt_metric(r['precision'])
                rec    = fmt_metric(r['recall']);     f1   = fmt_metric(r['f1'])
                dom    = fmt_dom_stats(r['domains'])
                t      = fmt_metric(r['time'], 2)
                method = r['method']

                # Type column: multirow for entire scenario
                if first_of_scenario and row_idx == 0:
                    col_type = f"\\multirow{{{n_total}}}{{*}}{{{scenario}}}"
                    first_of_scenario = False
                else:
                    col_type = ""

                # ps / pn columns: multirow for group
                col_ps = f"\\multirow{{{n_in_group}}}{{*}}{{{ps}}}" if row_idx == 0 else ""
                col_pn = f"\\multirow{{{n_in_group}}}{{*}}{{{pn}}}" if row_idx == 0 else ""

                if row_idx == 0:
                    lines.append(f"        {col_type} & {col_ps} & {col_pn}")
                    lines.append(f"               & {method} "
                                 f"& {acc} & {prec} & {rec} & {f1} & {dom} & {t} \\\\")
                else:
                    lines.append(f"         &  &  & {method} "
                                 f"& {acc} & {prec} & {rec} & {f1} & {dom} & {t} \\\\")

            if g_idx < len(order) - 1:
                lines.append("         \\cmidrule{2-10}")

        if scenario == 'Partial':
            lines.append("        \\midrule")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Document generator
# ---------------------------------------------------------------------------

def generate_tex(rows_sea, rows_syn):
    """Build the full TeX document string."""
    lines = []

    lines.append(r'\documentclass{article}')
    lines.append(r'\usepackage{booktabs, multirow, array, caption, graphicx}')
    lines.append(r'\usepackage[margin=1in]{geometry}')
    lines.append(r'\begin{document}')
    lines.append(r'')

    # ── Table 1: SEAndroid ──
    lines.append(r'\begin{table}[ht]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{SEAndroid Dataset Results}')
    lines.append(r'    \label{tab:results-seandroid}')
    lines.append(r'    \resizebox{\textwidth}{!}{%')
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
    lines.append(r'\newpage') # Ensure separation and visibility
    lines.append(r'')

    # ── Table 2: Synthetic ──
    lines.append(r'\begin{table}[ht]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{Synthetic Dataset Results}')
    lines.append(r'    \label{tab:results-synthetic}')
    lines.append(r'    \resizebox{\textwidth}{!}{%')
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

####################################################################################
# Usage:
# python3 visual_seandroid_synthetic_tables.py
# 
# Expects 'Results_SEAndroid.csv' and 'Results_Synthetic.csv' in the current directory.
# Generates 'Tables_Results.tex' (LaTeX).
####################################################################################
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
