import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.utils.data import safe
from src.utils.paths import PROJECT_ROOT

SUBDIR = {'syn':'Synthetic','nys':'NY_Subway','covid':'JapanCOVID19'}
DISPLAY = {'syn':'Synthetic','nys':'New York subway','covid':'Japan COVID-19'}
SCALED = ('nys', 'covid')       # these two have large raw values -> reported as RMSE/100

# methods grouped as in Table I of the paper (batch / static / online / unsupervised); a
# horizontal rule is drawn between groups, and best/2nd are highlighted within each group.
groups_all = [
    [('OGTR','B: OGTR'), ('TRSS','B: TRSS'), ('Tikhonov','B: Tikhonov'),
     ('Proposed(batch)','B: \\textbf{GLASS (batch)}')],
    [('GCN','S: GCN'), ('ChebNet','S: ChebNet'), ('LR','S: LR')],
    [('OGTR_online','O: OGTR (online)'), ('Proposed(online)','O: \\textbf{GLASS (online)}')],
    [('Inpainting_DAU','U: Inpainting\\_DAU'), ('Proposed(unsupervised)','U: \\textbf{GLASS (unsupervised)}')],
]


def rows_of(dataset, task):
    """RMSE rows of one dataset: (row labels, group row ranges, sweep values, cell strings)."""
    npz_dir = PROJECT_ROOT / 'out/npz' / dataset

    def has_npz(m):
        return bool(glob.glob(str(npz_dir / safe(m) / f"{task}_*.npz")))

    # keep only the methods actually run for this dataset/task (e.g. GNN baselines may be absent)
    groups = [[(m, lbl) for (m, lbl) in g if has_npz(m)] for g in groups_all]
    groups = [g for g in groups if g]
    index = [m for g in groups for (m, _) in g]
    labels = [lbl for g in groups for (_, lbl) in g]
    if not index:
        raise FileNotFoundError(f"no restored npz under {npz_dir} for task '{task}'. Run src/eval/compare_methods.py first.")

    bounds, start = [], 0
    for g in groups:
        bounds.append((start, start + len(g)))
        start += len(g)

    files = sorted(glob.glob(str(npz_dir / safe(index[0]) / f"{task}_*.npz")))
    by_v = defaultdict(list)
    for f in files:
        _, tag, idx = Path(f).stem.split('_')
        by_v[tag].append(idx)
    tags = sorted(by_v)            # zero-padded -> lexical sort == numeric order

    _gt = {}
    def gt_of(idx):
        if idx not in _gt:
            _gt[idx] = np.load(PROJECT_ROOT / 'datasets' / SUBDIR[dataset] / f'data_{idx}.npz')['groundtruth']
        return _gt[idx]

    # RMSE: mean over data indices of per-index RMSE
    matrix = np.array([[np.mean([np.sqrt(np.mean((np.load(npz_dir / safe(m) / f"{task}_{tag}_{idx}.npz")['restored']
                                                  - gt_of(idx))**2)) for idx in by_v[tag]])
                        for tag in tags] for m in index])
    if dataset in SCALED:
        matrix = matrix / 100
    matrix = np.round(matrix, decimals=2)

    ncol = len(tags)
    cells = [[f'{float(v):.02f}' for v in row] for row in matrix]

    def decorate(rows, best_cmd, second_cmd):
        """Mark the best and the second best of ``rows``. The ranking uses the printed (rounded)
        value, so methods that coincide there get the same mark."""
        for k in range(ncol):
            col = sorted({matrix[r, k] for r in rows})
            for r in rows:
                if matrix[r, k] == col[0]:
                    cells[r][k] = '\\' + best_cmd + '{' + cells[r][k] + '}'
                elif len(col) > 1 and matrix[r, k] == col[1]:
                    cells[r][k] = '\\' + second_cmd + '{' + cells[r][k] + '}'

    # best (bold) / 2nd (underline) within each group
    for (a, b) in bounds:
        decorate(range(a, b), 'textbf', 'underline')
    # best (red) / 2nd (blue) over all the methods
    decorate(range(len(index)), 'textcolor{red}', 'textcolor{blue}')

    return labels, bounds, [int(t)/100 for t in tags], cells



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', help="'syn', 'nys', 'covid' (several -> one tabular, one block each)")
    parser.add_argument('--task', default='ratio', help="'ratio' or 'noise'")
    args = parser.parse_args()

    blocks = [(ds,) + rows_of(ds, args.task) for ds in args.datasets]
    ncol = len(blocks[0][3])
    symbol = '\\psi' if args.task == 'ratio' else '\\eta'

    lines = ['\\begin{tabular}{l||' + 'c'*ncol + '}', '\\hline']
    lines.append('Methods $\\backslash$ Metrics & $' + symbol + '=' + str(blocks[0][3][0]) + '$ & '
                 + ' & '.join(str(c) for c in blocks[0][3][1:]) + ' \\\\ \\hline\\hline')
    for bi, (dataset, labels, bounds, _, cells) in enumerate(blocks):
        head = f'({chr(97+bi)}) {DISPLAY[dataset]}'
        if dataset in SCALED:
            head += ' (RMSE $/100$)'
        lines.append('\\multicolumn{' + str(ncol+1) + '}{c}{' + head + '} \\\\ \\hline')
        for gi, (a, b) in enumerate(bounds):
            for r in range(a, b):
                last_of_block = (gi == len(bounds) - 1 and r == b - 1)
                sep = ' \\\\ \\hline' if (r == b - 1 and not last_of_block) else ' \\\\'
                lines.append(labels[r] + ' & ' + '&'.join(cells[r]) + sep)
        lines.append('\\hline')
    lines.append('\\end{tabular}')
    text = '\n'.join(l for l in lines if l != '')

    name = f'{args.task}_all' if len(args.datasets) > 1 else f'{args.datasets[0]}_{args.task}'
    out_path = PROJECT_ROOT / 'out/table' / f'{name}.tex'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + '\n')
    print(text)
    print(f'\n-> saved {out_path}', flush=True)


if __name__ == '__main__':
    main()
