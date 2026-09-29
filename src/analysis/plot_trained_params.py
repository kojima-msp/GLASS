"""Trained parameters of GLASS (batch): the FIR taps, the regularization weight, and the
spectral response of the Sobolev operator of each layer.

    uv run python src/analysis/plot_trained_params.py dataset=covid
"""
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the 'science' styles
import seaborn as sns
import torch

plt.style.use(['science','ieee','high-vis'])
for _axis in ('xtick', 'ytick'):     # the science style draws minor ticks as well
    plt.rcParams[f'{_axis}.major.size'] = 0
    plt.rcParams[f'{_axis}.minor.size'] = 0
    plt.rcParams[f'{_axis}.minor.visible'] = False

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra

from src.utils.data import DataLoader
from src.utils.paths import PROJECT_ROOT

LAYERS = [0, 12, 25, 37, 49]     # representative layers shown in the response panel (K = 50)


def plot_params(dataset, model_name, K, n_taps):
    model = torch.load(model_name, map_location='cpu')
    taps = np.stack([torch.softmax(model[f'd_h.{k}'][1:n_taps+1], dim=0).numpy() for k in range(K)], axis=1)
    alpha = 10.0 ** np.array([float(model[f'lambda_.{k}']) for k in range(K)])
    eps = np.array([float(torch.exp(model[f'eps_.{k}'])) for k in range(K)])
    beta = np.array([float(torch.exp(model[f'beta_.{k}'])) for k in range(K)])

    lam_max = float(np.linalg.eigvalsh(DataLoader(dataset, idx=10).L).max())
    lam = np.linspace(0, lam_max, 21)

    # narrow: the three datasets are placed side by side, so the figure is used at this size
    fig = plt.figure(figsize=(2.6, 5))
    gs = fig.add_gridspec(5, 1, hspace=0.9)
    ax = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[3:, 0])]

    sns.heatmap(taps, cmap='turbo', vmin=0, vmax=1, ax=ax[0], xticklabels=False,
                cbar_kws={'pad': 0.02})
    ax[0].set_yticks(np.arange(n_taps) + 0.5, range(1, n_taps+1), rotation=0)
    ax[0].set_ylabel(r'$d_l^{(k)}$')
    ax[0].set_xticks(np.arange(0, K, 10) + 0.5, np.arange(0, K, 10), rotation=0)
    ax[0].set_xlabel(r'$k$')

    # half a cell of margin on both sides lines this panel up with the heatmap above
    ax[1].plot(np.arange(K), alpha, marker='o', markersize=1.5)
    ax[1].set_xlim([-0.5, K-0.5])
    alpha_top = np.ceil(alpha.max()*4)/4          # round up to a multiple of the tick spacing
    ax[1].set_ylim([0, alpha_top])
    ax[1].set_yticks(np.arange(0, alpha_top + 1e-9, 0.25))
    ax[1].set_xlabel(r'$k$')
    ax[1].set_ylabel(r'$\alpha^{(k)}$')
    ax[1].grid()

    # response of the regularizer, normalized so that every curve is one at the largest eigenvalue
    ax[2].plot(lam/lam.max(), lam/lam.max(), 'k--', label=r'$\mathbf{L}$')
    for k, marker in zip(LAYERS, ['o', 'v', '^', 's', 'x'], strict=True):
        r = (lam + eps[k])**beta[k]
        ax[2].plot(lam/lam.max(), r/r.max(), marker=marker, markersize=2, label=rf'$k={k}$')
    ax[2].set_xlim([0, 1])
    ax[2].set_ylim([0, 1.02])
    ax[2].set_xlabel(r'$\lambda / \lambda_{\max}$')
    ax[2].set_ylabel('Normalized response')
    ax[2].legend(fontsize=5.5, ncol=2, loc='lower right', handlelength=1.6,
                 columnspacing=0.8, labelspacing=0.25, borderpad=0.3, framealpha=0.9)
    ax[2].grid()

    # the colorbar narrows the heatmap, so give the alpha panel the same width
    fig.canvas.draw()
    pos_heat, pos_alpha = ax[0].get_position(), ax[1].get_position()
    ax[1].set_position([pos_alpha.x0, pos_alpha.y0, pos_heat.width, pos_alpha.height])

    out_path = PROJECT_ROOT / 'out/img/trained_params' / f'{dataset}.pdf'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'-> saved {out_path}', flush=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    plot_params(cfg.dataset, cfg.proposed_parameter_path, cfg.K, cfg.proposed_L)


if(__name__=='__main__'):
    main()
