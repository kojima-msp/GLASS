"""RMSE against the sampling ratio for several numbers of learnable taps L.

Every batch model under models_dir/release is used, including the one of the other experiments.
L is read from the weights themselves, so no record of the ablation runs has to be kept.

    uv run python src/analysis/plot_rmse_each_L.py dataset=syn
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the 'science' styles
import torch
from sklearn.metrics import mean_squared_error

plt.style.use(['science','ieee','high-vis'])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = True

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra

from src.methods.proposed_batch_model import Batch_Model
from src.utils.data import DataLoader
from src.utils.device import setup_device
from src.utils.paths import PROJECT_ROOT

TEST_RANGE = range(10, 15)
MARKERS = {1: ',', 3: 'o', 5: 'v', 8: '^', 10: 'x'}


def trained_models(models_dir, dataset):
    """L -> checkpoint. L is read from the weights, since d_h holds L+1 taps."""
    models = {}
    for ck in sorted(glob.glob(f'{models_dir}/release/batch_{dataset}_L*.pth')):
        models[len(torch.load(ck, map_location='cpu')['d_h.0']) - 1] = ck
    return models


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    dataset = cfg.dataset
    models = trained_models(cfg.models_dir, dataset)

    device = setup_device()

    ratios = [r/10 for r in range(1, 10)]
    fig, ax = plt.subplots(figsize=(4,3))
    axins = ax.inset_axes([0.35, 0.55, 0.50, 0.40])
    scores = {}

    for L in sorted(models):
        model = Batch_Model(n_taps=L, layer=cfg.K).to(device)
        model.load_state_dict(torch.load(models[L], map_location=device))
        model.eval()

        score_list = []
        for k in ratios:
            data_list = DataLoader.load_range(dataset, TEST_RANGE, noise_level=max(cfg.noise_levels), sampling_rate=k)
            with torch.no_grad():
                rmse = [np.sqrt(mean_squared_error(model.forward(d.data, d.J, d.L).cpu().numpy(), d.groundtruth))
                        for d in data_list]
            score_list.append(float(np.mean(rmse)))
        scores[L] = score_list
        print(f'L={L}: ' + ' '.join(f'{s:.3f}' for s in score_list), flush=True)

        for a in (ax, axins):
            a.plot(ratios, score_list, label=rf'$L={L}$', marker=MARKERS.get(L, 'o'))

    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([cfg.get('ymin') or 0, cfg.get('ymax') or max(max(s) for s in scores.values())*1.05])

    # zoom into the tail, where the curves are indistinguishable at full scale. L = 1 is left
    # out of the range: its single tap is fixed to one, so it sits far above the others.
    x1, x2 = 0.5, 0.9
    tail = [s[ratios.index(x1):] for L, s in scores.items() if L > 1]
    y1, y2 = min(map(min, tail)), max(map(max, tail))
    pad = (y2 - y1) * 0.15
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1 - pad, y2 + pad)
    axins.tick_params(labelsize=6)
    ax.indicate_inset_zoom(axins)

    ax.set_xlabel('Sampling Ratio')
    ax.set_ylabel('RMSE')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=2)
    out_path = PROJECT_ROOT / 'out/img/rmses_each_L.pdf'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f'-> saved {out_path}', flush=True)


if(__name__=='__main__'):
    main()
