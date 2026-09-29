import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = True

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra

from src.utils.data import safe, vtag
from src.utils.paths import PROJECT_ROOT
from src.utils.plot_datasets import plot_colorbar, plot_covid, plot_nys, plot_syn

SUBDIR = {'syn':'Synthetic','nys':'NY_Subway','covid':'JapanCOVID19'}
# output file label -> method, i.e. the sub-directory of out/npz/<dataset>
METHODS = {'tikhonov':'Tikhonov','OGTR':'OGTR','TRSS':'TRSS', 'inpaintingDAU':'Inpainting_DAU',
           'OGTR_online':'OGTR(online)',
           'proposed_batch':'Proposed(batch)','proposed_online':'Proposed(online)','proposed_unsupervised':'Proposed(unsupervised)'}
# dataset -> (visualized frame, top of the color scale of the difference panels)
FRAME = {'syn': (350, 5), 'nys': (350, 500), 'covid': (250, 1000)}


def plot_dataset(dataset,node_set,E,signal,idx,out_name,vmin,vmax,mode='diff',groundtruth=None,J=None):
    if(mode=='diff'):
        signal = np.abs(signal-groundtruth)
    plot = {'syn': plot_syn, 'nys': plot_nys, 'covid': plot_covid}[dataset]
    plot(node_set,E,signal,idx,out_name,vmin,vmax,mode,J)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    dataset = cfg.dataset
    test_sampling = cfg.get('test_sampling', 0.5)
    test_noise = cfg.get('test_noise', 0.1)
    data_idx = cfg.get('data_idx', 10)            # which test realization
    mode = cfg.get('vis_mode', 'diff')             # 'original' or 'diff'

    # ---- inputs (reproduced from the precomputed dataset file; no DataLoader) ----
    z = np.load(PROJECT_ROOT / 'datasets' / SUBDIR[dataset] / f'data_{data_idx:02d}.npz')
    gt = z['groundtruth']
    J = (z['uniform'] < test_sampling)
    observed = J * (gt + test_noise * z['noise'])   # noise pre-scaled by signal std (see DataLoader)

    TARGET_IDX, vmax_diff = FRAME[dataset]
    if(dataset=='syn'):
        node_set, E = z['node'], z['E']
    if(dataset=='nys'):
        node_set, E = np.load(PROJECT_ROOT/'datasets/NY_Subway/out_full.npz')['coodinates'], z['W']
    if(dataset=='covid'):
        node_set, E = np.load(PROJECT_ROOT/'datasets/JapanCOVID19/data.npz')['position'], z['W']

    vis_dir = PROJECT_ROOT / 'out/img/visualization' / dataset
    vis_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = PROJECT_ROOT / 'out/npz' / dataset

    sig_min, sig_max = float(gt[:,TARGET_IDX].min()), float(gt[:,TARGET_IDX].max())
    vmin = 0 if mode=='diff' else sig_min
    vmax = vmax_diff if mode=='diff' else sig_max

    def out(name):
        return str(vis_dir / f'{name}_{vtag(test_sampling)}_{TARGET_IDX}.pdf')

    buff = f'{TARGET_IDX=}\n{test_sampling=}\n{test_noise=}\n{data_idx=}\n'

    # original & observed; nothing is missing in the ground truth panel
    plot_dataset(dataset,node_set,E,gt,TARGET_IDX,out('original'),vmin=sig_min,vmax=sig_max,
                 mode='original',J=np.ones(gt.shape[0],dtype=bool))
    buff += f'observed:{np.sqrt(np.mean((gt[:,TARGET_IDX]-observed[:,TARGET_IDX])**2))}\n'
    plot_dataset(dataset,node_set,E,observed,TARGET_IDX,out('observed'),vmin=sig_min,vmax=sig_max,
                 mode='original',J=J[:,TARGET_IDX])

    # restored signals (loaded from per-method npz produced by compare_methods.py)
    for label,method in METHODS.items():
        restored = np.load(npz_dir / safe(method) / f'ratio_{vtag(test_sampling)}_{data_idx:02d}.npz')['restored']
        buff += f'{method}:{np.sqrt(np.mean((gt[:,TARGET_IDX]-restored[:,TARGET_IDX])**2))}\n'
        plot_dataset(dataset,node_set,E,restored,TARGET_IDX,out(label),vmin=vmin,vmax=vmax,
                     mode=mode,groundtruth=gt)

    # one bar for the signal panels, one for the difference panels
    plot_colorbar(sig_min,sig_max,str(vis_dir/'cbar_signal.pdf'))
    plot_colorbar(vmin,vmax,str(vis_dir/'cbar_diff.pdf'))

    with open(vis_dir/'rmse.txt','w') as f:
        f.write(buff)

if(__name__=='__main__'):
    main()
