import numpy as np
import rootutils
import torch
import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra

from src.methods.registry import build_methods
from src.utils.data import DataLoader, safe, vtag
from src.utils.paths import PROJECT_ROOT

TEST_RANGE = range(10, 15)   # data indices used for evaluation (test split)
ADAPTS_AT_TEST = {'Inpainting_DAU', 'Proposed(unsupervised)'}   # these optimize at test time -> need autograd

def restore(cfg,mode):
    '''Run the selected methods and save the restored signals, grouped per method:
        out/npz/<dataset>/<method>/<mode>_<v>_<idx>.npz  (key: restored).
    Inputs are reproducible from DataLoader, so only restorations are stored and a
    newly added method runs on identical data. Select with +methods=[OGTR,TRSS,...].'''
    methods = build_methods(cfg)
    selected = list(cfg.methods) if ('methods' in cfg and cfg.methods) else list(methods.keys())

    # how each method is called; anything not listed is a trained network taking (Y, J, L)
    CALL = {
        'OGTR':        lambda f,d: f(d.data,d.J,d.L,lambda_=cfg['OGTR_lambda'],K=cfg['K']),
        'OGTR_online': lambda f,d: f(d.data,d.J,d.L,lambda_=cfg['OGTR_online_lambda'],K=cfg['K']),
        'LR':          lambda f,d: f(d.data,d.J,d.L,lambda_=cfg['LR_lambda'],K=cfg['K']),
        'TRSS':        lambda f,d: f(d.data,d.J,d.L,lambda_=cfg['TRSS_lambda'],epsilon=cfg['TRSS_epsilon'],beta=cfg['TRSS_beta'],K=cfg['K']),
        'Tikhonov':    lambda f,d: f(d.data,d.J,d.L,gamma=cfg['Tikhonov_gamma'],beta=cfg['Tikhonov_beta'],K=cfg['K']),
        # the two test-time adapting methods are factories, so f() builds a fresh model
        'Inpainting_DAU':         lambda f,d: f().forward_unsupervised(d.data,d.J,d.DeltaG).cpu().detach().numpy(),
        'Proposed(unsupervised)': lambda f,d: np.asarray(f().forward(d.data,d.J,d.L)),
    }

    def run(method,d):
        call = CALL.get(method, lambda f,d: f.forward(d.data,d.J,d.L).cpu().detach().numpy())
        return call(methods[method], d)

    out_dir = PROJECT_ROOT / "out/npz" / cfg.dataset
    for method in selected:
        (out_dir / safe(method)).mkdir(parents=True, exist_ok=True)

    # ratio mode: sweep sampling rate with noise fixed; noise mode: sweep noise with rate fixed
    sweep = cfg.ratio if mode=='ratio' else cfg.noise_levels
    for v in tqdm.tqdm(sweep):
        noise_level = max(cfg.noise_levels) if mode=='ratio' else v
        sampling_rate = v if mode=='ratio' else max(cfg.ratio)
        for idx in TEST_RANGE:
            d = DataLoader(cfg.dataset, idx=idx, noise_level=noise_level, sampling_rate=sampling_rate)
            for method in selected:
                mdir = out_dir / safe(method)
                if method in ADAPTS_AT_TEST:
                    restored = run(method,d)
                else:
                    with torch.no_grad():
                        restored = run(method,d)
                np.savez(mdir / f"{mode}_{vtag(v)}_{idx:02d}.npz", restored=np.asarray(restored, dtype=np.float32))

    print(f'saved restored signals ({", ".join(selected)}) under {out_dir}')

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    # mode=null runs both sweeps; mode=ratio|noise runs only that one
    if cfg.mode is not None and cfg.mode not in ('ratio', 'noise'):
        raise ValueError("mode must be 'ratio' or 'noise' (omit it to run both sweeps)")
    tasks = ['ratio', 'noise'] if cfg.mode is None else [cfg.mode]
    for task in tasks:
        restore(cfg,task)

if(__name__=='__main__'):
    main()