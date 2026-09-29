import functools
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.utils.paths import PROJECT_ROOT

if not OmegaConf.has_resolver("project_root"):
    OmegaConf.register_new_resolver("project_root", lambda: str(PROJECT_ROOT))


# preprocessed dataset directories (one self-contained data_{idx}.npz per data index)
DATASET_DIR = {'syn': 'Synthetic', 'nys': 'NY_Subway', 'covid': 'JapanCOVID19'}


# naming of the restored signals, written by compare_methods.py and read by the analysis scripts:
#   out/npz/<dataset>/<safe(method)>/<mode>_<vtag(value)>_<idx>.npz
def safe(method):
    '''filesystem-safe sub-directory name for a method.'''
    return method.replace('(', '_').replace(')', '')


def vtag(v):
    '''sweep value -> zero-padded integer tag for filenames (0.5 -> "050"), so files sort.'''
    return f"{round(float(v)*100):03d}"


@functools.cache
def _data_dir():
    return Path(OmegaConf.load(PROJECT_ROOT / "conf" / "base.yaml").data_dir)


@functools.cache
def _load_npz(dataset, idx):
    z = np.load(_data_dir() / DATASET_DIR[dataset] / f'data_{idx:02d}.npz')
    return {k: z[k] for k in z.files}


@functools.cache
def _sweep_defaults():
    # default (noise_levels, ratio) of load_range(...), taken from the config
    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "base.yaml")
    return list(cfg.noise_levels), list(cfg.ratio)


class DataLoader:
    def __init__(self,dataset,idx = 10,noise_level = 0.1,sampling_rate=0.5):
        # one precomputed realization; idx 0-9 are train, 10-14 are test
        data = _load_npz(dataset, idx)

        # float32 everywhere: torch.tensor(np_array) keeps the numpy dtype regardless of
        # torch.set_default_dtype
        self.groundtruth = data['groundtruth'].astype(np.float32)
        self.L = data['L'].astype(np.float32)
        self.DeltaG = data['DeltaG'].astype(np.float32)
        self.J = np.where(data['uniform'] < sampling_rate, 1, 0).astype(np.float32)
        # data['noise'] is scaled by the signal std, so noise_level is the noise/signal std ratio
        self.data = (self.J * (self.groundtruth + noise_level * data['noise'])).astype(np.float32)

    @staticmethod
    def _as_list(value, default):
        if value is None:
            return default
        return [value] if np.isscalar(value) else list(value)

    @classmethod
    def load_range(cls, dataset, load_range, noise_level=None, sampling_rate=None):
        """Load every index in ``load_range`` (e.g. range(10) for train, range(10,15) for test).
        ``noise_level`` and ``sampling_rate`` may each be a scalar, a list of values to sweep, or
        None to sweep the full default set (conf/base.yaml: ratio / noise_levels)."""
        default_noises, default_rates = _sweep_defaults()
        noises = cls._as_list(noise_level, default_noises)
        rates = cls._as_list(sampling_rate, default_rates)
        return [cls(dataset, idx, nl, sr) for idx in load_range for nl in noises for sr in rates]
