import numpy as np
import optuna
import rootutils
import tqdm
from sklearn.metrics import mean_squared_error

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra

from src.methods.LaplacianRegularization import LR
from src.methods.OGTR import OGTR, OGTR_online
from src.methods.Tikhonov import Tikhonov
from src.methods.TRSS import TRSS
from src.utils.data import DataLoader
from src.utils.device import setup_device


# a lambda that makes the solver diverge must score as worst, not abort the whole study
def _rmse(gt, x):
    return np.sqrt(mean_squared_error(gt, x)) if np.all(np.isfinite(x)) else np.inf


def _lambda_only(trial):
    return {"lambda_": trial.suggest_float("lambda", 0.001, 100, log=True)}

def _trss(trial):
    return {"lambda_": trial.suggest_float("lambda", 0.001, 100, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.001, 100, log=True),
            "beta": trial.suggest_float("beta", 0.001, 3, log=True)}

def _tikhonov(trial):
    return {"gamma": trial.suggest_float("gamma", 0.001, 100, log=True),
            "beta": trial.suggest_float("beta", 0.001, 100, log=True)}

# method -> (restoration function, search space)
SEARCH_SPACE = {
    'OGTR':        (OGTR,        _lambda_only),
    'OGTR_online': (OGTR_online, _lambda_only),
    'LR':          (LR,          _lambda_only),
    'TRSS':        (TRSS,        _trss),
    'Tikhonov':    (Tikhonov,    _tikhonov),
}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    if "method" not in cfg:
        raise ValueError(f"specify method via CLI, e.g. +method=TRSS  (choices: {'/'.join(SEARCH_SPACE)})")

    setup_device()

    # tune on the train realizations (0-9) across every sampling rate -> no test leakage
    data_list = DataLoader.load_range(cfg.dataset, range(10))
    restore, suggest = SEARCH_SPACE[cfg.method]

    def objective(trial):
        params = suggest(trial)
        return sum(_rmse(d.groundtruth, restore(d.data, d.J, d.L, K=cfg.K, **params))
                   for d in tqdm.tqdm(data_list))

    print(cfg.method)
    study = optuna.create_study()
    study.optimize(objective, n_trials=cfg.n_trials)
    print(cfg.dataset, study.best_trial)

if(__name__=='__main__'):
    main()
