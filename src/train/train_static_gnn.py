import datetime
import json
import random

import numpy as np
import rootutils
import torch
from torch import optim

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra
from omegaconf import OmegaConf

from src.methods.static_gnn import GCN, ChebNet
from src.utils.data import DataLoader
from src.utils.device import setup_device
from src.utils.paths import PROJECT_ROOT


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    if "model" not in cfg:
        raise ValueError("specify the model via CLI, e.g. +model=cheb  (choices: gcn/cheb)")

    device = setup_device()

    if cfg.model == 'gcn':
        model = GCN(hidden=cfg.static_gnn_hidden, n_layers=cfg.static_gnn_layers)
        name = 'GCN'
    elif cfg.model == 'cheb':
        model = ChebNet(hidden=cfg.static_gnn_hidden, n_layers=cfg.static_gnn_layers, cheb_K=cfg.cheb_K)
        name = 'ChebNet'
    else:
        raise ValueError(f"unknown model '{cfg.model}' (choices: gcn/cheb)")

    # ==================== make saving dir & saving config files ====================
    now = datetime.datetime.now().astimezone().strftime('%Y%m%d%H%M%S')
    save_dir = PROJECT_ROOT / 'out/train' / name / cfg.dataset / now
    save_dir.mkdir(parents=True)

    with open(save_dir / '_options.json', mode="w") as json_file:
        json.dump(OmegaConf.to_container(cfg, resolve=True), json_file, indent=2, ensure_ascii=False)

    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.static_gnn_lr)
    loss_min = np.inf

    for epoch in range(cfg.static_gnn_epochs):
        loss_sum = 0
        # train on the train realizations (0-9) across sampling rates and noise levels
        for sampling_rate in random.sample(list(cfg.ratio), len(cfg.ratio)):
            noise_level = random.choice(list(cfg.noise_levels))
            data_list = DataLoader.load_range(cfg.dataset, range(10),
                                              noise_level=noise_level, sampling_rate=sampling_rate)
            for data in data_list:
                out = model.forward(data.data, data.J, data.L)
                loss = criterion(out, torch.tensor(data.groundtruth).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.cpu().detach().numpy()

        if loss_min > loss_sum:
            torch.save(model.state_dict(), save_dir / f"{str(epoch).zfill(6)}.pth")
            loss_min = loss_sum
            print(f'{cfg.dataset} {cfg.model} Epoch{epoch}: loss_sum {loss_sum:.6g}', flush=True)

    return model


if __name__ == '__main__':
    main()
