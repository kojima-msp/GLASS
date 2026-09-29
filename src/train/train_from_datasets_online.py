import datetime
import json

import numpy as np
import rootutils
import torch
import tqdm
from sklearn.metrics import mean_squared_error
from torch import optim

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra
from omegaconf import OmegaConf

from src.methods.proposed_online_model import Online_Model
from src.utils.data import DataLoader
from src.utils.device import setup_device
from src.utils.paths import PROJECT_ROOT


def train(model, params, debug=False):

    # ==================== make saving dir & saving config files ====================
    now = datetime.datetime.now().astimezone().strftime('%Y%m%d%H%M%S')
    save_dir = PROJECT_ROOT / 'out/train/ProposedOnline' / params['dataset'] / now
    save_dir.mkdir(parents=True)

    with open(save_dir / '_options.json', mode="w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # ==================== preprocess ====================
    device = setup_device()
    model = model.to(device)
    model.train()

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam([{'params': model.model.lambda_.parameters(), 'lr': params['lr_lambda']},
                            {'params': model.model.d_h.parameters(),     'lr': params['lr_dh']},
                            {'params': list(model.model.eps_.parameters()) + list(model.model.beta_.parameters()),
                             'lr': params['lr_sobolev']}])

    # training pool: all (ratio x noise x realization) samples, pre-stacked on the GPU.
    # Indices 0-9 are the train split. L is kept per-sample (syn has a graph per realization).
    dt = torch.get_default_dtype()
    pool = [d for sr in params['ratio'] for nl in params['noise_levels']
            for d in DataLoader.load_range(params['dataset'], range(10), noise_level=nl, sampling_rate=sr)]
    Y_all    = torch.stack([torch.as_tensor(d.data,        dtype=dt, device=device) for d in pool])
    J_all    = torch.stack([torch.as_tensor(d.J,           dtype=dt, device=device) for d in pool])
    gt_all   = torch.stack([torch.as_tensor(d.groundtruth, dtype=dt, device=device) for d in pool])
    L_all    = torch.stack([torch.as_tensor(d.L,           dtype=dt, device=device) for d in pool])
    n_samples = len(pool)
    B = params['online_batch']

    loss_min = float('inf')

    for epoch in tqdm.tqdm(range(params['Epoch'])):

        loss_epoch = 0

        order = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, B):
            idx = order[i:i+B]
            out = model.forward(Y_all[idx], J_all[idx], L_all[idx])
            loss = criterion(out, gt_all[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        if loss_min > loss_epoch:
            # save the inner Batch_Model weights: that is what Online_Model(trained_model_path=...) loads
            torch.save(model.model.state_dict(), save_dir / f"{str(epoch).zfill(6)}.pth")
            loss_min = loss_epoch
            print(f'loss of Epoch{epoch}: {loss_min}', flush=True)

        if debug:
            evaluate(model, params)

    return model


def evaluate(model, params):
    model.eval()
    with torch.no_grad():
        for sampling_ratio in [k/10 for k in range(1, 10)]:
            noise_level = max(params['noise_levels'])
            # data indices 10-14 are the test realizations
            data_list = DataLoader.load_range(params['dataset'], range(10, 15),
                                              noise_level=noise_level, sampling_rate=sampling_ratio)
            rmse_list = []
            for data in data_list:
                out = model.forward(data.data, data.J, data.L)
                rmse_list.append(np.sqrt(mean_squared_error(out.cpu().detach().numpy(), data.groundtruth)))
            print(sampling_ratio, np.array(rmse_list).mean())
    model.train()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    params = OmegaConf.to_container(cfg, resolve=True)

    torch.set_default_dtype(torch.float32)
    # warm-start from the batch-trained weights; all parameters stay trainable
    model = Online_Model(n_taps=params['proposed_L'], layer=params['K'],
                         trained_model_path=params['proposed_parameter_path'])

    train(model, params, debug=False)


if(__name__ == '__main__'):
    main()