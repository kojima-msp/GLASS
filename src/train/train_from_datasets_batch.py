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

from src.methods.proposed_batch_model import Batch_Model
from src.utils.data import DataLoader
from src.utils.device import setup_device
from src.utils.paths import PROJECT_ROOT


def train(model,params,debug=False):

    # ==================== make saving dir & saving config files ====================
    now = datetime.datetime.now().astimezone().strftime('%Y%m%d%H%M%S')
    save_dir = PROJECT_ROOT / 'out/train/ProposedBatch' / params['dataset'] / now
    save_dir.mkdir(parents=True)

    with open(save_dir / '_options.json', mode="w") as json_file:
        json.dump(params, json_file, indent=2, ensure_ascii=False)

    # ==================== preprocess ====================
    device = setup_device()
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    groups = [{'params': model.lambda_.parameters(),                          'lr': params['lr_lambda']},
              {'params': model.d_h.parameters(),                              'lr': params['lr_dh']},
              {'params': list(model.eps_.parameters()) + list(model.beta_.parameters()), 'lr': params['lr_sobolev']}]
    optimizer = optim.Adam(groups)

    loss_min = float('inf')

    def stack(data_list, attr):
        return torch.as_tensor(np.stack([getattr(d, attr) for d in data_list]),
                               dtype=torch.get_default_dtype(), device=device)

    for epoch in tqdm.tqdm(range(params['Epoch'])):

        loss_epoch = 0

        for sampling_ratio in np.random.permutation(params['ratio']):
            for noise_level in np.random.permutation(params['noise_levels']):

                # ==================== data load ====================
                # data indices 0-9 are the training realizations
                dataset = params['dataset']
                data_list = DataLoader.load_range(dataset,range(10),noise_level=noise_level,sampling_rate = sampling_ratio)
                # ==================== train ====================
                # one optimizer step per (ratio, noise): all realizations in a single batch
                out = model.forward(stack(data_list,'data'), stack(data_list,'J'), stack(data_list,'L'))
                loss = criterion(out, stack(data_list,'groundtruth'))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()

        if(loss_min>loss_epoch):
            torch.save(model.state_dict(), save_dir / f"{str(epoch).zfill(6)}.pth")
            loss_min=loss_epoch
            print(f'loss of Epoch{epoch}: {loss_min}', flush=True)

        if debug and epoch % 10 == 0:
            evaluate(model,params)

    return model

def evaluate(model,params):
    # ==================== preprocess ====================
    model.eval()

    for sampling_ratio in [k/10 for k in range(1,10)]:

        noise_level = max(params['noise_levels'])

        # ==================== data load ====================
        # data indices 10-14 are the test realizations
        dataset = params['dataset']
        data_list = DataLoader.load_range(dataset,range(10,15),noise_level=noise_level,sampling_rate = sampling_ratio)

        # ==================== evel ====================
        rmse_list = []
        with torch.no_grad():
            for data in data_list:
                out = model.forward(data.data,data.J,data.L)
                rmse_list.append(np.sqrt(mean_squared_error(out.cpu().detach().numpy(),data.groundtruth)))
        print(sampling_ratio, np.array(rmse_list).mean(), flush=True)
    model.train()

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    params = OmegaConf.to_container(cfg, resolve=True)

    # ==================== model init ====================
    # model tensors follow the default dtype, so set it before constructing the model
    torch.set_default_dtype(torch.float32)
    model = Batch_Model(n_taps = params['proposed_L'],layer=params['K'])

    # ==================== train ====================
    train(model,params)

if(__name__=='__main__'):
    main()
