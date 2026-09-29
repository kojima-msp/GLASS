import torch

from src.methods.Inpainting_DAU import Inpainting_DAU
from src.methods.LaplacianRegularization import LR
from src.methods.OGTR import OGTR, OGTR_online
from src.methods.proposed_batch_model import Batch_Model
from src.methods.proposed_online_model import Online_Model
from src.methods.proposed_unsupervised import Unsupervised_Model
from src.methods.static_gnn import GCN, ChebNet
from src.methods.Tikhonov import Tikhonov
from src.methods.TRSS import TRSS
from src.utils.device import setup_device


def build_methods(params):
    """Every method of the comparison, keyed by the name used in compare_methods.py.
    """
    device = setup_device()

    def load(model, path, cast_float=False):
        model = model.to(device).eval()
        model.load_state_dict(torch.load(path, map_location=device))
        # the GNN weights were trained in float64; cast to match the float32 pipeline
        return model.float() if cast_float else model

    return {
        # model-based methods, called as plain functions
        'OGTR': OGTR,
        'OGTR_online': OGTR_online,
        'TRSS': TRSS,
        'Tikhonov': Tikhonov,
        'LR': LR,

        # test-time adapting methods are registered as factories: they update their own weights
        # in forward, so each restoration has to start from a freshly built model
        'Inpainting_DAU': lambda: Inpainting_DAU(
            layer=params['K'],P=params['InpaintingDAU_P'],Q=params['InpaintingDAU_Q']).to(device),
        'Proposed(unsupervised)': lambda: Unsupervised_Model(
            n_taps=params['proposed_L'],layer=params['K']).to(device),

        # trained networks, shared across restorations
        'Proposed(batch)': load(Batch_Model(n_taps=params['proposed_L'],layer=params['K']),
                                params['proposed_parameter_path']),
        'Proposed(online)': Online_Model(
            n_taps=params['proposed_L'],layer=params['K'],
            trained_model_path=params['proposed_online_parameter_path']).to(device).eval(),
        'GCN': load(GCN(hidden=params['static_gnn_hidden'], n_layers=params['static_gnn_layers']),
                    params['gcn_parameter_path'], cast_float=True),
        'ChebNet': load(ChebNet(hidden=params['static_gnn_hidden'], n_layers=params['static_gnn_layers'],
                                cheb_K=params['cheb_K']), params['cheb_parameter_path'], cast_float=True),
    }
