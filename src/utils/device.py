import torch


def setup_device():
    """Pick the device and make it the default.

    The model-based methods convert their numpy inputs with torch.tensor(), so they run wherever
    the default device points.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(device == 'cuda'):
        torch.cuda.set_device(0)
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cuda')
    return device
