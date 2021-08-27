import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)

# in case the computer testing has no gpu, tests will just pass
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_tt():
    X = torch.randn(16,16,16)
    y1 = tn.Tensor(X, ranks_tt=3).torch()
    y2 =  tn.Tensor(X, ranks_tt=3, device=device).torch().cpu()
    # metrics(y1, X)
    # metrics(y2, X)
    assert torch.abs(y1-y2).max() < 1e-5

test_tt()