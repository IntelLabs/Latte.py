import numpy as np
from ..ensemble import DataEnsemble

def MemoryDataLayer(net, shape):
    ens = DataEnsemble(net.batch_size, shape)
    net.add_ensemble(ens)
    return ens
