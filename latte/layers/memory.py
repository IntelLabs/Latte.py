import numpy as np
from ..ensemble import DataEnsemble

def MemoryDataLayer(net, shape):
    value = np.zeros((net.batch_size, *shape))
    ens = DataEnsemble(value)
    net.add_ensemble(ens)
    return ens, value
