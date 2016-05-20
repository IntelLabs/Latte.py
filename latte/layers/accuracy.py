import numpy as np
from ..ensemble import AccuracyEnsemble

class AccuracyEnsemble(AccuracyEnsemble):
    def __init__(self, net, shape):
        super().__init__(np.zeros(shape, np.float32))
        self.prob = np.zeros((net.batch_size, ) + shape, np.float32)
        self.net = net

    def forward(self, bottom, label):
        accuracy = np.sum(np.argmax(bottom, axis=-1) == label.flatten())
        self.net.accuracy = accuracy / label.size

def AccuracyLayer(net, bottom, label):
    ens = AccuracyEnsemble(net, bottom.shape)
    net.add_ensemble(ens)
    net.add_loss_connection(bottom, ens)
    net.add_loss_connection(label, ens)
    return ens
