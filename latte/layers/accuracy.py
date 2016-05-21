import numpy as np
from ..ensemble import AccuracyEnsemble

class AccuracyEnsemble(AccuracyEnsemble):
    def __init__(self, net, shape):
        super().__init__(np.zeros(shape, np.float32))
        self.net = net

    def forward(self, bottom, label):
        accuracy = 0.0
        for n in range(bottom.shape[0]):
            if np.argmax(bottom[n]) == label[n, 0]:
                accuracy += 1
        self.net.accuracy = accuracy / label.size

def AccuracyLayer(net, bottom, label):
    ens = AccuracyEnsemble(net, bottom.shape)
    net.add_ensemble(ens)
    net.add_loss_connection(bottom, ens)
    net.add_loss_connection(label, ens)
    return ens
