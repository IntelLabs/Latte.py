import numpy as np
from ..ensemble import AccuracyEnsemble
import math

class AccuracyEnsemble(AccuracyEnsemble):
    def __init__(self, net, shape):
        super().__init__(np.zeros(shape, np.float32))
        self.net = net

    def forward(self, bottom, label):
        accuracy = 0.0
        confusion_matrix = np.zeros((bottom.shape[0], bottom.shape[0]))
        for n in range(bottom.shape[0]):
            actual = int(label[n,0])
            pred = np.argmax(bottom[n])
            confusion_matrix[actual][pred] += 1        
        
        if np.sum(confusion_matrix) == 0:
            self.net.accuracy = 0.0
        else:
            self.net.accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

def SegAccuracyLayer(net, bottom, label):
    ens = AccuracyEnsemble(net, bottom.shape)
    net.add_ensemble(ens)
    net.add_loss_connection(bottom, ens)
    net.add_loss_connection(label, ens)
    return ens
