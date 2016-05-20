import numpy as np
from ..ensemble import LossEnsemble

class SoftmaxLossEnsemble(LossEnsemble):
    def __init__(self, net, shape):
        super().__init__(np.zeros(shape, np.float32))
        self.prob = np.zeros((net.batch_size, ) + shape, np.float32)
        self.net = net

    def forward(self, bottom, label):
        loss = 0.0
        for n in range(bottom.shape[0]):
            x = bottom[n]
            e_x = np.exp(x - np.max(x))
            self.prob[n] = e_x / e_x.sum()
            loss -= np.log(max(self.prob[n, int(label[n, 0])], np.finfo(np.float32).min))
        self.net.loss += loss

    def backward(self, bot_grad, label):
        np.copyto(bot_grad, self.prob)
        for n in range(self.prob.shape[0]):
            bot_grad[n, int(label[n, 0])] -= 1
        bot_grad /= np.prod(bot_grad.shape)


def SoftmaxLossLayer(net, bottom, label):
    ens = SoftmaxLossEnsemble(net, bottom.shape)
    net.add_ensemble(ens)
    net.add_loss_connection(bottom, ens)
    net.add_loss_connection(label, ens)
    return ens
