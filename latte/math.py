import numpy as np

def compute_softmax_loss(output, prob, label):
    assert output.shape == prob.shape
    assert len(label) == len(output)
    batch_size = output.shape[0]
    loss = 0.0
    for n in range(batch_size):
        x = output[n]
        e_x = np.exp(x - np.max(x))
        prob[n] = e_x / e_x.sum()
        loss -= np.log(max(prob[n, int(label[n, 0])], np.finfo(np.float32).min))
    return loss / batch_size

def softmax_loss_backprop(output_grad, prob, label):
    batch_size = output_grad.shape[0]
    np.copyto(output_grad, prob)
    for n in range(batch_size):
        output_grad[n, int(label[n, 0])] -= 1
    output_grad /= batch_size

def compute_accuracy(output, label):
    batch_size = output.shape[0]
    accuracy = 0.0
    for n in range(batch_size):
        if np.argmax(output[n]) == label_value[n, 0]:
            accuracy += 1
    return accuracy / batch_size
