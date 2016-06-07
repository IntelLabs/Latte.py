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
        if np.argmax(output[n]) == label[n, 0]:
            accuracy += 1
    return accuracy / batch_size

def compute_seg_accuracy(output, label):
    accuracy = 0.0
    confusion_matrix = np.zeros((output.shape[0], output.shape[0]))
    for n in range(batch_size):
        actual = int(label[n,0])
        pred = np.argmax(output[n])
        confusion_matrix[actual][pred] += 1

    if np.sum(confusion_matrix) != 0:
        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)

    return accuracy / batch_size
