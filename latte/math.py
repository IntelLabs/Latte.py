import numpy as np
from numba import jit

@jit
def compute_softmax_loss(output, prob, label):
    assert output.shape == prob.shape
    assert len(label) == len(output)
    batch_size = output.shape[0]
    loss = 0.0
    for n in range(batch_size):
        x = output[n]
        e_x = np.exp(x - np.max(x))
        prob[n] = e_x / e_x.sum()
        loss -= np.log(max(prob[n, int(label[n, 0])], np.finfo(np.float32).eps))
    return loss / batch_size


@jit
def compute_seg_softmax_loss(output, prob, label, ignore_label):
    assert output.shape == prob.shape
    assert len(label) == len(output)
    batch_size = output.shape[0]
    loss = 0.0
    batch_weight = 0.0
    
    dim = int(np.prod(prob.shape)/batch_size)
    assert prob.ndim == 4
    spatial_dim = int(prob.shape[2] * prob.shape[3])

    for i in range(prob.shape[0]):
        x = output[i]
        e_x = np.exp(x - np.max(x))
        prob[i] = e_x / e_x.sum()
        
        for j in range(spatial_dim):
            gt_label = int(label.flat[i * spatial_dim + j])
            
            if gt_label != ignore_label and gt_label >= 0 and gt_label < prob.shape[1]:
                batch_weight += 1
                loss -= np.log(max(prob.flat[i * dim + gt_label * spatial_dim + j], np.finfo(np.float32).eps))
                
    return loss / batch_weight

@jit
def seg_softmax_loss_backprop(output_grad, prob, label, ignore_label):
    batch_size = output_grad.shape[0]
    np.copyto(output_grad, prob)
    dim = int(np.prod(output_grad.shape)/batch_size)
    assert output_grad.ndim == 4
    spatial_dim = int(output_grad.shape[2] * output_grad.shape[3])
    batch_weight = 0.0

    for i in range(batch_size):
        for j in range(spatial_dim):
            gt_label = int(label.flat[i * spatial_dim + j])
    
            if gt_label != ignore_label and gt_label >= 0 and gt_label < output_grad.shape[1]:
                batch_weight += 1
                output_grad.flat[i * dim + gt_label * spatial_dim + j] -= 1

    output_grad /= batch_weight

@jit
def softmax_loss_backprop(output_grad, prob, label):
    batch_size = output_grad.shape[0]
    np.copyto(output_grad, prob)
    for n in range(batch_size):
        output_grad[n, int(label[n, 0])] -= 1
    output_grad /= batch_size

@jit
def compute_accuracy(output, label):
    batch_size = output.shape[0]
    accuracy = 0.0
    for n in range(batch_size):
        if np.argmax(output[n]) == label[n, 0]:
            accuracy += 1
    return accuracy / batch_size

@jit
def compute_seg_accuracy(output, label, ignore_label):
    assert output.ndim == 4
    batch_size = output.shape[0]
    accuracy = 0.0
    confusion_matrix = np.zeros((output.shape[0], output.shape[0]))
    for n in range(batch_size):
        pred = np.argmax(output[n])
        for h in range(output.shape[2]):
            for w in range(output.shape[3]):
                actual = int(label.flat[h * output.shape[3] + w])
                
                if actual != ignore_label and actual >= 0 and actual < output.shape[1]:
                    confusion_matrix[actual][pred] += 1

    if np.sum(confusion_matrix) != 0:
        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)

    return accuracy / batch_size
