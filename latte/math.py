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
        loss -= np.log(max(prob[n, int(label[n, 0])], np.finfo(np.float32).min))
    return loss / batch_size


def compute_seg_softmax_loss(output, prob, label):
	assert output.shape == prob.shape
	assert len(label) == len(output)
	batch_size = output.shape[0]
	loss = 0.0
	
	dim = int(np.prod(prob.shape)/batch_size)
	assert prob.ndim == 4
	spatial_dim = int(prob.shape[2] * prob.shape[3])

	print("shape: " + str(prob.shape))
	print("dim: " + str(dim) + ", spatial_dim: " + str(spatial_dim))

	for i in range(prob.shape[0]):
		x = output[i]
		e_x = np.exp(x - np.max(x))
		prob[i] = e_x / e_x.sum()

		for j in range(spatial_dim):
			gt_label = int(label.flat[i * spatial_dim + j])
			print("gt_label: " + str(gt_label))
			print("index: " + str(i*dim+gt_label*spatial_dim+j))
			loss -= np.log(max(prob.flat[i * dim + gt_label * spatial_dim + j], np.finfo(np.float32).min))

	return loss / batch_size

def seg_softmax_loss_backprop(output_grad, prob, label):
	batch_size = output_grad.shape[0]
	np.copyto(output_grad, prob)
	dim = int(np.prod(output_grad.shape)/batch_size)
	assert output_grad.ndim == 4
	spatial_dim = int(output_grad.shape[2] * output_grad.shape[3])

	for i in range(batch_size):
		for j in range(spatial_dim):
			gt_label = int(label.flat[i * spatial_dim + j])
			output_grad.flat[i * dim + gt_label * spatial_dim + j] -= 1

	output_grad /= batch_size

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
