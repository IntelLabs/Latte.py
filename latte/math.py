'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import numpy as np
import sys
from numba import jit
import math
import os
import latte
import latte.util as util

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
        loss -= np.log(max(prob[n, int(label[n, 0])], np.finfo(np.float32).tiny))
    return loss / batch_size


@jit
def compute_seg_softmax_loss(output, prob, label, ignore_label):
    
    assert output.shape == prob.shape
    np.copyto(prob, output)
    batch_size = output.shape[0]
    channels = output.shape[1]
    loss = 0.0
    batch_weight = 0.0
    spatial_dim = int(output.shape[2]*output.shape[3])
    dim = int(np.prod(output.shape)/batch_size)
    
    assert prob.ndim == 4
    scale_data = np.zeros((batch_size, 1, prob.shape[2], prob.shape[3]))
    prob_data = np.zeros_like(scale_data)

    # compute probabilities
    for i in range(batch_size):
        scale_data = output[i]
        for j in range(channels):
            for k in range(spatial_dim):
                scale_data.flat[k] = max(scale_data.flat[k], output.flat[i*dim+j*spatial_dim+k])


        for j in range(channels):
            for k in range(spatial_dim):
                x = prob.flat[i*dim+j*spatial_dim+k]
                x -= scale_data.flat[k]
                e_x = math.exp(x)
                prob.flat[i*dim+j*spatial_dim+k] = e_x 

        prob[i] = prob[i]/np.sum(prob[i], axis=0, keepdims=True)
        '''
        for k in range(spatial_dim): 
            for j in range(channels): 
                prob_data.flat[k] += prob.flat[i*dim+j*spatial_dim+k]

        for j in range(channels):
            for k in range(spatial_dim):
                prob.flat[i*dim+j*spatial_dim+k] /= prob_data.flat[k]
        '''
    # calculate softmax loss
    for i in range(batch_size):
        for j in range(spatial_dim):
            gt_label = int(label.flat[i * spatial_dim + j])
            if gt_label == ignore_label:
                continue
 
            elif gt_label >= 0 and gt_label < channels:
                batch_weight += 1
                loss -= np.log(max(prob.flat[i * dim + gt_label * spatial_dim + j], np.finfo(np.float32).tiny))

    #print("output max: " + str(np.max(output)) + ", output min: " + str(np.min(output)))
    #print("prob max: " + str(np.max(prob)) + ", prob min: " + str(np.min(prob)))
    #print("loss: " + str(loss))
    #print("batch_weight: " + str(batch_weight))
 
    return loss / batch_weight

@jit
def seg_softmax_loss_backprop(output_grad, prob, label, ignore_label):
    assert output_grad.shape == prob.shape
    np.copyto(output_grad, prob)
    
    batch_size = prob.shape[0]
    channels = prob.shape[1]
    dim = int(np.prod(prob.shape)/batch_size)
    assert prob.ndim == 4
    spatial_dim = int(prob.shape[2] * prob.shape[3])
    batch_weight = 0.0

    for i in range(batch_size):
        for j in range(spatial_dim):
            gt_label = int(label.flat[i * spatial_dim + j])
   
            if gt_label == ignore_label:
                continue 
            
            elif gt_label >= 0 and gt_label < channels:
                batch_weight += 1
                output_grad.flat[i * dim + gt_label * spatial_dim + j] -= 1

    output_grad /= (batch_weight)

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
    spatial_dim = output.shape[2]*output.shape[3]
    dim = int(np.prod(output.shape)/batch_size)
    label_dim = int(np.prod(label.shape)/batch_size)
    confusion_matrix = np.zeros((batch_size, output.shape[1], output.shape[1]))
    
    scale_data = np.zeros((batch_size, 1, output.shape[2], output.shape[3]), dtype=np.float32)
    pred_labels = np.zeros_like(scale_data)

    for n in range(batch_size):
        scale_data[n] = output[n,0]
        for j in range(spatial_dim):
            for c in range(output.shape[1]):
                if output.flat[n*dim+c*spatial_dim+j] >= scale_data.flat[n*label_dim+j]:
                    scale_data.flat[n*label_dim+j] = output.flat[n*dim+c*spatial_dim+j]
                    pred_labels.flat[n*label_dim+j] = c

    for n in range(batch_size):
        for h in range(output.shape[2]):
            for w in range(output.shape[3]):
                actual = int(label[n,0,h,w])
                pred = int(pred_labels[n,0,h,w])
                if actual != ignore_label and actual >= 0 and actual < output.shape[1]:
                    confusion_matrix[n,actual,pred] += 1

    for n in range(batch_size):
        if np.sum(confusion_matrix[n]) != 0:
            accuracy += np.trace(confusion_matrix[n])/np.sum(confusion_matrix[n])

    return accuracy / batch_size
