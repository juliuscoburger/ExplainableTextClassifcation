import utils
import numpy as np
import torch
import string
from model import Model
from torch.autograd import Variable
import torch.nn as nn


def encode(words):
    alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list('’') + list('\n')
    encoded_data = torch.zeros(1, 70, 1014)
    chars = words
    for index, char in enumerate(chars):
        if char in alphabet and index < 1014:
            encoded_data[0][alphabet.index(char)][index] = 1
    return encoded_data


# inputs the model and the string to do the analysis on
# decoded option is set True if the input is allready encoded
def LRP(module, input_str, true_class, num_classes, encoded=False):
    if not encoded:
        test_data = encode(input_str)
    else:
        test_data = input_str
        test_data.resize_((1, 70, 1014))
        
    print(module(test_data), input_str)

    layers = list(module.modules())[1:]
    layers2 = []

    for i1 in range(len(layers)):
        if isinstance(layers[i1], torch.nn.modules.container.Sequential):
            for i2 in layers[i1]:
                layers2.append(i2)

    L = len(layers2)

    # print("\nSTART PROPAGATE")
    # propagate through the Network
    A = [test_data] + [None] * L
    first_linear = 0
    for l in range(L):
        # print(l, A[l].shape, layers2[l])
        if isinstance(layers2[l], torch.nn.Linear) and first_linear == 0:
            A[l + 1] = layers2[l].forward(A[l].view(A[l].size(0), -1))
            first_linear = 1
        else:
            A[l + 1] = layers2[l].forward(A[l])

    # print("Output: ", A[-1], A[-1].shape)
    # print("\nA:" , A[-1].shape)

    linear = 0
    T = torch.FloatTensor((1.0 * (np.arange(num_classes) == true_class).reshape([num_classes])))
    R = [None] * L + [(A[-1] * T).data]

    for l in range(0, L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers2[l], torch.nn.MaxPool1d):
            layers2[l] = torch.nn.AvgPool1d(kernel_size=3, stride=3)

        rho = lambda p: p;
        incr = lambda z: z + 1e-9

        if isinstance(layers2[l], torch.nn.Linear): linear += 1
        if linear == 3:
            z = incr(utils.newlayer(layers2[l], rho).forward(A[l].view(A[l].size(0), -1)))
            linear += 1
        else:
            z = incr(utils.newlayer(layers2[l], rho).forward(A[l]))  # step 1

        s = (R[l + 1] / z).data  # step 2
        (z * s).sum().backward();
        c = A[l].grad  # step 3
        R[l] = (A[l] * c).data

    return R


# return the sentence without the most relevant word
def delete_most_important_word(sentence, relevances, most=True):
    s = sentence.cpu().detach().numpy()

    words = [[]]
    word_r = [0]

    for i in range(1014):
        # print(s[:,i].sum())
        if s[:, i].sum() == 0:
            words.append([])
            word_r.append(0)
        else:
            words[-1].append(i)
            word_r[-1] += relevances[i]

    words2 = []
    word_r2 = []

    for i in range(len(words)):
        if words[i] != []:
            words2.append(words[i])
            word_r2.append(word_r[i])

    if most:
        top_ind = words2[np.argmax(word_r2)]
    else:
        top_ind = words2[np.argmin(word_r2)]

    # print(word_r2)

    sentence_out = []
    for i in range(1014):
        if i not in top_ind:
            sentence_out.append(s[:, i])

    zeros = [0] * 70
    while (len(sentence_out) < 1014):  # pad with spaces
        sentence_out.append(zeros)
    out = list(map(list, zip(*sentence_out)))  # transpose to get the right shape

    return out


# for nonflip module
# return the sentence without the x most relevant chars
# setting most deletes the most important chars, otherwise the least important
def delete_most_important_chars(sentence, relevances, x, most=True):
    ind = np.argsort(relevances)

    if most:
        top_ind = ind[-x:]
    else:
        top_ind = ind[:x]

    sentence_out = []
    s = sentence.cpu().detach().numpy()
    for i in range(1014):
        if i not in top_ind:
            sentence_out.append(s[:, i])

    zeros = [0] * 70
    while (len(sentence_out) < 1014):  # pad with spaces
        sentence_out.append(zeros)
    out = list(map(list, zip(*sentence_out)))  # transpose to get the right shape

    return out



def integrated_gradient(module, input_str, true_class, encoded=False):
    baseline = encode("")
    if not encoded:
        test_data = encode(input_str)
    else:
        test_data = input_str
        test_data.resize_((1, 70, 1014))
        
       
    test_data.requires_grad_(True)
    loss_crit = nn.CrossEntropyLoss()

    steps = 50
    inputs = test_data
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads = []

    for si in scaled_inputs:
        out = module.forward(inputs)  
        loss = loss_crit(out, torch.tensor([true_class]))
        loss.backward()
        grads.append(inputs.grad[0].cpu().detach().numpy())
        
    grads = np.array(grads)
    avg_grads = np.average(grads[:-1], axis=0)
    integrated_grad = (inputs[0].detach().cpu().detach().numpy() - baseline[0].cpu().detach().numpy()) * avg_grads

        
    return integrated_grad


