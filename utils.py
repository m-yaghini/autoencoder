import torch
from torch.autograd import Variable
from numpy import np
from math import inf


def normalization_transform(x):
    import torch.nn.functional as F
    x = torch.from_numpy(x).float()
    x_normalized = F.normalize(x, p=2, dim=0)
    return x_normalized


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.2, 0.2)


def evalaute_running_output_loss(model, dataloader, criterion, output_features_labels=False, iterations=inf):
    if output_features_labels:
        reconstructed_test_features_list = []
        test_labels_set = []
        for d in dataloader:
            _features, _labels = d
            _features = Variable(_features)
            _output = model(_features)
            reconstructed_test_features_list.append(_output.data.numpy())
            test_labels_set.append(_labels)

        test_reconstructed_features = np.vstack(reconstructed_test_features_list)
        test_labels = np.hstack(test_labels_set)
        return test_reconstructed_features, test_labels
    else:
        running_output, running_loss = 0, 0
        for i, d in enumerate(dataloader):
            if i >= iterations:
                break
            _features, _labels = d
            _features = Variable(_features)
            _output = model(_features)
            _loss = criterion(_output, _features)
            running_output += _output.data[0]  # just for printing purposes: take one sample of the batch.
            running_loss += _loss
        return running_output / (i + 1), running_loss / (i + 1)


