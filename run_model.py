import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import LogisticRegression
from code.autoencoder import Autoencoder
from code.utils import normalization_transform, init_weights, evalaute_running_output_loss
from code.dataset import WikipediaDataSet


def train_autoencoder_model(test_train_split_dataset_path, model_name, numof_features,
                            num_epochs, batch_size, learning_rate, save_model_path=None):
    '''
    Train the autoencoder model.
    :param test_train_split_dataset_path: (str) the path to .npz file with train/test split file of processed
            (numeric) features and labels.
    :param model_name: (str) name of model file.
    :param numof_features: (int) the number of input features in the file.
    :param num_epochs: (int) number of learning epochs.
    :param batch_size: (int) size of the learning/testing batch/
    :param learning_rate: (float) learning rate
    :param save_model_path: (str) save the model under name `model_name` in this path.
    :return: autoencoder model
    '''
    train_dataset = WikipediaDataSet(test_train_split_dataset_path, train=True,
                                     transforms=normalization_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(numof_features)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.apply(init_weights)

    reconstructed_train_features_list = []
    train_labels_list = []

    try:
        for epoch in range(num_epochs):
            train_loss = 0
            # ==== Training ====
            model.train()
            for train_docs in train_dataloader:
                model.zero_grad()  # important

                train_features, train_labels = train_docs
                train_features = train_features.view(train_features.size(0), -1)
                train_features = Variable(train_features)
                # ===================forward=====================
                train_output = model(train_features)
                train_loss = criterion(train_output, train_features)
                # ===================backward====================
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                reconstructed_train_features_list.append(train_output.data.numpy())
                train_labels_list.append(train_labels.numpy())

            # ==== Testing ====
            model.eval()
            _, test_loss = evalaute_running_output_loss(model, test_dataloader, criterion)
            print('epoch [{}/{}], train loss:{:.8f}, test loss:{:.8f}'
                  .format(epoch + 1, num_epochs, train_loss.data[0], test_loss.data[0]))

    except KeyboardInterrupt:
        print('Exiting from training early')

    # gathering data for training documents classifier
    train_reconstructed_features = np.vstack(reconstructed_train_features_list)
    train_labels = np.hstack(train_labels_list)

    # saving progress
    np.savez_compressed(save_model_path +'reconstructed_train',
                        train_reconstructed_features=train_reconstructed_features,
                        train_labels=train_labels)

    if save_model_path:
        torch.save(model, save_model_path + 'trained_model' + model_name)

    return model


if __name__ == "__main__":

    num_epochs = 300
    batch_size = 2048
    learning_rate = 1e-3
    numof_features = 300
    already_trained = False
    model_name = '_300_200_100_50'
    data_path = './data/'
    test_train_split_dataset_path = data_path + 'full_data_test_train_split_with_embeddings.npz'

    test_dataset = WikipediaDataSet(test_train_split_dataset_path, train=False,
                                    transforms=normalization_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()

    train_reconstructed_features = None
    train_labels = None

    if not already_trained:
        model = train_autoencoder_model(test_train_split_dataset_path, model_name, numof_features, num_epochs,
                                        batch_size, learning_rate, save_model_path=data_path)

    else:
        model = torch.load(data_path + 'trained_model' + model_name)
        data = np.load(data_path + 'reconstructed_train.npz')
        train_reconstructed_features = data['train_reconstructed_features']
        train_labels = data['train_labels']

    print("Inference: test features compression")
    final_testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_reconstructed_features, test_labels = evalaute_running_output_loss(model, final_testloader, criterion,
                                                                            output_features_labels=True)
    np.savez_compressed(data_path + 'reconstructed_test',
                        test_reconstructed_features=test_reconstructed_features,
                        test_labels=test_labels)

    print("Classification using the compressed features:")
    clf_reconstructed = LogisticRegression(verbose=True, solver='sag', n_jobs=4, tol=0.1)
    clf_reconstructed.fit(train_reconstructed_features, train_labels)
    reconstructed_score = clf_reconstructed.score(test_reconstructed_features, test_labels)
    print("=== Reconstructed Score: {}".format(reconstructed_score))
