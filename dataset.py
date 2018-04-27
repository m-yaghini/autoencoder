from torch.utils.data import Dataset


class WikipediaDataSet(Dataset):
    def __init__(self, path, train, transforms=None):
        self.data = np.load(path)  # ('./pilot_data_test_train_split.npz')
        self.transforms = transforms
        if train:
            self.features = self.data['X_train']
            self.labels = self.data['y_train']
        else:
            self.features = self.data['X_test']
            self.labels = self.data['y_test']

    def __getitem__(self, index):
        single_doc_features = self.features[index]
        single_doc_label = self.labels[index]
        if self.transforms is not None:
            single_doc_feature_tensor = self.transforms(single_doc_features)
        else:
            single_doc_feature_tensor = torch.from_numpy(single_doc_features).float()

        return single_doc_feature_tensor, single_doc_label

    def __len__(self):
        return len(self.labels)
