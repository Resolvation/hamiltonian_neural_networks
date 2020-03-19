from torch.utils.data import Dataset


class HamiltonianDataset(Dataset):
    def __init__(self, n_samples, root):
        self.n_samples = n_samples
        self.root = root

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n_samples
