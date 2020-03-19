from torch.utils.data import Dataset


class HamiltonianDataset(Dataset):
    def __init__(self, mode, n_samples, root):
        if mode not in {'vae', 'hnn'}:
            raise ValueError('Wrong mode.')
        self.mode = mode
        self.n_samples = n_samples
        self.root = root

    def __getitem__(self, index):
        if self.mode == 'vae':
            return self.data[index // 30, index % 30]
        elif self.mode == 'hnn':
            return self.data[index].view(-1, 32, 32)

    def __len__(self):
        if self.mode == 'vae':
            return 30 * self.n_samples
        elif self.mode == 'hnn':
            return self.n_samples
