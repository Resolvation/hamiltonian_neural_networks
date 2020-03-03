import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from vae import VAE
from utils import change_lr, linear_lr


class Data(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.load('../data/mass_spring.tar')

    def __getitem__(self, index):
        return self.data[index // 30, index % 30]

    def __len__(self):
        return self.data.shape[0] * 30


lr = 1e-4

trainloader = DataLoader(Data(), batch_size=30, shuffle=True, num_workers=4)

model = VAE().cuda()
model.training = True
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)

n_epochs = 100

for epoch in range(1, n_epochs + 1):
    epoch_loss = 0.

    change_lr(optimizer, lr * linear_lr(epoch, n_epochs))

    for image in trainloader:
        image = image.cuda()
        optimizer.zero_grad()

        rec, _, _ = model(image)
        loss = ((rec - image).pow(2)).sum()
        loss.backward()

        epoch_loss += loss.item()

        optimizer.step()

    if epoch % 10 == 0:
        image = image[0].cpu().detach()
        rec = rec[0].cpu().detach()
        diff = 10 * abs(image - rec)
        ToPILImage()(torch.cat((image, rec, diff), dim=2)).save(f'{epoch}.jpg')
        print(f'Epoch: {epoch}\tLoss: {epoch_loss / len(Data()):.04f}')
