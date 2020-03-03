import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from vae import VAE


class Data(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.load('../data/mass_spring.tar')

    def __getitem__(self, index):
        return self.data[index // 30, index % 30]

    def __len__(self):
        return self.data.shape[0] * 30


trainloader = DataLoader(Data(), batch_size=30, shuffle=True, num_workers=4)

model = VAE().cuda()
model.training = True
optimizer = optim.SGD(model.parameters(), lr=1e-5, weight_decay=1e-5)

n_epochs = 100

for epoch in range(n_epochs):
    epoch_loss = 0.

    for image in trainloader:
        image = image.cuda()
        optimizer.zero_grad()

        rec, _, _ = model(image)
        loss = ((rec - image).pow(2)).sum()
        loss.backward()

        epoch_loss += loss.item()

        optimizer.step()

    if epoch % 10 == 9:
        ToPILImage()(torch.cat((image[0].cpu().detach(), rec[0].cpu().detach()), dim=2)).save(f'{epoch + 1}.jpg')
        print(f'Epoch: {epoch + 1}\tLoss: {epoch_loss / len(Data()):.04f}')


