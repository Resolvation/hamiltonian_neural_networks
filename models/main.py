from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader

from data.mass_spring import MassSpring
from logger import Logger
from vae import VAE
from utils import change_lr, linear_lr


n_epochs = 100
lr = 1e-4

trainloader = DataLoader(MassSpring('vae'),
                         batch_size=30, shuffle=True, num_workers=4)

model = VAE().cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)


logger = Logger('../logs')

for epoch in tqdm(range(1, n_epochs + 1)):
    epoch_loss = 0.

    epoch_lr = lr * linear_lr(epoch, n_epochs)
    change_lr(optimizer, epoch_lr)

    for image in trainloader:
        image = image.cuda()
        optimizer.zero_grad()

        rec, _, _ = model(image)
        loss = ((rec - image).pow(2)).sum()
        loss.backward()

        epoch_loss += loss.item()

        optimizer.step()

    logger.log(epoch, epoch_lr, epoch_loss / len(trainloader.dataset))
    logger.save_image(epoch, [(image[0], rec[0])])

    if epoch % 20 == 0:
        logger.save_pth(epoch, model)
