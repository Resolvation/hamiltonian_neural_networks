from torch import optim
from torch.utils.data import DataLoader

from data import MassSpring, Pendulum
from logger import Logger
from models import HNN, VAE
from utils import change_lr, linear_lr


dataset = 'mass_spring'
model = 'hnn'
lr = 3e-5
n_epochs = 400


def beta(epoch):
    return 0 if epoch < n_epochs // 4 else 0.01


def hnn_loss(image, rec, mu, logvar, epoch):
    return (rec - image).pow(2).sum() / 30 \
            + beta(epoch) * (mu.pow(2) + logvar.exp() - logvar).sum()


if dataset == 'mass_spring':
    dataset = MassSpring
elif dataset == 'pendulum':
    dataset = Pendulum
else:
    raise ValueError('Wrong dataset name.')

if model == 'vae':
    model_name = 'vae'
    model = VAE
elif model == 'hnn':
    model_name = 'hnn'
    model = HNN
else:
    raise ValueError('Wrong model.')

trainloader = DataLoader(dataset(model_name, n_samples=2000, verbose=True),
                         batch_size=30, shuffle=True, num_workers=4)
testloader = DataLoader(dataset(model_name, n_samples=200, verbose=True),
                         batch_size=10, shuffle=False, num_workers=4)

model = model().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)

logger = Logger(verbose=True)

for epoch in range(1, n_epochs + 1):
    epoch_loss = 0.

    epoch_lr = lr * linear_lr(epoch, n_epochs)
    change_lr(optimizer, epoch_lr)

    for image in trainloader:
        image = image.cuda()
        optimizer.zero_grad()

        rec, mu, logvar = model(image)
        if model_name == 'vae':
            loss = ((rec - image).pow(2)).sum()
        elif model_name == 'hnn':
            loss = hnn_loss(image, rec, mu, logvar, epoch)
        loss.backward()

        epoch_loss += loss.item()

        optimizer.step()

    logger.log(epoch, epoch_lr, epoch_loss / len(trainloader.dataset))
    if model_name == 'vae':
        logger.save_image(epoch, [(image[0], rec[0])])
    elif model_name == 'hnn':
        images = [(image[0, i: i + 3], rec[0, i: i + 3]) for i in range(0, 90, 3)]
        logger.save_image(epoch, images)

    if epoch % 20 == 0:
        logger.save_pth(epoch, model)

if model_name == 'hnn':
    model.eval()

    total_loss = 0

    for i, image in enumerate(testloader):
        image = image.cuda()

        rec, mu, logvar = model(image)

        total_loss += hnn_loss(image, rec, mu, logvar, n_epochs)

        images = [(image[0, i: i + 3], rec[0, i: i + 3]) for i in range(0, 90, 3)]
        logger.save_image(f'test_{i}', images)

    logger.log('test', -1, total_loss / len(testloader.dataset))
