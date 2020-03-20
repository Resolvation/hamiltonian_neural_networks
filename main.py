import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader

from data import MassSpring, Pendulum, TwoBody
from logger import Logger
from models import HNN
from utils import change_lr, linear_lr


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-e', '--epochs', default=400, type=int)
parser.add_argument('-lr', '--learning_rate', default=3e-4, type=float)
parser.add_argument('-b', '--beta', default=1e-3, type=float)
parser.add_argument('-bs', '--batch_size', default=20, type=int)
args = parser.parse_args()


if args.dataset == 'mass_spring':
    dataset = MassSpring
elif args.dataset == 'pendulum':
    dataset = Pendulum
elif args.dataset == 'two_body':
    dataset = TwoBody
else:
    raise ValueError('Wrong dataset name.')


def hnn_loss(image, rec, mu, logvar):
    return (rec - image).pow(2).sum() / 30 \
           + args.beta * (mu.pow(2) + logvar.exp() - logvar).sum()


trainloader = DataLoader(dataset(n_samples=4000), batch_size=args.batch_size,
                         shuffle=True, num_workers=4)
testloader = DataLoader(dataset(n_samples=400), batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

model = HNN().cuda()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

logger = Logger(verbose=True)

for epoch in range(1, args.epochs + 1):
    epoch_loss = 0.

    epoch_lr = args.learning_rate * linear_lr(epoch, args.epochs)
    change_lr(optimizer, epoch_lr)

    for image in trainloader:
        optimizer.zero_grad()
        image = image.cuda()

        rec, mu, logvar = model(image)
        loss = hnn_loss(image, rec, mu, logvar)

        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

    logger.log(epoch, epoch_lr, epoch_loss / len(trainloader.dataset))

    images = [(image[0, i: i + 3], rec[0, i: i + 3]) for i in range(0, 90, 3)]

    if epoch % 5 == 0:
        logger.save_image(epoch, images)

    if epoch % 20 == 0:
        logger.save_pth(epoch, model)

with torch.no_grad():
    total_loss = 0

    for i, image in enumerate(testloader):
        image = image.cuda()

        rec, mu, logvar = model(image)
        total_loss += hnn_loss(image, rec, mu, logvar)

        images = [(image[0, i: i + 3], rec[0, i: i + 3]) for i in range(0, 90, 3)]
        logger.save_image(f'test_{i}', images)

    logger.log('test', -1, total_loss / len(testloader.dataset))
