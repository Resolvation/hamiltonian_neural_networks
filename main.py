from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader

from data import MassSpring, Pendulum
from logger import Logger
from models import HNN, VAE
from utils import change_lr, linear_lr


dataset = 'mass_spring'
model = 'hnn'
lr = 1.5e-4
n_epochs = 600
input_length = 3 * 25
output_length = 3 * 30


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

trainloader = DataLoader(dataset(model_name),
                         batch_size=100, shuffle=True, num_workers=4)

model = model(input_length, output_length).cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)


logger = Logger()

for epoch in tqdm(range(1, n_epochs + 1)):
    epoch_loss = 0.

    epoch_lr = lr   # * linear_lr(epoch, n_epochs)
    # change_lr(optimizer, epoch_lr)

    for image in trainloader:
        image = image.cuda()
        optimizer.zero_grad()

        rec, _, _ = model(image[:, : input_length])
        if model_name == 'vae':
            loss = ((rec - image).pow(2)).sum()
        elif model_name == 'hnn':
            loss = (rec[:, : input_length] - image[:, : input_length]).pow(2).sum()
        loss.backward()

        epoch_loss += loss.item()

        optimizer.step()

    if model_name == 'hnn':
        epoch_loss /= input_length

    logger.log(epoch, epoch_lr, epoch_loss / len(trainloader.dataset))
    if model_name == 'vae':
        logger.save_image(epoch, [(image[0], rec[0])])
    elif model_name == 'hnn':
        images = [(image[0, i: i + 3], rec[0, i: i + 3]) for i in range(0, output_length, 3)]
        logger.save_image(epoch, images)

    if epoch % 20 == 0:
        logger.save_pth(epoch, model)
