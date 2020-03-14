from datetime import datetime
import os

import torch
from torchvision.transforms import ToPILImage


class Logger:
    def __init__(self, root='logs', name=None, verbose=False):
        self.root = root
        if name is None:
            self.name = str(datetime.now())
        else:
            self.name = name
        self.verbose = verbose

        if not os.path.exists(root):
            os.makedirs(root)
        self.path = os.path.join(root, self.name)
        os.makedirs(self.path)
        self.main = os.path.join(self.path, 'main.log')
        os.mknod(self.main)

    def save_image(self, epoch, doublets):
        images = []
        for orig, rec in doublets:
            orig = orig.cpu().detach()
            rec = rec.cpu().detach()
            diff = 10 * abs(orig - rec)
            images.append(torch.cat((orig, rec, diff), dim=1))
        log_image = ToPILImage()(torch.cat(images, dim=2))
        log_image.save(os.path.join(self.path, f'{epoch}.jpg'))

    def save_pth(self, epoch, model):
        torch.save(model.state_dict(), os.path.join(self.path, f'{epoch}.pth.tar'))

    def log(self, epoch, lr, loss):
        line = f'Epoch: {epoch}\tlr: {lr:.03e}\tLoss: {loss:.04f}\n'
        if self.verbose:
            print(line[: -1])
        with open(self.main, 'a') as f:
            f.write(line)

