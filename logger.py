from datetime import datetime
import os

import torch
from torchvision.transforms import ToPILImage


class Logger:
    def __init__(self, root='logs'):
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)
        self.name = str(datetime.now())
        self.path = os.path.join(root, self.name)
        os.makedirs(self.path)
        self.main = os.path.join(self.path, 'main.log')
        os.mknod(self.main)

    def save_image(self, epoch, orig, rec):
        orig = orig.cpu().detach()
        rec = rec.cpu().detach()
        diff = 10 * abs(orig - rec)
        log_image = ToPILImage()(torch.cat((orig, rec, diff), dim=2))
        log_image.save(os.path.join(self.path, f'{epoch}.jpg'))

    def save_pth(self, epoch, model):
        torch.save(model.state_dict(), os.path.join(self.path, f'{epoch}.pth.tar'))

    def log(self, epoch, lr, loss):
        with open(self.main, 'a') as f:
            f.write(f'Epoch: {epoch}\tlr: {lr:.03e}\tLoss: {loss:.04f}\n')

