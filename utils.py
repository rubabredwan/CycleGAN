import random
import torch
import torchvision
import visdom
import time
import sys
import numpy as np
    
def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

def denormalize(img):
    return img / 2 + 0.5

class Logger():
    def __init__(self, n_epochs, batches_epoch, vis):
        self.viz = vis
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.prev_time = time.time()
        self.loss_windows = {}
        self.image_windows = {}

    def update_images(self, images):
        for title, image in images.items():
            image = denormalize(image[0])
            if title not in self.image_windows:
                self.image_windows[title] = self.viz.image(image, opts={'title': title})
            else:
                self.viz.image(image, win=self.image_windows[title])

    def update_losses(self, step, losses):

        self.epoch = step / self.batches_epoch
        self.batch = step % self.batches_epoch

        sys.stdout.write('\r%.4f Epoch %03d/%03d [%04d/%04d]' % (time.time()-self.prev_time, self.epoch, self.n_epochs, self.batch, self.batches_epoch))
        
        self.prev_time = time.time()

        for loss_name, loss in losses.items():   
            temp = loss.cpu().detach().numpy() / 250
            sys.stdout.write(' -- %s: %.4f' % (loss_name, temp))
            if loss_name not in self.loss_windows:
                self.loss_windows[loss_name] = self.viz.line(X=np.array([step]), Y=np.array([temp]), 
                                                                opts={'xlabel': 'steps', 'ylabel': loss_name, 'title': loss_name})
            else:
                self.viz.line(X=np.array([step]), Y=np.array([temp]), win=self.loss_windows[loss_name], update='append')

        sys.stdout.write('\n')


def imshow(img, vis):
    img = img / 2 + 0.5
    vis.image(img)        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
   
