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

    def log(self, step, losses=None, images=None):

        self.epoch = step / self.batches_epoch
        self.batch = step % self.batches_epoch

        sys.stdout.write('\r%.4f Epoch %03d/%03d [%04d/%04d]' % (time.time()-self.prev_time, self.epoch, self.n_epochs, self.batch, self.batches_epoch))
        
        self.prev_time = time.time()

        for loss_name, loss in losses.items():   
            temp = loss.cpu().detach().numpy()
            sys.stdout.write(' -- %s: %.4f' % (loss_name, temp))
            if loss_name not in self.loss_windows:
                self.loss_windows[loss_name] = self.viz.line(X=np.array([step]), Y=np.array([temp]), 
                                                                opts={'xlabel': 'steps', 'ylabel': loss_name, 'title': loss_name})
            else:
                self.viz.line(X=np.array([step]), Y=np.array([temp]), win=self.loss_windows[loss_name], update='append')

        A2B = torch.cat((images['real_A'][0:8], images['fake_B'][0:8]), 0)
        B2A = torch.cat((images['real_B'][0:8], images['fake_A'][0:8]), 0)
        A2B = torchvision.utils.make_grid(denormalize(A2B), nrow=8, padding=2)
        B2A = torchvision.utils.make_grid(denormalize(B2A), nrow=8, padding=2)
        if len(self.image_windows) == 0:
            self.image_windows['A2B'] = self.viz.image(A2B, opts={'title':'A to B'})
            self.image_windows['B2A'] = self.viz.image(B2A, opts={'title':'B to A'})
        else:
            self.viz.image(A2B, win = self.image_windows['A2B'], opts={'title':'A to B'})
            self.viz.image(B2A, win = self.image_windows['B2A'], opts={'title':'B to A'})
            
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
   
