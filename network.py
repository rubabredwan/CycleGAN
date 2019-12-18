import torch
import torch.nn as nn
import torch.nn.functional as F

def disc_block(in_nc, out_nc, kernel_size, stride, norm, padding):
    layers = []
    layers.append(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding))
    if norm:
        layers.append(nn.InstanceNorm2d(out_nc))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
        model = []
        
        model += disc_block(3, 64, kernel_size=4, stride=2, norm=False, padding=1)
        model += disc_block(64, 128, kernel_size=4, stride=2, norm=True, padding=1)
        model += disc_block(128, 256, kernel_size=4, stride=2, norm=True, padding=1)
        model += disc_block(256, 512, kernel_size=4, stride=1, norm=True, padding=1)
        model += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, x.shape[-1])
        return x.view(x.shape[0])
    
class Residual_block(nn.Module):
    def __init__(self, nc):
        super(Residual_block, self).__init__()

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=0),
                 nn.InstanceNorm2d(nc),
                 nn.ReLU(inplace=True),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=0),
                 nn.InstanceNorm2d(nc)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)
    
class Generator(nn.Module):
    def __init__(self, nr):
        super(Generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 64, kernel_size=7, stride=1),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(inplace=True)]

        for _ in range(nr):
            model += [Residual_block(256)]

        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, kernel_size=7, stride=1),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
