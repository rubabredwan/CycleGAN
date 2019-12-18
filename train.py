import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import utils
from network import Generator, Discriminator

import visdom
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--load_pretrained', type=int, default=0)
opt = parser.parse_args()

vis = visdom.Visdom()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_A = datasets.ImageFolder('~/data/ukiyoe2photo/A', transform=trans)
train_B = datasets.ImageFolder('~/data/ukiyoe2photo/B', transform=trans)
loader_A = torch.utils.data.DataLoader(train_A, batch_size=1, shuffle=True)
loader_B = torch.utils.data.DataLoader(train_B, batch_size=1, shuffle=True)

G_A2B = Generator(9).to(device)
G_B2A = Generator(9).to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

if opt.load_pretrained:
    G_A2B.load_state_dict(torch.load('pretrained/G_A2B.pth'))
    G_B2A.load_state_dict(torch.load('pretrained/G_B2A.pth'))
    D_A.load_state_dict(torch.load('pretrained/D_A.pth'))
    D_B.load_state_dict(torch.load('pretrained/D_B.pth'))
else:
    G_A2B.apply(weights_init)
    G_B2A.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


G_params = list(G_A2B.parameters()) + list(G_B2A.parameters())
D_params = list(D_A.parameters()) + list(D_B.parameters())
optimizer_G = optim.Adam(G_params, lr=0.0002, betas=[0.5, 0.999])
optimizer_D = optim.Adam(D_params, lr=0.0002, betas=[0.5, 0.999])

fake_A_buffer = utils.ReplayBuffer()
fake_B_buffer = utils.ReplayBuffer()
logger = utils.Logger(opt.n_epochs, len(loader_A), vis)

step = 0

losses = {'loss_G': 0, 'loss_G_identity': 0, 'loss_G_GAN': 0, 'loss_G_cycle': 0, 'loss_D': 0}

for epoch in range(1, opt.n_epochs+1):
    for (A, B) in zip(enumerate(loader_A), enumerate(loader_B)):

        step = step + 1

        real_A = A[1][0].to(device)
        real_B = B[1][0].to(device)
        target_real = torch.ones(real_A.shape[0]).to(device)
        target_fake = torch.zeros(real_B.shape[0]).to(device)

        # Generator update

        optimizer_G.zero_grad()

        # Identity Loss
        same_B = G_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 2.5
        same_A = G_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 2.5
        losses['loss_G_identity'] += loss_identity_A + loss_identity_B

        # GAN Loss
        fake_B = G_A2B(real_A)
        pred_fake = D_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        fake_A = G_B2A(real_B)
        pred_fake = D_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        losses['loss_G_GAN'] += loss_GAN_A2B + loss_GAN_B2A

        # Cycle Reconstruction Loss
        recovered_A = G_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
        recovered_B = G_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
        losses['loss_G_cycle'] += loss_cycle_ABA + loss_cycle_BAB

        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        losses['loss_G'] += loss_G
        loss_G.backward()
        
        optimizer_G.step()

        # Discriminator update

        optimizer_D.zero_grad()

        pred_real = D_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        fake_A = fake_A_buffer.push_and_pop(fake_A).to(device)
        pred_fake = D_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        loss_D_A = (loss_D_real + loss_D_fake) * 0.5

        pred_real = D_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        fake_B = fake_B_buffer.push_and_pop(fake_B).to(device)
        pred_fake = D_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D = loss_D_A + loss_D_B
        losses['loss_D'] += loss_D
        loss_D.backward()
        
        optimizer_D.step()


        # Update visdom
        if (step % 250 == 0):
            logger.update_images({'fake_A': fake_A, 'fake_B': fake_B})
            logger.update_losses(step, losses)
            losses = {'loss_G': 0, 'loss_G_identity': 0, 'loss_G_GAN': 0, 'loss_G_cycle': 0, 'loss_D': 0}


        
    torch.save(G_A2B.state_dict(), 'pretrained/G_A2B.pth')
    torch.save(G_B2A.state_dict(), 'pretrained/G_B2A.pth')
    torch.save(D_A.state_dict(), 'pretrained/D_A.pth')
    torch.save(D_B.state_dict(), 'pretrained/D_B.pth')

    for param_group in optimizer_G.param_groups:
        param_group['lr'] -= 0.0002 / 50
    for param_group in optimizer_D.param_groups:
        param_group['lr'] -= 0.0002 / 50
