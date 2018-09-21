import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
#from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

torch.manual_seed(123)


def plot_loss(loss_array,name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_array)
    plt.savefig('loss_'+name)


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def calc_gradient_penalty(netD, real_data, fake_data):
    LAMBDA=10
    BATCH_SIZE = real_data.size(0)
    DIM = real_data.size(2)
    device = 0

    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 1, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 1, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def run_trainer(train_loader, netD, netG, args):

    mode = args.mode

    free_params(netD)
    free_params(netG)

    

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5,0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4,betas=(0.5,0.9))

    #free_params(encoder)
    #free_params(decoder)
    #free_params(disc)
    
    #enc_optim = optim.Adam(encoder.parameters(), lr = args.lr)
    #dec_optim = optim.Adam(decoder.parameters(), lr = args.lr)
    #disc_optim = optim.Adam(disc.parameters(), lr = 0.5 * args.lr)

    #enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
    #G_scheduler = StepLR(optimizerG, step_size=30, gamma=0.5)
    #D_scheduler = StepLR(optimizerD, step_size=30, gamma=0.5)

    one = torch.Tensor([1])
    mone = one * -1

    one = one.cuda()
    mone = mone.cuda()


    noise = torch.FloatTensor(args.batch_size, args.n_z, 1, 1)
    noise = noise.cuda()
    noise = Variable(noise)
    

    netG.apply(weights_init)
    netD.apply(weights_init)

    for epoch in range(1000):
        G_loss_epoch = 0
        recon_loss_epoch = 0
        D_loss_epoch = 0
    
        step = 0

        data_iter = iter(train_loader)
        i = 0

        while(i<len(train_loader)): 
            j = 0
            while j < args.Diters and i < len(train_loader):
                
                j += 1



                free_params(netD)
                frozen_params(netG)
                netD.zero_grad()
                netG.zero_grad()

                images,labels = data_iter.next()

                i += 1


                # clamp parameters to a cube
                if(mode=='wgan'):
                    for p in netD.parameters():
                        p.data.clamp_(-0.01, 0.01)

                images = Variable(images)
                images = images.cuda()

                #train disc
                #train disc with real
                output = netD(images)
                errD_real = output.mean()
                #errD_real.backward(one)


                #train disc with fake

                noise = noise.resize_(args.batch_size, args.n_z, 1, 1).normal_(0,1)
                fake = netG(noise)
                output = netD(fake.detach()) 

                errD_fake = output.mean()

                if(mode=='wgan'):
                    errD_fake.backward(mone)
                    errD_real.backward(one)
                if(mode=='wgan-gp'):
                    errD_real.backward(mone)
                    errD_fake.backward(one)
                    gradient_penalty = calc_gradient_penalty(netD, images, fake)
                    gradient_penalty.backward()

                errD = errD_fake - errD_real #+ gradient_penalty


                optimizerD.step()

                D_loss_epoch += errD.data.cpu().item()

            #train G
            #might have to freeze disc params


            frozen_params(netD)
            free_params(netG)
            netG.zero_grad()
            
            noise = noise.resize_(args.batch_size, args.n_z, 1, 1).normal_(0,1)
            fake = netG(noise)


            errG = netD(fake)
            errG = errG.mean()

            if(mode=='wgan'):
                errG.backward(one)
            if(mode=='wgan-gp'):
                errG.backward(mone)
                
            optimizerG.step()

            G_loss_epoch += errG.mean().data.cpu().item()

            step += 1

            if step % 1 == 0 and i == 5 :
                print('saving images')
                save_image(fake[0:6].data.cpu().detach(), './recon.png')
                save_image(images[0:6].data.cpu().detach(), './orig.png')

                


        #recon_loss_array.append(recon_loss_epoch)
        #d_loss_array.append(d_loss_epoch)

        #if(epoch%5==0):
        #    print('plotting losses')
        #    plot_loss(recon_loss_array,'recon')
        #    plot_loss(d_loss_array, 'disc')

        if(epoch % 1 == 0):
            print("Epoch, G_loss, D_loss" 
                  ,epoch + 1, G_loss_epoch, D_loss_epoch)

