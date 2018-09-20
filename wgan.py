import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.res_stride = 1
        self.res_dropout_ratio = 0.0


        self.fourth = nn.Sequential(
            nn.ConvTranspose2d(self.n_z, 40, 3, 1, 0, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(True)
            )


        self.third = nn.Sequential(nn.ConvTranspose2d(40, 20, 5, 2, 1, bias=False),
                                   nn.BatchNorm2d(20),
                                   nn.LeakyReLU(0.2, inplace=True)
                                   )

        self.second = nn.Sequential(nn.ConvTranspose2d(20,10,4,2,1,bias=False),
                                    nn.BatchNorm2d(10),
                                    nn.LeakyReLU(0.2,inplace=True)
                                    ) #14x14
            
        self.first = nn.Sequential(nn.ConvTranspose2d(10,1,4,2,1,bias=False),
                                   nn.Sigmoid()
                                   )

    def forward(self, x):
        x = x.view(-1, self.n_z, 1, 1)
        x = self.fourth(x.view(-1, self.n_z, 1, 1))
        x = self.third(x)
        x = self.second(x)
        x = self.first(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.first = nn.Sequential(nn.Conv2d(self.n_channel, 10, 4, 2, 1),
                                   nn.BatchNorm2d(10),
                                   nn.LeakyReLU(0.2,inplace=True)
                                   )
        self.second = nn.Sequential(nn.Conv2d(10,20,4,2,1),
                                    nn.BatchNorm2d(20),
                                    nn.LeakyReLU(0.2,inplace=True),
                                    )

        self.third = nn.Sequential(nn.Conv2d(20,40,5,2,1),
                                   nn.BatchNorm2d(40),
                                   nn.LeakyReLU(0.2,inplace=True))

        self.fourth = nn.Sequential(nn.Conv2d(40,self.n_z,3,1,0),
                                    nn.BatchNorm2d(self.n_z),
                                    nn.ReLU()
                                    )

        self.fifth = nn.Conv2d(self.n_z,1,1,1,0)


        
    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        x = self.third(x)
        x = self.fourth(x)
        x = self.fifth(x)
        return  x



def return_model(args):
    decoder = Decoder(args)
    disc = Discriminator(args)

    decoder = decoder.cuda()
    disc = disc.cuda()

    print('return model - decoder.cuda(), disc.cuda()')

    return decoder, disc
