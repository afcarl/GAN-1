from __future__ import print_function
import argparse
import random
import numpy as np

#torch framework
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import models.dcgan as dcgan 

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True)
parser.add_argument('--cuda', action='store_true', help='cuda enable')
parser.add_argument('--ngpu', default=1, help='number of gpus to use' )
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--scaleSize', type=int, default=64, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=64, help='crop size')
parser.add_argument('--nz', type=int, default=64)
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch', type=int, default=10000, help='num of training epoch')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--lr_decay', type=int, default=1000, help='decay lr this epoch')
parser.add_argument('--lambda_k', type=float, default=0.001, help='learning rate fo k')
parser.add_argument('--gamma', type=float, default=0.5, help='balance between D and G')
parser.add_argument('--hidden_size', type=int, default=64, help='bottleneck dimension of Discriminator')
parser.add_argument('--experiment', default=None, help='where to store samples and models')

opt = parser.parse_args()
print(opt)

if opt.cuda:
     torch.cuda.manual_seed_all(random.randint(1,10000))

cudnn.benchmark = True

####### load the data #######
dataset = datasets.ImageFolder(root=opt.dataroot,
		           transform=transforms.Compose([
			       transforms.Scale(opt.scaleSize),
			       transforms.CenterCrop(opt.fineSize),
                               transforms.ToTensor()
                               #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                           ])
					)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(2))


###### model #######

ngf = opt.ngf
ndf = opt.ndf
nz = opt.nz
nc = 3

netG = dcgan.Generator(nc, ngf, nz, opt.fineSize)
netD = dcgan.Discriminator(nc, ndf, opt.hidden_size, opt.fineSize)
if(opt.cuda):
    netG.cuda()
    netD.cuda()

#### setup optimizer #####
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

criterion = nn.L1Loss()

####### Variables ##########
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
real = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
label = torch.FloatTensor(1)

noise = Variable(noise)
real = Variable(real)
label = Variable(label)
if(opt.cuda):
    noise = noise.cuda()
    real = real.cuda()
    label = label.cuda()
    criterion.cuda()


####### Train #######
def adjust_learning_rate(optimizer, niter):
    """Sets the lr to the initial LR decayed by  10 every 30 epochs"""
    lr = opt.lr * (0.95 ** (niter // opt.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

k = 0
gan_iterations = 0
for epoch in range(1, opt.epoch+1):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        images = data_iter.next()
        i += 1
        gan_iterations += 1
        #### trainining D
        data_cpu, _ = images
        netD.zero_grad()
     
        real.data.resize_(data_cpu.size()).copy_(data_cpu)
        noise.data.resize_(data_cpu.size(0), opt.nz)
        noise.data.uniform_(-1,1)
         
        fake = netG(noise)
        fake_recons = netD(fake.detach())
        real_recons = netD(real)
   
        errD_real = torch.mean(torch.abs(real_recons-real))
        errD_fake = torch.mean(torch.abs(fake_recons-fake))

        errD = errD_real - k*errD_fake
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        fake = netG(noise)
        fake_recons = netD(fake)
        errG = torch.mean(torch.abs(fake_recons-fake))
        errG.backward()
        optimizerG.step()

        balance = (opt.gamma * errD_real - errD_fake).data[0]
        k = min(max(k + opt.lambda_k * balance, 0), 1)
        measure = errD_real.data[0] + np.abs(balance)

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Measure: %.4f K: %.4f learning rate: %.8f'
                  % (epoch, opt.epoch, errD.data[0], errG.data[0], measure, k, optimizerD.param_groups[0]['lr']))
       
        #### lr decay
        optimizerD = adjust_learning_rate(optimizerD, gan_iterations)
        optimizerG = adjust_learning_rate(optimizerG, gan_iterations)
        #### visualization
        if(gan_iterations % 1000 == 0):
            vutils.save_image(fake.data,
                        '%s/fake_samples_iteration_%03d.png' % (opt.experiment, gan_iterations))
            vutils.save_image(real.data,
              '%s/real_samples_iteration_%03d.png' % (opt.experiment, gan_iterations))
    if(epoch % 1000 == 0):
        torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.experiment, gan_iterations))
        torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.experiment, gan_iterations))
