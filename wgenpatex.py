import torch
from torch import nn
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import model
from os import mkdir
from os.path import isdir

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def imread(img_name):
    """
    loads an image as torch.tensor on the selected device
    """ 
    np_img = plt.imread(img_name)
    tens_img = torch.tensor(np_img, dtype=torch.float, device=DEVICE)
    if torch.max(tens_img) > 1:
        tens_img/=255
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)
    if tens_img.shape[2] > 3:
        tens_img = tens_img[:,:,:3]
    tens_img = tens_img.permute(2,0,1)
    return tens_img.unsqueeze(0)

def imshow(tens_img):
    """
    shows a tensor image
    """ 
    np_img = np.clip(tens_img.squeeze(0).permute(1,2,0).data.cpu().numpy(), 0,1)
    if np_img.shape[2] < 3:
        np_img = np_img[:,:,0]
        ax = plt.imshow(np_img)
        ax.set_cmap('gray')
    else:
        ax = plt.imshow(np_img)
    plt.axis('off')
    return plt.show()

def imsave(save_name, tens_img):
    """
    save a tensor image
    """ 
    np_img = np.clip(tens_img.squeeze(0).permute(1,2,0).data.cpu().numpy(), 0,1)
    if np_img.shape[2] < 3:
        np_img = np_img[:,:,0]
    plt.imsave(save_name, np_img)
    return 

class gaussian_downsample(nn.Module):
    """
    Downsampling module with Gaussian filtering
    """ 
    def __init__(self, kernel_size, sigma, stride, pad=False):
        super(gaussian_downsample, self).__init__()
        self.gauss = nn.Conv2d(3, 3, kernel_size, stride=stride, groups=3, bias=False)        
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights.to(DEVICE)
        self.gauss.weight.requires_grad_(False)
        self.pad = pad
        self.padsize = kernel_size-1

    def forward(self, x):
        if self.pad:
            x = torch.cat((x, x[:,:,:self.padsize,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize]), 3)
        return self.gauss(x)

    def init_weights(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

class semidual(nn.Module):
    """
    Computes the semi-dual loss between inputy and inputx for the dual variable psi
    """    
    def __init__(self, inputy, device=DEVICE, usekeops=False):
        super(semidual, self).__init__()        
        self.psi = nn.Parameter(torch.zeros(inputy.shape[0], device=device))
        self.yt = inputy.transpose(1,0)
        self.usekeops = usekeops
        self.y2 = torch.sum(self.yt **2,0,keepdim=True)
    def forward(self, inputx):
        if self.usekeops:
            from pykeops.torch import LazyTensor
            y = self.yt.transpose(1,0)
            x_i = LazyTensor(inputx.unsqueeze(1).contiguous())
            y_j = LazyTensor(y.unsqueeze(0).contiguous())
            v_j = LazyTensor(self.psi.unsqueeze(0).unsqueeze(2).contiguous())
            sx2_i = LazyTensor(torch.sum(inputx**2,1,keepdim=True).unsqueeze(2).contiguous())
            sy2_j = LazyTensor(self.y2.unsqueeze(2).contiguous())
            rmv = sx2_i + sy2_j - 2*(x_i*y_j).sum(-1) - v_j
            amin = rmv.argmin(dim=1).view(-1)
            loss = torch.mean(torch.sum((inputx-y[amin,:])**2,1)-self.psi[amin]) + torch.mean(self.psi)
        else:
            cxy = torch.sum(inputx**2,1,keepdim=True) + self.y2 - 2*torch.matmul(inputx,self.yt)
            loss = torch.mean(torch.min(cxy - self.psi.unsqueeze(0),1)[0]) + torch.mean(self.psi)
        return loss
    
class gaussian_layer(nn.Module): 
    """
    Gaussian layer for the dowsampling pyramid
    """ 	   
    def __init__(self, gaussian_kernel_size, gaussian_std, stride = 2, pad=False):
        super(gaussian_layer, self).__init__()
        self.downsample = gaussian_downsample(gaussian_kernel_size, gaussian_std, stride, pad=pad)
    def forward(self, input):
        self.down_img = self.downsample(input)
        return self.down_img

class identity(nn.Module):  
    """
    Identity layer for the dowsampling pyramid
    """   
    def __init__(self):
        super(identity, self).__init__()
    def forward(self, input):
        self.down_img = input
        return input

def create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride = 2, pad=False):
    """
    Create a dowsampling Gaussian pyramid
    """ 
    layer = identity()
    gaussian_pyramid = nn.Sequential(layer)
    for i in range(n_scales-1):
        layer = gaussian_layer(gaussian_kernel_size, gaussian_std, stride, pad=pad)
        gaussian_pyramid.add_module("Gaussian_downsampling_{}".format(i+1), layer)
    return gaussian_pyramid

class patch_extractor(nn.Module):   
    """
    Module for creating custom patch extractor
    """ 
    def __init__(self, patch_size, pad=False):
        super(patch_extractor, self).__init__()
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size-1

    def forward(self, input, batch_size=0):
        if self.pad:
            input = torch.cat((input, input[:,:,:self.padsize,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize]), 3)
        patches = self.im2pat(input).squeeze(0).transpose(1,0)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx,:]
        return patches

def optim_synthesis(args):
    """
    Perform the texture synthesis of an examplar image
    """
    target_img_name = args.target_image_path
    patch_size = args.patch_size
    n_iter_max = args.n_iter_max
    n_iter_psi = args.n_iter_psi
    n_patches_in = args.n_patches_in
    n_patches_out = args.n_patches_out
    n_scales = args.scales
    usekeops = args.keops
    visu = args.visu
    save = args.save
    
    # fixed parameters
    monitoring_step=50
    saving_folder='tmp/'
    
    # parameters for Gaussian downsampling
    gaussian_kernel_size = 4
    gaussian_std = 1
    stride = 2
    
    # load image
    target_img = imread(target_img_name)
    
    # synthesized size
    if args.size is None:
        nrow = target_img.shape[2]
        ncol = target_img.shape[3]
    else:
        nrow = args.size[0]
        ncol = args.size[1]
    
    if save:
        if not isdir(saving_folder):
            mkdir(saving_folder)
        imsave(saving_folder+'original.png', target_img)

    # Create Gaussian Pyramid downsamplers
    target_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=False)                  
    input_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=True)
    target_downsampler(target_img) # evaluate on the target image

    # create patch extractors
    target_im2pat = patch_extractor(patch_size, pad=False)
    input_im2pat = patch_extractor(patch_size, pad=True)

    # create semi-dual module at each scale
    semidual_loss = []
    for s in range(n_scales):
        real_data = target_im2pat(target_downsampler[s].down_img, n_patches_out) # exctract at most n_patches_out patches from the downsampled target images 
        layer = semidual(real_data, device=DEVICE, usekeops=usekeops)
        semidual_loss.append(layer)
        if visu:
            imshow(target_downsampler[s].down_img)

    # Weights on scales
    prop = torch.ones(n_scales, device=DEVICE)/n_scales # all scales with same weight
    
    # initialize the generated image
    fake_img = torch.rand(1, 3, nrow,ncol, device=DEVICE, requires_grad=True)
    
    # intialize optimizer for image
    optim_img = torch.optim.Adam([fake_img], lr=0.01)
    
    # initialize the loss vector
    total_loss = np.zeros(n_iter_max)

    # Main loop
    t = time.time()
    for it in range(n_iter_max):
    
        # 1. update psi
        input_downsampler(fake_img.detach()) # evaluate on the current fake image
        for s in range(n_scales):            
            optim_psi = torch.optim.ASGD([semidual_loss[s].psi], lr=1, alpha=0.5, t0=1)
            for i in range(n_iter_psi):
                fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in)
                optim_psi.zero_grad()
                loss = -semidual_loss[s](fake_data)
                loss.backward()
                optim_psi.step()
            semidual_loss[s].psi.data = optim_psi.state[semidual_loss[s].psi]['ax']

        # 2. perform gradient step on the image
        optim_img.zero_grad()        
        tloss = 0
        for s in range(n_scales):
            input_downsampler(fake_img)           
            fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in)
            loss = prop[s]*semidual_loss[s](fake_data)
            loss.backward()
            tloss += loss.item()
        optim_img.step()

        # save loss
        total_loss[it] = tloss
    
        # save some results
        if it % monitoring_step == 0:
            print('iteration '+str(it)+' - elapsed '+str(int(time.time()-t))+'s - loss = '+str(tloss))
            if visu:
                imshow(fake_img)
            if save:
                imsave(saving_folder+'it'+str(it)+'.png', fake_img)

    print('DONE - total time is '+str(int(time.time()-t))+'s')

    if visu:
        plt.plot(total_loss)
        plt.show()
        if save:
            plt.savefig(saving_folder+'loss_multiscale.png')
        plt.close()
    if save:
        np.save(saving_folder+'loss.npy', total_loss)
    
    return fake_img

def learn_model(args):

    target_img_name = args.target_image_path
    patch_size = args.patch_size
    n_iter_max = args.n_iter_max
    n_iter_psi = args.n_iter_psi
    n_patches_in = args.n_patches_in
    n_patches_out = args.n_patches_out
    n_scales = args.scales
    usekeops = args.keops
    visu = args.visu
    save = args.save
    
    # fixed parameters
    monitoring_step=100
    saving_folder='tmp/'
    
    # parameters for Gaussian downsampling
    gaussian_kernel_size = 4
    gaussian_std = 1
    stride = 2
    
    # load image
    target_img = imread(target_img_name)

    if save:
        if not isdir(saving_folder):
            mkdir(saving_folder)
        imsave(saving_folder+'original.png', target_img)

    # Create Gaussian Pyramid downsamplers
    target_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=False)                  
    input_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=False)
    target_downsampler(target_img) # evaluate on the target image

    # create patch extractors
    target_im2pat = patch_extractor(patch_size, pad=False)
    input_im2pat = patch_extractor(patch_size, pad=False)

    # create semi-dual module at each scale
    semidual_loss = []
    for s in range(n_scales):
        real_data = target_im2pat(target_downsampler[s].down_img, n_patches_out) # exctract at most n_patches_out patches from the downsampled target images 
        layer = semidual(real_data, device=DEVICE, usekeops=usekeops)
        semidual_loss.append(layer)
        if visu:
            imshow(target_downsampler[s].down_img)
            #plt.pause(0.01)

    # Weights on scales
    prop = torch.ones(n_scales, device=DEVICE)/n_scales # all scales with same weight
    
    # initialize generator
    G = model.generator(n_scales)
    fake_img = model.sample_fake_img(G, target_img.shape, n_samples=1)

    # intialize optimizer for image
    optim_G = torch.optim.Adam(G.parameters(), lr=0.01)
    
    # initialize the loss vector
    total_loss = np.zeros(n_iter_max)

    # Main loop
    t = time.time()
    for it in range(n_iter_max):

        # 1. update psi
        fake_img = model.sample_fake_img(G, target_img.shape, n_samples=1)
        input_downsampler(fake_img.detach())
        
        for s in range(n_scales):            
            optim_psi = torch.optim.ASGD([semidual_loss[s].psi], lr=1, alpha=0.5, t0=1)
            for i in range(n_iter_psi):
                 # evaluate on the current fake image
                fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in)
                optim_psi.zero_grad()
                loss = -semidual_loss[s](fake_data)
                loss.backward()
                optim_psi.step()
            semidual_loss[s].psi.data = optim_psi.state[semidual_loss[s].psi]['ax']

        # 2. perform gradient step on the image
        optim_G.zero_grad()        
        tloss = 0
        input_downsampler(fake_img) 
        for s in range(n_scales):        
            fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in)
            loss = prop[s]*semidual_loss[s](fake_data)
            tloss += loss
        tloss.backward()
        optim_G.step()

        # save loss
        total_loss[it] = tloss.item()
    
        # save some results
        if it % monitoring_step == 0:
            print('iteration '+str(it)+' - elapsed '+str(int(time.time()-t))+'s - loss = '+str(tloss.item()))
            if visu:
                imshow(fake_img)
            if save:
                imsave(saving_folder+'it'+str(it)+'.png', fake_img)

    print('DONE - total time is '+str(int(time.time()-t))+'s')

    if visu:
        plt.plot(total_loss)
        plt.show()
        plt.pause(0.01)
        if save:
            plt.savefig(saving_folder+'loss.png')
        plt.close()
    if save:
        np.save(saving_folder+'loss.npy', total_loss)
        
    return G
