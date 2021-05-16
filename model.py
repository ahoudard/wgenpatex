import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator's convolutional blocks 2D
class Conv_block2D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block2D, self).__init__()

        self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(n_ch_out, momentum=m)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

# Up-sampling block
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    
# Up-sampling + batch normalization block
class Up_Bn2D(nn.Module):
    def __init__(self, n_ch):
        super(Up_Bn2D, self).__init__()

        self.up = Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm2d(n_ch)

    def forward(self, x):
        x = self.bn(self.up(x))
        return x

# The whole network
class generator(nn.Module):
    def __init__(self, nlayers=5, ch_in=3, ch_step=2, device=DEVICE):
        super(generator, self).__init__()

        self.ch_in = ch_in
        self.nlayers = nlayers

        self.first_conv = Conv_block2D(ch_in,ch_step).to(device)
        self.cb1 = nn.ModuleList()
        self.cb2 = nn.ModuleList()
        self.up = nn.ModuleList()
        
        for n in range(0, nlayers):
            self.up.append(Up_Bn2D((n+1)*ch_step).to(device))
            self.cb1.append(Conv_block2D(ch_in,ch_step).to(device))
            self.cb2.append(Conv_block2D((n+2)*ch_step,(n+2)*ch_step).to(device))
        
        self.last_conv = nn.Conv2d((nlayers+1)*ch_step, 3, 1, padding=0, bias=False).to(device)

    def forward(self, z):

        nlayers=self.nlayers
        y = self.first_conv(z[0])
        for n in range(0,nlayers):
            y = self.up[n](y)
            y = torch.cat((y, self.cb1[n](z[n+1])), 1)
            y = self.cb2[n](y)
        y = self.last_conv(y)
        return y

# Function to generate an output sample
def sample_fake_img(G, size, n_samples=1):
    # dimension of the first input noise
    strow = int(np.ceil(size[2]-2)/2**G.nlayers)
    stcol = int(np.ceil(size[3]-2)/2**G.nlayers)
    # input noise and forward pass
    ztab = [torch.rand(n_samples, G.ch_in, 8+2**k*strow+4*int(k!=0), 8+2**k*stcol+4*int(k!=0), device=DEVICE, dtype=torch.float) for k in range(0, G.nlayers+1)]
    Z = [Variable(z) for z in ztab]
    return G(Z)

#def sample_fake_img(G, target_size, n_samples=1):
#    sample_size = max(2**(np.ceil(np.log2(target_size[2]))), 2**G.nlayers)
#    print(sample_size)
#    zk = [torch.rand(n_samples,G.ch_in, int(sample_size/(2**k)), int(sample_size/(2**k)), device=DEVICE, dtype=torch.float) for k in range(0, G.nlayers+1)]
#    print(zk)
#    for z in zk:
#        print(z.shape)
#    Z = [Variable(z) for z in zk ]
#    return G(Z)
