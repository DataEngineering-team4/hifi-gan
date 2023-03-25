import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            self._create_weight_norm_conv1d(channels, channels, kernel_size, dilation=dilation[0]),
            self._create_weight_norm_conv1d(channels, channels, kernel_size, dilation=dilation[1]),
            self._create_weight_norm_conv1d(channels, channels, kernel_size, dilation=dilation[2])
        ])
        self.convs1.apply(self._init_weights)

        self.convs2 = nn.ModuleList([
            self._create_weight_norm_conv1d(channels, channels, kernel_size, dilation=1),
            self._create_weight_norm_conv1d(channels, channels, kernel_size, dilation=1),
            self._create_weight_norm_conv1d(channels, channels, kernel_size, dilation=1)
        ])
        self.convs2.apply(self._init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, negative_slope=LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, negative_slope=LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            self._remove_weight_norm(l)
        for l in self.convs2:
            self._remove_weight_norm(l)

    def _create_weight_norm_conv1d(self, in_channels, out_channels, kernel_size, dilation=1):
        return weight_norm(Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,
                                  padding=get_padding(kernel_size, dilation)))

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _remove_weight_norm(self, module):
        if isinstance(module, weight_norm):
            module = remove(module) 
        return module

class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.Sequential(
            nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                   padding=get_padding(kernel_size, dilation[0]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                   padding=get_padding(kernel_size, dilation[1])))
            ]),
            nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                   padding=get_padding(kernel_size, 1))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                   padding=get_padding(kernel_size, 1)))
            ])
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for convs in self.convs:
            for c in convs:
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c(xt)
                x = xt + x
        return x

    def remove_weight_norm(self):
        for convs in self.convs:
            for l in convs:
                remove_weight_norm(l)

class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        
        # Initial convolution layer
        self.conv_pre = nn.Sequential(
            weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3)),
            nn.LeakyReLU(LRELU_SLOPE)
        )
        
        # Upsample layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)
            ))
            self.ups.apply(init_weights)

        # Residual blocks
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.resblocks.apply(init_weights)

        # Final convolution layer
        self.conv_post = nn.Sequential(
            weight_norm(Conv1d(ch, 1, 7, 1, padding=3)),
            nn.Tanh()
        )
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                resblock_idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[resblock_idx](x)
                else:
                    xs += self.resblocks[resblock_idx](x)
            x = xs / self.num_kernels
            x = nn.LeakyReLU(LRELU_SLOPE)(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre[0])
        remove_weight_norm(self.conv_post[0])        

class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        
        conv_layers = [
            nn.Conv2d(1, 32, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0)),
            nn.Conv2d(32, 128, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0)),
            nn.Conv2d(128, 512, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0)),
            nn.Conv2d(512, 1024, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0)),
            nn.Conv2d(1024, 1024, kernel_size=(kernel_size, 1), stride=1, padding=(2, 0)),
            nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        ]
        
        if use_spectral_norm:
            conv_layers = [spectral_norm(l) for l in conv_layers]
        
        self.convs = nn.Sequential(*conv_layers)

    def forward(self, x):
        fmap = []
        
        # pad sequence to multiple of period
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        
        x = x.view(b, c, t // self.period, self.period)
        
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, negative_slope=LRELU_SLOPE)
            fmap.append(x)
        
        x = x.view(b, -1)
        
        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, kernel_size=5, stride=3, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period, kernel_size, stride, use_spectral_norm)
            for period in self.periods
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2))
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(start_dim=1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(kernel_size=4, stride=2, padding=2),
            AvgPool1d(kernel_size=4, stride=2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        loss += torch.mean(torch.abs(dr - dg))
    return loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((dr - 1)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((dg - 1)**2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses