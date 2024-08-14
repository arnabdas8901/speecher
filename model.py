import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class MaskConv(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        super(MaskConv, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,:,width//2:] = 0
        else:
            self.mask[:,:,:,(width//2)+1:] = 0


    def forward(self, x):
        self.weight.data*=self.mask
        return super(MaskConv, self).forward(x)
class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        elif self.layer_type == 'timehalf':
            return F.avg_pool2d(x, (1, 2))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        elif self.layer_type == 'semi':
            return F.interpolate(x, scale_factor=(1.25, 1), mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

class UpSampleContent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=(5, 1), mode='nearest')

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class UpResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, upsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.upsample = UpSample(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.upsample:
            x = self.upsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.upsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdaIN_FreqGroup(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features*5, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2*5)

    def forward(self, x, s):
        freq_splits = torch.chunk(x, 5, dim=-2)
        x = torch.cat(freq_splits, dim=1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        x = (1 + gamma) * self.norm(x) + beta
        channel_splits = torch.chunk(x, 5, dim=1)
        return torch.cat(channel_splits, dim=-2)

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        #self.conv1 = MaskConv("A", dim_in, dim_out, 5, 1, 2)
        #self.conv2 = MaskConv("B", dim_out, dim_out, 5, 1, 2)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 5, 1, 2)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 5, 1, 2)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=64, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out

    def forward(self, x):
        out = self.get_feature(x)[...,0]
        return out

class Discriminator2dA2M(nn.Module):
    def __init__(self, dim_in=64, max_conv_dim=384, repeat_num=4, num_domain=7):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, num_domain, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out

    def forward(self, x, y):
        out = self.get_feature(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


class Speecher(nn.Module):
    def __init__(self, args, asr_model, f0_model):
        super().__init__()
        self.asr_model = asr_model
        self.f0_model = f0_model
        self.f0_conv = nn.Sequential(
            ResBlk(args.F0_channel, args.F0_bneck_channel, normalize=True, downsample="half"),
        )
        #self.content_upsample = UpSampleContent()
        self.content_upsample = nn.Sequential(
            UpResBlk(args.content_bneck_channel, args.content_bneck_channel, upsample='timepreserve'),
            UpResBlk(args.content_bneck_channel, args.content_bneck_channel, upsample='timepreserve'),
            UpResBlk(args.content_bneck_channel, args.content_bneck_channel, upsample='semi'),
        )
        self.decode = nn.ModuleList()
        dim_out = 16
        for lid in range(args.final_decoder_repeat):
            if lid == 2 or lid == 0:
                _downtype = 'half'
            elif lid == 4:
                _downtype = 'none'
            else:
                _downtype = 'timepreserve'
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out//2, args.spk_embed_bneck_len,
                             upsample=_downtype))
            dim_out = dim_out * 2

        self.decode.insert(0, AdainResBlk(args.max_channel, args.max_channel // 2, args.spk_embed_bneck_len))
        dim_in = args.max_channel
        dim_out = args.max_channel
        for _ in range(args.initial_decoder_repeat):
            self.decode.insert(
                0, AdainResBlk(dim_in,dim_out , args.spk_embed_bneck_len))
            dim_in = args.content_bneck_channel + args.F0_bneck_channel

        self.spk_emb_net = nn.Sequential(nn.Linear(args.spk_embed_len, args.spk_embed_len),
                                         nn.ReLU(),
                                         nn.Linear(args.spk_embed_len, args.spk_embed_len),
                                         nn.ReLU(),
                                         nn.Linear(args.spk_embed_len, args.spk_embed_bneck_len)
                                         )

        self.cont_emb_net = nn.Sequential(ResBlk(args.content_channel, args.content_bneck_channel, normalize=True, downsample="timehalf"),
                                          )
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(8, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, 1, 1, 0),
        )

    def forward(self, x_real, spk_emb, training=True):
        # x_real --> b, 1, 80, 192
        # spl_emb --> b, 192
        content_emb = self.asr_model.get_feature(x_real).unsqueeze(-2) # b, 128, 1, 96
        #content_emb = torch.nn.functional.instance_norm(content_emb)
        #content_emb = torch.nn.functional.layer_norm(content_emb.permute(0, 3, 1, 2), [128, 1]).permute(0, 2, 3, 1)
        content_emb = self.cont_emb_net(content_emb) # b, content_bneck_channel, 1, 48
        content_emb = self.content_upsample(content_emb) # b, content_bneck_channel, 5, 48
        prsdy_emb = self.f0_model.get_feature_GAN(x_real) # b, F0_channel, 10, 192

        prsdy_emb = self.f0_conv(prsdy_emb) # b, F0_bneck_channel, 5, 96
        prsdy_emb = F.adaptive_avg_pool2d(prsdy_emb, [content_emb.shape[-2], content_emb.shape[-1]]) # b, F0_bneck_channel, 5, 48

        x = torch.cat([content_emb, prsdy_emb], axis=1) # b, content_bneck_channel+F0_bneck_channel, 5, 48
        """if training:
            e = 0.01 * torch.randn_like(x)
            x = x + e"""

        spk_emb = self.spk_emb_net(spk_emb) # b, spk_embed_bneck_len
        for block in self.decode:
            # after block0 --> b, 512, 5, 48
            # after block1 --> b, 512, 5, 48
            # after block2 --> b, 256, 5, 48
            # after block3 --> b, 128, 5, 48
            # after block4 --> b, 64, 10, 48
            # after block5 --> b, 32, 20, 96
            # after block6 --> b, 16, 40, 96
            # after block7 --> b, 8, 80, 192
            x = block(x, spk_emb)

        return self.to_out(x)

class Speecher_A2M(nn.Module):
    def __init__(self, args, asr_model, f0_model):
        super().__init__()
        self.asr_model = asr_model
        self.f0_model = f0_model
        self.f0_conv = nn.Sequential(
            ResBlk(args.F0_channel, args.F0_bneck_channel, normalize=True, downsample="half"),
        )
        #self.content_upsample = UpSampleContent()
        self.content_upsample = nn.Sequential(
            UpResBlk(args.content_bneck_channel, args.content_bneck_channel, upsample='timepreserve'),
            UpResBlk(args.content_bneck_channel, args.content_bneck_channel, upsample='timepreserve'),
            UpResBlk(args.content_bneck_channel, args.content_bneck_channel, upsample='semi'),
        )
        self.decode = nn.ModuleList()
        dim_out = 16
        for lid in range(args.final_decoder_repeat):
            if lid == 2 or lid == 0:
                _downtype = 'half'
            elif lid == 4:
                _downtype = 'none'
            else:
                _downtype = 'timepreserve'
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out//2, args.spk_embed_bneck_len,
                             upsample=_downtype))
            dim_out = dim_out * 2

        self.decode.insert(0, AdainResBlk(args.max_channel, args.max_channel // 2, args.spk_embed_bneck_len))
        dim_in = args.max_channel
        dim_out = args.max_channel
        for _ in range(args.initial_decoder_repeat):
            self.decode.insert(
                0, AdainResBlk(dim_in,dim_out , args.spk_embed_bneck_len))
            dim_in = args.content_bneck_channel + args.F0_bneck_channel

        self.spk_emb_net = nn.Sequential(nn.Linear(args.spk_embed_len, args.spk_embed_len),
                                         nn.ReLU(),
                                         nn.Linear(args.spk_embed_len, args.spk_embed_len),
                                         nn.ReLU(),
                                         nn.Linear(args.spk_embed_len, args.spk_embed_bneck_len),
                                         nn.ReLU()
                                         )
        self.spk_emb_net_unshared = nn.ModuleList()
        for _ in range(args.num_speakers):
            self.spk_emb_net_unshared += [nn.Linear(args.spk_embed_bneck_len, args.spk_embed_bneck_len)]

        self.cont_emb_net = nn.Sequential(ResBlk(args.content_channel, args.content_bneck_channel, normalize=True, downsample="timehalf"),
                                          )
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(8, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, 1, 1, 0),
        )

    def forward(self, x_real, spk_emb, spk_lbl, training=True):
        # x_real --> b, 1, 80, 192
        # spl_emb --> b, 192
        # spk_lbl --> b
        content_emb = self.asr_model.get_feature(x_real).unsqueeze(-2) # b, 128, 1, 96
        content_emb = self.cont_emb_net(content_emb) # b, content_bneck_channel, 1, 48
        content_emb = self.content_upsample(content_emb) # b, content_bneck_channel, 5, 48
        prsdy_emb = self.f0_model.get_feature_GAN(x_real) # b, F0_channel, 10, 192

        prsdy_emb = self.f0_conv(prsdy_emb) # b, F0_bneck_channel, 5, 96
        prsdy_emb = F.adaptive_avg_pool2d(prsdy_emb, [content_emb.shape[-2], content_emb.shape[-1]]) # b, F0_bneck_channel, 5, 48

        x = torch.cat([content_emb, prsdy_emb], axis=1) # b, content_bneck_channel+F0_bneck_channel, 5, 48
        """if training:
            e = 0.01 * torch.randn_like(x)
            x = x + e"""

        spk_emb = self.spk_emb_net(spk_emb) # b, spk_embed_bneck_len
        spk_out = []
        for layer in self.spk_emb_net_unshared:
            spk_out += [layer(spk_emb)]

        spk_out = torch.stack(spk_out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(spk_lbl.size(0))).to(spk_lbl.device)
        spk_emb = spk_out[idx, spk_lbl]  # (batch, style_dim)
        for block in self.decode:
            x = block(x, spk_emb)

        return self.to_out(x)

