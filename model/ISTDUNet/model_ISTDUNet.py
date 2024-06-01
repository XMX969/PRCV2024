import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

from .minet import miNet
from .resnet2020 import ResNet, Bottleneck, ResNetCt, BottleneckMode


class Down(nn.Module):
    def __init__(self,
                 inp_num = 1,
                 layers=[1, 2, 4, 8],
                 channels=[8, 16, 32, 64],
                 bottleneck_width=16,
                 stem_width=8,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 **kwargs
                 ):
        super(Down, self).__init__()

        # stemWidth = int(channels[0])
        stemWidth = int(8)
        self.stem = nn.Sequential(
            normLayer(1, affine=False),
            nn.Conv2d(1, stemWidth*2, kernel_size=3, stride=1, padding=1, bias=False),
            normLayer(stemWidth*2),
            activate()
        )
        self.down = ResNetCt(Bottleneck, layers, inp_num=inp_num,
                       radix=2, groups=4, bottleneck_width=bottleneck_width,
                       deep_stem=True, stem_width=stem_width, avg_down=True,
                       avd=True, avd_first=False, layer_parms=channels, **kwargs)

    def forward(self, x):
        # ret = []
        x = self.stem(x)
        # ret.append(x)
        x = self.down(x)
        ret=x
        return ret

class UPCt(nn.Module):
    def __init__(self, channels=[],
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU
                 ):
        super(UPCt, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(channels[0],
                      channels[1],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[1]),
            activate()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[1],
                      channels[2],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[2]),
            activate()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[2],
                      channels[3],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[3]),
            activate()
        )

    def forward(self, x):
        x1, x2, x3, x4 = x
        out = self.up1(x4)
        out = x3 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.up2(out)
        out = x2 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.up3(out)
        out = x1 + F.interpolate(out, scale_factor=2, mode='bilinear')
        return out

class Head(nn.Module):
    def __init__(self, inpChannel, oupChannel,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 # Dropout = 0.1
                 ):
        super(Head, self).__init__()
        interChannel = inpChannel // 4
        self.head = nn.Sequential(
            nn.Conv2d(inpChannel, interChannel,
                      kernel_size=3, padding=1,
                      bias=False),
            normLayer(interChannel),
            activate(),
            # nn.Dropout(),
            nn.Conv2d(interChannel, oupChannel,
                      kernel_size=1, padding=0,
                      bias=True)
        )

    def forward(self, x):
        return self.head(x)

class EDN(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        super(EDN, self).__init__()
        # it = lambda x: x

        # self.X1 = it
        # self.X2 = it
        # self.X3 = it
        from .eta import External_attention
        self.X1 = External_attention(channels[0])
        self.X2 = External_attention(channels[1])
        self.X3 = External_attention(channels[2])
        self.X4 = External_attention(channels[3])

        self.Conv1 = new_conv_block(channels[0], channels[0])
        self.Conv2 = new_conv_block(channels[1], channels[1])
        self.Conv3 = new_conv_block(channels[2], channels[2])
        self.Conv4 = new_conv_block(channels[3], channels[3])

    def forward(self, x):
        x1 ,x2, x3, x4 = x
        x1 = self.X1(x1)
        x2 = self.X2(x2)
        x3 = self.X3(x3)
        x4 = self.X4(x4)
        # x1 = self.Conv1(x1)
        # x2 = self.Conv2(x2)
        # x3 = self.Conv3(x3)
        # x4 = self.Conv4(x4)
        return [x1, x2, x3, x4]

class ISTDU_Net(miNet):
    def __init__(self, ):
        super(ISTDU_Net, self).__init__()
        self.encoder = None
        self.decoder = None

        # self.down = Down(channels=[8, 16, 32, 64])
        #
        # self.up = UPCt(channels=[256,128,64,32])

        self.down = Down(channels=[16, 32, 64, 128])
        # self.up = lambda x:x
        # self.up = UP(num_classes=num_classes, s=0.125)
        self.up = UPCt(channels=[512, 256,128,64])

        # self.head = Head(inpChannel=32, oupChannel=1)
        self.headDet = Head(inpChannel=64, oupChannel=1)

        self.headSeg = Head(inpChannel=64, oupChannel=1)
        # self.DN = DN()
        self.DN = EDN(channels=[64, 128, 256, 512])

    def funIndividual(self, x):
        x1 = self.down(x)
        return x1

    def funPallet(self, x):
        return x

    def funConbine(self, x):
        # ret = []
        # for i, j in zip(*x):
        #     ret.append(i+j)
        # return ret
        return x

    def funEncode(self, x):
        return x
        # return self.transformer(x)

    def funDecode(self, x):
        x = self.DN(x)
        x = self.up(x)
        return x

    def funOutput(self, x):
        # return self.head(x)
        # return torch.sigmoid(self.head(x))
        return torch.sigmoid(self.headSeg(x)) #torch.sigmoid(self.headDet(x)), 

def conv_relu_bn(in_channel, out_channel, dirate=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=dirate,
            dilation=dirate,
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class new_conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(new_conv_block, self).__init__()
        self.conv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 1),
            conv_relu_bn(in_ch, out_ch, 1),
            conv_relu_bn(out_ch, out_ch, 1),
        )
        self.cdc_layer = nn.Sequential(
            CDC_conv(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.dconv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 2),
            conv_relu_bn(in_ch, out_ch, 4),
            conv_relu_bn(out_ch, out_ch, 2),
        )
        self.final_layer = conv_relu_bn(out_ch * 3, out_ch, 1)

    def forward(self, x):
        conv_out = self.conv_layer(x)
        cdc_out = self.cdc_layer(x)
        dconv_out = self.dconv_layer(x)
        out = torch.concat([conv_out, cdc_out, dconv_out], dim=1)
        out = self.final_layer(out)
        return out

class CDC_conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        kernel_size=3,
        padding=1,
        dilation=1,
        theta=0.7,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        [c_out, c_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        diff_out = F.conv2d(
            input=x,
            weight=kernel_diff,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
        )
        out = norm_out - self.theta * diff_out
        return out
