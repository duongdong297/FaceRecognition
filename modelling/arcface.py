import torch
import torch.nn as nn
import math

# 1. MOBILEFACENET BACKBONE 
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.PReLU(in_channels * expansion),
            nn.Conv2d(in_channels * expansion, in_channels * expansion, 3, stride, 1, groups=in_channels * expansion, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.PReLU(in_channels * expansion),
            nn.Conv2d(in_channels * expansion, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return x + self.conv(x) if self.connect else self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, groups=in_channels, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if not linear:
            self.prelu = nn.PReLU(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x if self.linear else self.prelu(x)

class MobileFacenet(nn.Module):
    def __init__(self, embedding_size=512):
        super(MobileFacenet, self).__init__()
        self.conv3 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv3 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.in_channels = 64
        
        # Standard MobileFaceNet Settings
        bottleneck_setting = [
            [2, 64, 5, 2], [4, 128, 1, 2], [2, 128, 6, 1], [4, 128, 1, 2], [2, 128, 2, 1]
        ]
        
        self.bottlenecks = self._make_layer(Bottleneck, bottleneck_setting)
        self.conv1 = ConvBlock(128, 512, 1, 1, 0)
        self.linear_GDConv7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        self.linear_conv1 = ConvBlock(512, embedding_size, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                layers.append(block(self.in_channels, c, s if i == 0 else 1, t))
                self.in_channels = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv3(x)
        x = self.dw_conv3(x)
        x = self.bottlenecks(x)
        x = self.conv1(x)
        x = self.linear_GDConv7(x)
        x = self.linear_conv1(x)
        x = x.view(x.shape[0], -1)
        return x
    


def build_model(weights_path, device):
    print(f" Loading ArcFace backbone from {weights_path}")
    model = MobileFacenet(embedding_size=512)
    
    try:
        # Load weights
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Xử lý prefix 'backbone.' nếu có
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        clean_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone.' in k}
        
        if not clean_state_dict:
            clean_state_dict = state_dict
            
        model.load_state_dict(clean_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading ArcFace weights: {e}")
        return None

    model.to(device)
    model.eval()
    return model