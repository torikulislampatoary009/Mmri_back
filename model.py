import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Classic U-Net: 4 down blocks, bottleneck, 4 up blocks with skip connections.
    Output is raw logits (apply sigmoid for probabilities)."""

    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.downs = nn.ModuleList()
        prev = in_ch
        for ch in c:
            self.downs.append(DoubleConv(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c[-1], c[-1] * 2)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev = c[-1] * 2
        for ch in reversed(c):
            self.ups.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.up_convs.append(DoubleConv(ch * 2, ch))
            prev = ch
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([skip, x], dim=1))
        return self.head(x)
