import torch
from torch import nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, module_builders: dict, in_dim=3, out_dim=1):
        super().__init__()

        self.encoder1 = nn.Sequential(
            *module_builders['enc1'](in_dim, 64),
        )
        self.encoder2 = nn.Sequential(
            *module_builders['enc2'](64, 128),
        )
        self.encoder3 = nn.Sequential(
            *module_builders['enc3'](128, 256),
        )
        self.encoder4 = nn.Sequential(
            *module_builders['enc4'](256, 512),
        )
        self.encoder5 = nn.Sequential(
            *module_builders['enc5'](512, 1024),
        )

        self.transpose: nn.ModuleList = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
        ])

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *module_builders['dec1'](1024, 512),
                ),
                nn.Sequential(
                    *module_builders['dec2'](512, 256),
                ),
                nn.Sequential(
                    *module_builders['dec3'](256, 128),
                ),
                nn.Sequential(
                    *module_builders['dec4'](128, 64),
                )
            ]
        )

        self.clf = nn.Conv2d(64, out_dim, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))
        tr1 = self.transpose[0](enc5)
        dec1 = self.decoders[0](torch.cat([enc4, tr1], 1))

        tr2 = self.transpose[1](dec1)
        dec2 = self.decoders[1](torch.cat([enc3, tr2], 1))

        tr3 = self.transpose[2](dec2)
        dec3 = self.decoders[2](torch.cat([enc2, tr3], 1))

        tr4 = self.transpose[3](dec3)
        dec4 = self.decoders[3](torch.cat([enc1, tr4], 1))
        return self.clf(dec4)
