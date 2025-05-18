""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.ln1=nn.LayerNorm([self.in_dim,32,32])
        self.inc = (DoubleConv(32,32,in_dim, hidden_dim//16))
        self.down1 = (Down(16,16,hidden_dim//16, hidden_dim//8))
        self.down2 = (Down(8,8,hidden_dim//8, hidden_dim//4))
        self.down3 = (Down(4,4,hidden_dim//4, hidden_dim//2))
        factor = 2 if bilinear else 1
        self.down4 = (Down(2,2,hidden_dim//2, hidden_dim // factor))
        self.up1 = (Up(4,4,hidden_dim, hidden_dim//2 // factor, bilinear))
        self.up2 = (Up(8,8,hidden_dim//2, hidden_dim//4 // factor, bilinear))
        self.up3 = (Up(16,16,hidden_dim//4, hidden_dim//8 // factor, bilinear))
        self.up4 = (Up(32,32,hidden_dim//8, hidden_dim//16, bilinear))
        self.outc = (OutConv(hidden_dim//16, out_dim))
        self.ln2=nn.LayerNorm([self.out_dim,32,32])

    def forward(self, x):
        x = self.ln1(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.ln2(x)
        return x

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)