class MSGConv(nn.Module):
    """ Multi-Scale Ghost Convolution """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels (intrinsic features)

        self.primary_conv = Conv(c1, c_, k, s, p=None, g=g, act=act)

        self.split_c = c_ // 2

        self.cheap_dw1 = nn.Conv2d(
            self.split_c, self.split_c,
            kernel_size=3, stride=1, padding=1,
            groups=self.split_c, bias=False
        )

        self.cheap_dw2 = nn.Conv2d(
            self.split_c, self.split_c,
            kernel_size=5, stride=1, padding=2,
            groups=self.split_c, bias=False
        )

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x1 = self.primary_conv(x)

        # x1 shape: [B, c_, H, W]
        if self.split_c * 2 != x1.shape[1]:
            y = torch.cat([x1, x1], dim=1)
            return y

        x1_part1, x1_part2 = torch.split(x1, self.split_c, dim=1)

        y1 = self.cheap_dw1(x1_part1)
        y2 = self.cheap_dw2(x1_part2)

        y1 = self.act(y1)
        y2 = self.act(y2)

        # 5. 拼接：[本征特征, 3x3特征, 5x5特征]
        return torch.cat((x1, y1, y2), dim=1)
