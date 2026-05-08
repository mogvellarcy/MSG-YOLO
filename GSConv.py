class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p=None, g=g, act=act)

        self.cv2 = Conv(c_, c_, 5, 1, p=None, g=c_, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        out = torch.cat((x1, x2), 1)

        b, c, h, w = out.data.shape
        b_c = b * c // 2
        y = out.view(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w)
        return y
