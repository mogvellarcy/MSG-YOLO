class LightDetect(Detect):
    def __init__(self, nc: int = 80, reg_max=1, end2end=False, ch: tuple = (), *args):
        if (not ch or isinstance(ch, int)) and len(args) > 0:
            if isinstance(args[-1], (list, tuple)):
                ch = args[-1]

        super().__init__(nc=nc, ch=ch)

        self.reg_max = reg_max
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        del self.cv2, self.cv3
        if hasattr(self, 'one2one_cv2'): del self.one2one_cv2
        if hasattr(self, 'one2one_cv3'): del self.one2one_cv3
        c_coupled = max(ch[0] // 4, 32)

        self.cv = nn.ModuleList(
            nn.Sequential(
                DWConv(x, x, 3),
                Conv(x, c_coupled, 1),
                DWConv(c_coupled, c_coupled, 3),
                Conv(c_coupled, c_coupled, 1),

                nn.Conv2d(c_coupled, 4 + self.nc, 1)
            ) for x in ch
        )

        self.one2one_cv = copy.deepcopy(self.cv) if end2end else None

    @property
    def end2end(self):
        return self.one2one_cv is not None

    def forward_head(self, x, box_head=None, cls_head=None):
        head_module = box_head
        bs = x[0].shape[0]
        box_list = []
        cls_list = []
        for i in range(self.nl):
            out = head_module[i](x[i])
            b_out, c_out = out.split((4, self.nc), dim=1)
            box_list.append(b_out.view(bs, 4, -1))
            cls_list.append(c_out.view(bs, self.nc, -1))
        return dict(boxes=torch.cat(box_list, dim=-1), scores=torch.cat(cls_list, dim=-1), feats=x)

    @property
    def one2many(self):
        return dict(box_head=self.cv, cls_head=None)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv, cls_head=None)

    def bias_init(self):
        m = self
        heads = [self.cv]
        if self.end2end: heads.append(self.one2one_cv)
        for head in heads:
            for i, module in enumerate(head):  
                module[-1].bias.data[:4] = 1.0
                module[-1].bias.data[4:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
