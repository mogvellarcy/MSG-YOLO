class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        assert style in ['lp', 'pl']

        if style == 'pl':
            assert in_channels >= scale ** 2 * 2

        if style == 'lp':
            out_channels = 2 * groups * scale ** 2
            self.offset = nn.Conv2d(in_channels, out_channels, 1)
        else:
            out_channels = 2 * groups * scale ** 2
            self.offset = nn.Conv2d(in_channels // (scale ** 2), out_channels, 1)

        self.pixel_shuffle = nn.PixelShuffle(scale)

        normal_init(self.offset, std=0.001)

    def forward(self, x):
        B, C, H, W = x.size()

        offset = self.offset(x)

        offset = self.pixel_shuffle(offset)  # [B, 2*groups, H*scale, W*scale]
        offset = offset.reshape(B, self.groups, 2, H * self.scale, W * self.scale)

        dev = x.device
        dtype = x.dtype

        pos_x = torch.linspace(-1, 1, steps=W * self.scale, dtype=dtype, device=dev)
        pos_y = torch.linspace(-1, 1, steps=H * self.scale, dtype=dtype, device=dev)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')

        grid = torch.stack((grid_x, grid_y), dim=-1)  # [H*scale, W*scale, 2]
        grid = grid.unsqueeze(0).unsqueeze(1)  # [1, 1, H*scale, W*scale, 2]

        offset = offset.permute(0, 1, 3, 4, 2)

        offset_normalized = torch.empty_like(offset)

        offset_normalized[..., 0] = offset[..., 0] / (W * self.scale) * 2  # X轴
        offset_normalized[..., 1] = offset[..., 1] / (H * self.scale) * 2  # Y轴

        sample_grid = grid + offset_normalized

        sample_grid = sample_grid.reshape(B * self.groups, H * self.scale, W * self.scale, 2)

        x_reshaped = x.reshape(B, self.groups, C // self.groups, H, W)
        x_reshaped = x_reshaped.reshape(B * self.groups, C // self.groups, H, W)

        out = F.grid_sample(
            x_reshaped,
            sample_grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=False
        )

        return out.reshape(B, C, H * self.scale, W * self.scale)
