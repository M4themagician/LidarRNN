import torch
import torch.nn as nn
from network.building_blocks import ResidualBlock, DenseUpsamplingConvolution
from torchvision.ops import batched_nms


class LidarRNN(nn.Module):
    down_channels = [128, 256, 512, 1024]
    up_channels = [1024, 512]
    nms_position_radius = 1.5
    nms_iou_threshold = 0.1
    conf_threshold = 0.5
    max_objects = 50

    def __init__(self, output_width_px, pixel_scale):
        super().__init__()
        # setup const tensor with absolute map localization for object decoding
        xy = (0, output_width_px - 1)
        xy = [c - output_width_px / 2 for c in xy]
        xy = [(c + 0.5) * pixel_scale for c in xy]
        relative_coords_tensor = torch.linspace(
            xy[0], xy[1], output_width_px, dtype=torch.float32
        )
        grid_x, grid_y = torch.meshgrid(
            relative_coords_tensor, relative_coords_tensor, indexing="xy"
        )
        grid = torch.dstack([grid_x, grid_y])
        self.register_buffer("coordinate_grid", grid, persistent=False)

        self.output_width_px = output_width_px
        layers = []
        in_channels = 1
        for c in self.down_channels:
            layers.append(
                ResidualBlock(in_channels, c, nn.ReLU, depth=2, separable=True)
            )
            in_channels = c

        for c in self.up_channels:
            layers.append(DenseUpsamplingConvolution(in_channels, c, 2))
            in_channels = c
        self.layers = nn.Sequential(*layers)
        self.out_conv = nn.Sequential(
            ResidualBlock(
                2 * in_channels, 64, nn.ReLU, depth=2, downsample=False, separable=True
            ),
            nn.Conv2d(64, 2 + 9, kernel_size=1),
        )
        self.hidden_conv = nn.Conv2d(2 * in_channels, in_channels, 1)
        self.hidden_channels = in_channels

    def init_hidden(self):
        return torch.zeros(
            (1, self.hidden_channels, self.output_width_px, self.output_width_px)
        )

    def forward(self, x, hidden):
        features = torch.cat((self.layers(x), hidden), dim=1)
        hidden = self.hidden_conv(features)
        out = self.out_conv(features)
        return out, hidden

    def infer(self, x, hidden):
        x, hidden = self.forward(x, hidden)
        x = x.squeeze(0).permute(1, 2, 0)

        x[..., 2:4] = x[..., 2:4] + self.coordinate_grid
        x = torch.flatten(x, 0, 1)

        P, Class_idxs = torch.max(torch.softmax(x[..., :2], dim=-1), dim=-1)
        P = P[Class_idxs != 0]
        x = x[Class_idxs != 0, ...]

        xy_offsets = x[..., 2:4]
        cos_yaw = x[..., 4]
        sin_yaw = x[..., 5]
        lw_v_xy_yaw_rate = x[..., 6:]
        heading = torch.atan2(sin_yaw, cos_yaw)
        nms_boxes = torch.cat(
            (
                xy_offsets - self.nms_position_radius,
                xy_offsets + self.nms_position_radius,
            ),
            dim=1,
        )
        top_box_idxs = batched_nms(nms_boxes, P, 0 * P, self.nms_iou_threshold)
        top_k = top_box_idxs.size()[0]
        for k, i in enumerate(top_box_idxs):
            if P[i] < self.conf_threshold or k == self.max_objects - 1:
                top_k = k
                break
        top_box_idxs_threshed = top_box_idxs[:top_k]
        out_boxes = torch.cat(
            (P.unsqueeze(1), xy_offsets, heading.unsqueeze(1), lw_v_xy_yaw_rate), dim=1
        )
        top_boxes = torch.index_select(out_boxes, 0, top_box_idxs_threshed)

        return top_boxes, hidden
