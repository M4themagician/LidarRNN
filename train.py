import torch
import torch.optim as opt
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import cv2
import numpy as np

from data.lidar_data import make_lidar_data
from data.geometry import rotate
from network.lidarrnn import LidarRNN
from loss.loss import BoxTrackingLoss


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    print(f"Running basic DDP on rank {rank}.")
    setup(rank, world_size)
    torch.set_num_threads(1)
    dataset = make_lidar_data()
    device_id = rank % torch.cuda.device_count()
    net = LidarRNN(dataset.width_px // 4, dataset.pixel_scale * 4).to(device_id)
    net = DDP(net, device_ids=[device_id], find_unused_parameters=True)
    # net.load_state_dict(torch.load("weights_new.pth", map_location=f"cuda:{device_id}"))
    hidden = net.module.init_hidden().to(device_id)
    loss_fn = BoxTrackingLoss().to(device_id)
    optimizer = opt.AdamW(net.parameters(), lr=1e-4 / world_size)
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=(100000, 500000, 750000), gamma=1 / 10)

    running_loss = 0
    iteration = 0
    print_n = 5000
    show = False

    dist.barrier()

    while True:
        optimizer.zero_grad(set_to_none=True)
        item = dataset.__getitem__(0)
        for key, v in item.items():
            item[key] = v.to(device_id)
        input = item["input_tensor"].unsqueeze(0).unsqueeze(0)
        prediction, hidden = net(input, hidden)
        loss = loss_fn.forward(prediction, item)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        iteration += 1
        scheduler.step()
        if iteration % print_n == 0 and rank == 0:
            print(f"Loss after {iteration} iterations: {running_loss/world_size/print_n}")
            running_loss = 0
            torch.save(net.state_dict(), "weights_new.pth")
            if show:
                with torch.inference_mode():
                    objects, _ = net.infer(item["input_tensor"].unsqueeze(0).unsqueeze(0), hidden.detach())
                    objects = objects.cpu().numpy()
                    input_image = item["input_tensor"].cpu().numpy()
                    color_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                    for det in objects:
                        P, x, y, heading, l, w, vx, vy, yaw_rate = det
                        corners = np.array(
                            [
                                (l / 2, w / 2),
                                (l / 2, -w / 2),
                                (-l / 2, -w / 2),
                                (-l / 2, w / 2),
                            ]
                        )
                        corners = [rotate(c, heading) for c in corners]
                        corners_px = np.array([dataset.world_to_pixel(c + np.array([x, y])) for c in corners]).astype(np.int32)
                        cv2.polylines(color_image, [corners_px], True, (0, 255, 0))
                        line_start = dataset.world_to_pixel((x, y)).astype(np.int32)
                        line_finish = dataset.world_to_pixel((x + vx, y + vy)).astype(np.int32)
                        cv2.line(color_image, line_start, line_finish, (0, 0, 255), 3)

                    cv2.imshow("Prediction", color_image)
                    key = cv2.waitKey(1)
        hidden.detach_()


def run(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    run(train, 12)
