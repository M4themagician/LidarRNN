from data.lidar_data import make_lidar_data, LidarData
from network.lidarrnn import LidarRNN
from loss.loss import BoxTrackingLoss

import torch.optim as opt
import torch.utils.data as data


def train():
    dataset = make_lidar_data()
    net = LidarRNN().cuda()
    loss_fn = BoxTrackingLoss().cuda()
    optimizer = opt.AdamW(net.parameters(), lr=1e-4)

    running_loss = 0
    iteration = 0
    while True:
        item = dataset.__getitem__(0)
        for key, v in item.items():
            item[key] = v.cuda()
        prediction = net(item['input_tensor'].unsqueeze(0).unsqueeze(0))
        loss = loss_fn.forward(prediction, item)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iteration += 1
        if iteration % 100 == 0:
            print(f"Loss after {iteration} iterations: {running_loss/100}")
            running_loss = 0
        


if __name__ == "__main__":
    train()