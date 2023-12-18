import torch
import torch.optim as opt
import cv2
import numpy as np

from data.lidar_data import make_lidar_data
from data.geometry import rotate
from network.lidarrnn import LidarRNN
from loss.loss import BoxTrackingLoss




def train():
    dataset = make_lidar_data()
    net = LidarRNN(dataset.width_px//4, dataset.pixel_scale*4).cuda()
    loss_fn = BoxTrackingLoss().cuda()
    optimizer = opt.AdamW(net.parameters(), lr=1e-4)

    running_loss = 0
    iteration = 0
    print_n = 1000 
    show = True
        
    while True:
        optimizer.zero_grad()
        item = dataset.__getitem__(0)
        for key, v in item.items():
            item[key] = v.cuda()
        prediction = net(item['input_tensor'].unsqueeze(0).unsqueeze(0))
        loss = loss_fn.forward(prediction, item)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iteration += 1
        if iteration % print_n == 0:
            print(f"Loss after {iteration} iterations: {running_loss/print_n}")
            running_loss = 0
            if show:
                with torch.inference_mode():
                    item = dataset.__getitem__(0)
                    objects = net.infer(item['input_tensor'].unsqueeze(0).unsqueeze(0)).cpu().numpy()
                    print(objects.shape)
                    input_image = item['input_tensor'].cpu().numpy()
                    color_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                    for det in objects:
                        P, x, y, heading, l, w = det
                        corners = np.array([(l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)])
                        corners = [rotate(c, heading) for c in corners]
                        corners_px = np.array([dataset.world_to_pixel(c + np.array([x,y])) for c in corners]).astype(np.int32)
                        cv2.polylines(color_image ,[corners_px],True,(0,255,0))
                    cv2.imshow("Prediction", color_image)
                    key = cv2.waitKey(1)
            
        


if __name__ == "__main__":
    train()