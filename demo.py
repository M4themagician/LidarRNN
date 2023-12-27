import cv2
import torch
import numpy as np
from torch.backends import cudnn
from time import perf_counter
from data.lidar_data import LidarData
from data.geometry import rotate
from network.lidarrnn import LidarRNN

def draw_predictions(item, objects):
    input_image = item['input_tensor'].cpu().numpy()
    color_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    raw_image = color_image.copy()
    for det in objects.cpu().numpy():
        P, x, y, heading, l, w, vx, vy, yaw_rate = det
        if P < 0.95:
            continue
        corners = np.array([(l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)])
        corners = [rotate(c, heading) for c in corners]
        corners_px = np.array([dataset.world_to_pixel(c + np.array([x,y])) for c in corners]).astype(np.int32)
        cv2.polylines(color_image ,[corners_px],True,(0,255,0))
        line_start = dataset.world_to_pixel((x,y)).astype(np.int32)
        line_finish = dataset.world_to_pixel((x+vx,y+vy)).astype(np.int32)
        cv2.line(color_image, line_start, line_finish, (0, 0, 255), 3)
    color_image = np.concatenate((raw_image, color_image), axis = 1)
    #print(color_image.shape)
    return color_image.astype(np.uint8)

if __name__ == "__main__":
    
    cudnn.benchmark = True
    with torch.inference_mode():
        debug = False
        delta_t = 1/30
        map_size = 800
        target_size = 200
        max_objects = 40
        pixel_scale = 0.2 if debug else 0.075
        map_width_meter = 0.9*map_size/2*pixel_scale if debug else map_size/2*pixel_scale
        dataset = LidarData(map_size, pixel_scale, max_objects, map_width_meter, delta_t, target_size, debug=debug)
        net = LidarRNN(dataset.width_px//4, dataset.pixel_scale*4)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('video.mp4', fourcc= fourcc, fps=30, frameSize=(2*map_size, map_size))
        
        net.load_state_dict(torch.load('weights.pth'))
        hidden = net.init_hidden().cuda()
        net = net.cuda()
        counter = 1
        frequency = 100
        times = []
        while True:
            item = dataset.__getitem__(0)
            for key, v in item.items():
                item[key] = v.cuda()
            input = item['input_tensor'].unsqueeze(0).unsqueeze(0)
            torch.cuda.synchronize()
            t0 = perf_counter()
            prediction, hidden = net.infer(input, hidden)
            torch.cuda.synchronize()
            t1 = perf_counter()
            times.append(t1-t0)
            vis = draw_predictions(item, prediction)
            writer.write(vis)
            counter += 1
            if counter % frequency == 0:
                print(f'Average inference time: {np.mean(times)}')
                times = []
            
    