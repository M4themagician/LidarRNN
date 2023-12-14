import cv2
import numpy as np
import torch
import torch.nn.functional as F



class LidarData():
    def __init__(self, width_px, pixel_scale, max_objects, min_spawn_distance, delta_t, target_width, debug = False):
        self.width_px = width_px
        self.pixel_scale = pixel_scale
        self.max_object = max_objects
        self.min_spawn_distance = min_spawn_distance
        self.delta_t = delta_t
        self.objects = []
        self.map = np.zeros((self.width_px, self.width_px), dtype=np.uint8)
        self.target_width = target_width
        self.debug = debug

        world_pos_at_edges = self.pixel_to_world((0, self.width_px - 1))
        relative_coords_tensor = torch.linspace(world_pos_at_edges[0], world_pos_at_edges[1], self.width_px, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(relative_coords_tensor, relative_coords_tensor, indexing='xy')
        grid = torch.dstack([grid_x, grid_y])
        self.rel_coordinate_grid = grid
        print(grid)

        
    def get_width_meter(self):
        return self.width_px*self.pixel_scale

    def get_min_spawn_distance(self):
        return self.min_spawn_distance
    
    def draw_object(self, object):
        """ draw an object into the map"""

    def add_object(self, lidar_object):
        self.objects.append(lidar_object)

    def draw(self, wait = 0):
        self.map = np.zeros((self.width_px, self.width_px), dtype=np.uint8)
        if self.debug:
            cv2.circle(self.map, self.world_to_pixel((0,0)).astype(np.int32), 5, color=((255,255,255)))
            cv2.rectangle(self.map, self.world_to_pixel((-self.min_spawn_distance, -self.min_spawn_distance)).astype(np.int32), self.world_to_pixel((self.min_spawn_distance,self.min_spawn_distance)).astype(np.int32), (255, 255, 255), 3)

        for o in self.objects:
            o.draw(self)

    def add_new_object(self):
        new_object = BoxObject(self.min_spawn_distance, self.delta_t, self.get_width_meter)
        if new_object.has_left_counter >= new_object.max_trajectory_steps -1:
            return 
        new_object_trajectory = new_object.get_relevant_trajectory()
        if abs(new_object_trajectory[0, 2] - new_object_trajectory[-1, 2])/np.pi > 2:
            return
        new_object_trajectory_length = new_object_trajectory.shape[0]
        for o in self.objects:
            test_trajectory = o.get_relevant_trajectory()
            test_trajectory_length = test_trajectory.shape[0]
            for i in range(min(new_object_trajectory_length, test_trajectory_length)):
                xy_new = new_object_trajectory[i, :2]
                xy_test = test_trajectory[i, :2]
                if np.linalg.norm(xy_new - xy_test) < new_object.cover_radius + o.cover_radius:
                    return
        self.add_object(new_object)

    def step(self):
        for o in self.objects:
            o.step()
            if o.has_left():
                self.objects.remove(o)
            
        if len(self.objects) < self.max_object:
            tries = 0
            while tries < 1:
                self.add_new_object()
                tries += 1

    def __getitem__(self, idx):
        # each time a sample is drawn, advance one step and draw the map
        self.step()
        self.draw()
        item = {}
        item['input_tensor'] = torch.from_numpy(self.map)

        ### define matching tensor and regression target tensor (full map width)
        # fill with ignore tensor (just draw boxes filled)
        # then fill closest cell to each object (or all cells within the object) according to
        # -----------------
        # .               .
        # .               .
        # .      x        .     cell x contains:
        # .               .         x,y relative coordinates (not offsets!)
        # .               .         cos(yaw), sin(yaw)
        # -----------------         v_x, v_y
        #                           yaw rate
        #                           l, w
        #                           

        ### afterwards scale (nearest) to network output size and subtract a tensor with relative cell locations
        # then each cell position is x_rel_to_cell, y_rel_to_cell

        matching_tensor = np.zeros((self.width_px, self.width_px))
        regression_targets = np.zeros((self.width_px, self.width_px, 9))
        for obj in self.objects:
            state = obj.get_centered_pose()
            box = obj.get_enclosing_box()
            box = [self.world_to_pixel((box[0], box[1])).astype(np.int32), self.world_to_pixel((box[2], box[3])).astype(np.int32)]
            box = [c for b in box for c in b]
            
            xmin, ymin, xmax, ymax = box# [min(self.width_px, max(0,c)) for c in box]
            xmin -= 1
            ymin -= 1
            xmax += 1
            ymax += 1
            xmin = min(self.width_px, max(0,xmin))
            ymin = min(self.width_px, max(0,ymin))
            xmax = min(self.width_px, max(0,xmax))
            ymax = min(self.width_px, max(0,ymax))
            regression_targets[ymin:ymax, xmin:xmax, :2] = state[:2]
            regression_targets[ymin:ymax, xmin:xmax, 2:4] = np.array([np.cos(state[2]), np.sin(state[2])])
            regression_targets[ymin:ymax, xmin:xmax, 4:6] = np.array([state[3]*np.cos(state[2]), state[3]*np.sin(state[2])])
            regression_targets[ymin:ymax, xmin:xmax, 6] = state[3]/obj.wheelbase*np.tan(state[4])
            regression_targets[ymin:ymax, xmin:xmax, 7:9] = np.array([obj.length, obj.width])
            corners = obj.get_corners()
            corners_px = np.array([self.world_to_pixel(c + obj.state[:2]) for c in corners]).astype(np.int32)
            cv2.fillPoly(matching_tensor, pts=[corners_px],color=(255),lineType=cv2.LINE_8)


            #cv2.rectangle(self.map, box[0], box[1], color=(255, 255, 255), thickness=-1)
            #print(enclosing_box)
        matching_tensor = torch.from_numpy(matching_tensor)
        regression_targets = torch.from_numpy(regression_targets)
        #print(regression_targets.size(), self.rel_coordinate_grid.size())
        regression_targets[..., :2] = regression_targets[..., :2] - self.rel_coordinate_grid
        regression_targets[matching_tensor==0] = 0
        #print(torch.max(regression_targets[..., :2]), torch.max(regression_targets[..., 7:9]))

        item['classification_targets'] = F.interpolate(matching_tensor.unsqueeze(0).unsqueeze(0), (self.target_width, self.target_width), mode='nearest').squeeze(0).squeeze(0)
        item['regression_targets'] = F.interpolate(regression_targets.permute(2, 0, 1).unsqueeze(0), (self.target_width, self.target_width), mode='nearest').squeeze(0).permute(1, 2, 0)
        return item

        



    def __len__(self):
        return 100000
            

            
        

    def world_to_pixel(self, xy):
        pixel_x = (xy[0] + ((self.width_px-0.5)*self.pixel_scale))/self.pixel_scale - self.width_px/2
        pixel_y = (xy[1] + ((self.width_px-0.5)*self.pixel_scale))/self.pixel_scale - self.width_px/2
        return np.array([pixel_x, pixel_y])
    
    def pixel_to_world(self, xy):
        xy = [c - self.width_px/2 for c in xy]
        xy = [(c+0.5)*self.pixel_scale for c in xy]
        return xy

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data.box_object import BoxObject

    def plot_item(item):
        class_idxs = item['classification_targets'].numpy()
        regression_targets = item['regression_targets'].numpy()
        #rel_position = regression_targets[..., 2]
        #print(rel_position.min(), rel_position.max())
        rel_position = np.linalg.norm(regression_targets[..., :2], axis=-1)
        heatmap = np.uint8(np.interp(rel_position, (rel_position.min(), rel_position.max()), (0, 255)))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imshow("Regression Targets", heatmap)
        cv2.imshow("Class Targets", class_idxs)


    #import imageio
    #image_lst = []
    debug = False
    write_video = False
    delta_t = 1/30
    max_trajectory_length = 1000
    map_size = 600
    target_size = 600
    max_objects = 20
    pixel_scale = 0.2 if debug else 0.075
    map_width_meter = 0.9*map_size/2*pixel_scale if debug else map_size/2*pixel_scale
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('video.mp4', fourcc= fourcc, fps=1/delta_t, frameSize=(map_size, map_size), isColor=False)
    map = LidarData(map_size, pixel_scale, max_objects, map_width_meter, delta_t, target_size, debug=debug)

    k = 0
    while True:
        map.step()
        map.draw()
        item = map.__getitem__(0)
        if k > 30:
            plot_item(item)

        cv2.imshow('Input Image', map.map)
        #image_lst.append(map.map.astype(np.uint8))
        if write_video:
            writer.write(map.map.astype(np.uint8))
        
        key = cv2.waitKey(1) #int(delta_t*1000))
        if key == 27:
            break
        elif key > 0:
            key = cv2.waitKey(0)
        k += 1
    
    if write_video:
        writer.release()

    #imageio.mimsave('video.gif', image_lst, fps=1/delta_t)







