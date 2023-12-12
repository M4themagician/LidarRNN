import cv2
import numpy as np
import torch



class LidarData():
    def __init__(self, width_px, pixel_scale, max_objects, min_spawn_distance, delta_t, debug = False):
        self.width_px = width_px
        self.pixel_scale = pixel_scale
        self.max_object = max_objects
        self.min_spawn_distance = min_spawn_distance
        self.delta_t = delta_t
        self.objects = []
        self.map = np.zeros((self.width_px, self.width_px), dtype=np.uint8)
        self.debug = debug
        
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
        cv2.imshow('Map', self.map)

    def add_new_object(self):
        new_object = BoxObject(self.min_spawn_distance, self.delta_t, self.get_width_meter)
        if new_object.has_left_counter >= new_object.max_trajectory_steps -1:
            return 
        new_object_trajectory = new_object.get_relevant_trajectory()
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
        # .               .         l, w
        # -----------------         cos(yaw), sin(yaw)
        #                           v_x, v_y
        #                           yaw rate

        ### afterwards scale (nearest) to network output size and subtract a tensor with relative cell locations
        # then each cell position is x_rel_to_cell, y_rel_to_cell
        



    def __len__(self):
        return 100000
            

            
        

    def world_to_pixel(self, xy):
        pixel_x = (xy[0] + (self.width_px*self.pixel_scale))/self.pixel_scale - self.width_px/2
        pixel_y = (xy[1] + (self.width_px*self.pixel_scale))/self.pixel_scale - self.width_px/2
        return np.array([pixel_x, pixel_y])
        
if __name__ == '__main__':
    from data.box_object import BoxObject
    #import imageio
    #image_lst = []
    debug = False
    write_video = False
    delta_t = 1/30
    max_trajectory_length = 1000
    map_size = 600
    max_objects = 50
    pixel_scale = 0.2 if debug else 0.075
    map_width_meter = 0.9*map_size/2*pixel_scale if debug else map_size/2*pixel_scale
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('video.mp4', fourcc= fourcc, fps=1/delta_t, frameSize=(map_size, map_size), isColor=False)
    map = LidarData(map_size, pixel_scale, max_objects, map_width_meter, delta_t, debug=debug)
    while True:
        map.step()
        map.draw()
        #image_lst.append(map.map.astype(np.uint8))
        if write_video:
            writer.write(map.map.astype(np.uint8))
        
        key = cv2.waitKey(1) #int(delta_t*1000))
        if key == 27:
            break
    
    if write_video:
        writer.release()

    #imageio.mimsave('video.gif', image_lst, fps=1/delta_t)







