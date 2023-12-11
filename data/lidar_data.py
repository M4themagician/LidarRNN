import cv2
import numpy as np



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

    def step(self):
        for o in self.objects:
            o.step()
            if o.has_left():
                self.objects.remove(o)
            
        if len(self.objects) < self.max_object:
            new_object = BoxObject(self.min_spawn_distance, self.delta_t, self.get_width_meter)
            if new_object.has_left_counter == new_object.max_trajectory_steps:
                return 
            new_object_trajectory = new_object.get_relevant_trajectory()
            new_object_trajectory_length = new_object_trajectory.shape[0]
            collides = False
            for o in self.objects:
                test_trajectory = o.get_relevant_trajectory()
                test_trajectory_length = test_trajectory.shape[0]
                for i in range(min(new_object_trajectory_length, test_trajectory_length)):
                    xy_new = new_object_trajectory[i, :2]
                    xy_test = test_trajectory[i, :2]
                    if np.linalg.norm(xy_new - xy_test) < new_object.cover_radius + o.cover_radius:
                        return

            self.add_object(new_object)
        

    def world_to_pixel(self, xy):
        pixel_x = (xy[0] + (self.width_px*self.pixel_scale))/self.pixel_scale - self.width_px/2
        pixel_y = (xy[1] + (self.width_px*self.pixel_scale))/self.pixel_scale - self.width_px/2
        return np.array([pixel_x, pixel_y])
        
if __name__ == '__main__':
    from data.box_object import BoxObject

    debug = False
    write_video = False
    delta_t = 1/30
    max_trajectory_length = 1000
    map_size = 600
    max_objects = 50
    pixel_scale = map_size/4*0.05 if debug else map_size/2*0.05
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('video.mp4', fourcc= fourcc, fps=1/delta_t, frameSize=(map_size, map_size), isColor=False)
    map = LidarData(map_size, 0.05, max_objects, pixel_scale, delta_t, debug=debug)
    test_object = BoxObject(map.get_min_spawn_distance(), delta_t, map.get_width_meter(), max_trajectory_steps=max_trajectory_length)
    map.add_object(test_object)
    while True:
        map.draw()
        if write_video:
            writer.write(map.map.astype(np.uint8))
        map.step()
        key = cv2.waitKey(1) #int(delta_t*1000))
        if key == 27:
            break
    
    if write_video:
        writer.release()







