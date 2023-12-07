import cv2
import numpy as np



class LidarData():
    def __init__(self, width_px, pixel_scale, max_objects, min_spawn_distance, delta_t):
        self.width_px = width_px
        self.pixel_scale = pixel_scale
        self.max_object = max_objects
        self.min_spawn_distance = min_spawn_distance
        self.delta_t = delta_t
        self.objects = []
        self.map = np.zeros((self.width_px, self.width_px), dtype=np.uint8)
        #cv2.circle(self.map, self.world_to_pixel((0,0)).astype(np.int32), 5, color=((255,255,255)))
        #cv2.rectangle(self.map, self.world_to_pixel((-self.min_spawn_distance, -self.min_spawn_distance)).astype(np.int32), self.world_to_pixel((self.min_spawn_distance,self.min_spawn_distance)).astype(np.int32), (255, 255, 255), 3)

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
        for o in self.objects:
            o.draw(self)
        cv2.imshow('Map', self.map)

    def step(self):
        for o in self.objects:
            if o.has_left():
                self.objects.remove(o)
            o.step()
        if len(self.objects) < self.max_object:
            self.add_object(BoxObject(self.min_spawn_distance, self.delta_t, self.get_width_meter))
        

    def world_to_pixel(self, xy):
        pixel_x = (xy[0] + (self.width_px*self.pixel_scale))/self.pixel_scale - self.width_px/2
        pixel_y = (xy[1] + (self.width_px*self.pixel_scale))/self.pixel_scale - self.width_px/2
        return np.array([pixel_x, pixel_y])
        
if __name__ == '__main__':
    from data.box_object import BoxObject

    delta_t = 1/30
    map = LidarData(600, 0.05, 20, 300*0.05, delta_t)
    test_object = BoxObject(map.get_min_spawn_distance(), delta_t, map.get_width_meter())
    map.add_object(test_object)
    while True:
        map.draw()
        map.step()
        key = cv2.waitKey(int(delta_t*1000))
        if key == 27:
            break







