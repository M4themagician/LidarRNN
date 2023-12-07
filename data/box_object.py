import numpy as np
import cv2
import math
import random

from data.lidar_data import LidarData
from data.geometry import rotate


class BoxObject():
    length_range = (2.69, 6) #m
    width_range = (1.66, 2.1) #m
    wheelbase_range = (1.87, 3.365) #m
    roundness_range = (0, 1)
    initial_heading_range = (-5*math.pi/180, 5*math.pi/180) #rad
    initial_speed_range = (0, 20) #m/s
    box_constraints_mins = np.array((-1e20, -1e20, -1e20, -100, -50*math.pi/180))
    box_constraints_maxs =  np.array((1e20, 1e20, 1e20, 100, 50*math.pi/180))
    def __init__(self, min_spawn_distance, delta_t, map_width_meter, max_trajectory_steps = 1000):
        self.length = random.uniform(*self.length_range)
        self.width = random.uniform(*self.width_range)
        self.wheelbase = random.uniform(max(self.wheelbase_range[0], self.length_range[0]), min(self.wheelbase_range[1], self.length) )
        self.rear_axle_from_rear_end = random.uniform(0, self.length - self.wheelbase)
        self.roundness = random.uniform(*self.roundness_range)
        self.delta_t = delta_t
        self.state = np.zeros(5)
        self.max_trajectory_steps = max_trajectory_steps
        self.trajectory = np.zeros((max_trajectory_steps, 5))
        self.cover_radius = math.sqrt((self.width/2)**2 + (self.length/2)**2)
        self.has_entered_counter = 0
        self.has_entered_counter_set = False
        self.min_spawn_distance = min_spawn_distance
        self.time_counter = 0
        self.has_left_counter = max_trajectory_steps
        self.init_state() # x, y, heading, speed, steering angle, steering_angle_acceleration, acceleration
        self.get_trajectory()

    def init_state(self):
        cover_radius = self.cover_radius
        x,y = 0,0 
        while max(abs(x), abs(y)) < self.min_spawn_distance + self.cover_radius + self.length/2:
            x = random.uniform(-(self.min_spawn_distance + 3*cover_radius), self.min_spawn_distance + 3*cover_radius)
            y = random.uniform(-(self.min_spawn_distance + 3*cover_radius), self.min_spawn_distance + 3*cover_radius)
        self.state[0] = x
        self.state[1] = y
        # initial heading should point towards map center 
        init_heading = math.atan2(-y, -x) + random.uniform(*self.initial_heading_range)
        #xy_offset = (0,0) #(-self.length/2 + self.rear_axle_from_rear_end, 0)
        #xy_offset_rotated = rotate(xy_offset, init_heading)
        # x = x - (self.length - self.rear_axle_from_rear_end)*(math.cos(init_heading) - math.sin(init_heading))
        # y = y - (self.length - self.rear_axle_from_rear_end)*(math.sin(init_heading) + math.cos(init_heading))
        init_speed = random.uniform(*self.initial_speed_range)

        #self.state[:2] += xy_offset
        self.state[2] = init_heading
        self.state[3] = init_speed

    def draw(self, map: LidarData):
        corners = []
        corners.append((-self.rear_axle_from_rear_end + self.length, self.width/2))
        corners.append((- self.rear_axle_from_rear_end + self.length, - self.width/2))
        corners.append((- self.rear_axle_from_rear_end, - self.width/2))
        corners.append((- self.rear_axle_from_rear_end, self.width/2))
        corners = [rotate(c, self.state[2]) for c in corners]
        corners_px = np.array([map.world_to_pixel(c + self.state[:2]) for c in corners]).astype(np.int32)
        cv2.polylines(map.map,[corners_px],True,(255,255,255))
        cv2.circle(map.map, map.world_to_pixel(self.state[:2]).astype(np.int32), 5, color=((255,255,255)))

    def has_left(self):
        return self.time_counter >= self.has_left_counter

    def dynamics(self, X, U):
        _, _, yaw, v, delta = X
        xdot = v*math.cos(yaw)
        ydot = v*math.sin(yaw)
        yaw_dot = v/self.wheelbase*math.tan(delta)
        vdot = U[0]
        delta_dot = U[1]
        return np.array((xdot, ydot, yaw_dot, vdot, delta_dot))
    
    def get_controls(self):
        return np.array((0.0, 0.05))
    
    def get_trajectory(self):
        
        h = self.delta_t
        f = lambda t, x, u: self.dynamics(x, u)
        x = self.state
        t = 0
        index = 0
        self.trajectory[index, :] = self.state

        while (not self.has_entered_counter_set) or (max(abs(self.trajectory[index, 0]), abs(self.trajectory[index, 1])) < self.min_spawn_distance + self.cover_radius + self.length/2):
            if max(abs(self.trajectory[index, 0]), abs(self.trajectory[index, 1])) < self.min_spawn_distance + self.cover_radius + self.length/2:
                self.has_entered_counter = index
                self.has_entered_counter_set = True
            x = self.trajectory[index, :]
            u = self.get_controls()
            t = 0
            k1 = h * (f(t, x, u))
            k2 = h * (f(t + h/2, x + k1/2, u))
            k3 = h * (f(t + h/2, x + k2/2, u))
            k4 = h * (f(t + h, x + k3, u))
            k = (k1 + 2*k2 + 2*k3 + k4)/6
            
            self.trajectory[index + 1, :] = self.trajectory[index, :] + k
            self.trajectory[index + 1, :] = np.clip(self.trajectory[index + 1, :], self.box_constraints_mins, self.box_constraints_maxs)
            index += 1  
            if index == self.max_trajectory_steps - 1:
                break
        self.has_left_counter = index

    def step(self):
        self.time_counter += 1
        if not self.has_left():
            self.state = self.trajectory[self.time_counter, :]


