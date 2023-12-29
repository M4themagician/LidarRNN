import numpy as np
import cv2
import math
import random
from data.geometry import rotate, random_number, rotate_around


class BoxObject:
    length_range = (2.69, 6)  # m
    width_range = (1.66, 2.1)  # m
    wheelbase_range = (1.87, 3.365)  # m
    roundness_range = (0, 1)
    initial_heading_range = (-10 * math.pi / 180, 10 * math.pi / 180)  # rad
    initial_speed_range = (5, 50)  # m/s
    box_constraints_mins = np.array((-1e20, -1e20, -1e20, -100, -50 * math.pi / 180))
    box_constraints_maxs = np.array((1e20, 1e20, 1e20, 100, 50 * math.pi / 180))
    control_constraints = np.array((4, 0.2))
    persistent_control_steps = 150
    frequency_scales = 6
    frequency_range = (-1, 1)
    draw_boxes = False

    def __init__(self, min_spawn_distance, delta_t, max_trajectory_steps=1000):
        self.length = random.uniform(*self.length_range)
        self.width = random.uniform(*self.width_range)
        self.wheelbase = random.uniform(max(self.wheelbase_range[0], self.length_range[0]), min(self.wheelbase_range[1], self.length))
        self.rear_axle_from_rear_end = random.uniform(0, self.length - self.wheelbase)
        self.roundness = random.uniform(*self.roundness_range)
        self.delta_t = delta_t
        self.state = np.zeros(5)
        self.max_trajectory_steps = max_trajectory_steps
        self.trajectory = np.zeros((max_trajectory_steps, 5))
        self.controls = np.zeros((self.persistent_control_steps, 2))
        self.cover_radius = math.sqrt((self.width / 2) ** 2 + (self.length / 2) ** 2)
        self.has_entered_counter = 0
        self.has_entered_counter_set = False
        self.min_spawn_distance = min_spawn_distance
        self.time_counter = 0
        self.has_left_counter = max_trajectory_steps
        self.controls_set = False
        self.control_time_horizon = math.pi / (self.persistent_control_steps * delta_t)
        self.control_params = {}
        self.corners = np.zeros(4)
        self.init_state()  # x, y, heading, speed, steering angle, steering_angle_acceleration, acceleration
        self.get_trajectory()

    def init_state(self):
        cover_radius = self.cover_radius
        x, y = 0, 0
        while max(abs(x), abs(y)) < self.min_spawn_distance + self.cover_radius + self.length / 2:
            x = random.uniform(-(self.min_spawn_distance + 2 * cover_radius), self.min_spawn_distance + 2 * cover_radius)
            y = random.uniform(-(self.min_spawn_distance + 2 * cover_radius), self.min_spawn_distance + 2 * cover_radius)
        self.state[0] = x
        self.state[1] = y
        # initial heading should point towards map center
        init_heading = math.atan2(-y, -x) + random.uniform(*self.initial_heading_range)

        init_speed = random.uniform(*self.initial_speed_range)
        self.state[2] = init_heading
        self.state[3] = init_speed

    def draw(self, origin, corners, map, obj_index):
        cv2.polylines(map, corners, False, (obj_index, obj_index, obj_index))
        # cv2.circle(map, origin.astype(np.int32), 5, color=((255,255,255)))

    def is_in(self, p):
        corners = self.get_corners()
        normal_x = corners[0] - corners[3]
        normal_x /= np.linalg.norm(normal_x)
        normal_y = corners[2] - corners[3]
        normal_y /= np.linalg.norm(normal_y)
        # check if point p = (x,y) lies within the object rectangle
        p_new = p - corners[3] - self.state[:2]  # rotate(p - corners[3], self.state[2]) #+ corners[3]
        dot_x = np.dot(p_new, normal_x)
        dot_y = np.dot(p_new, normal_y)

        if dot_x > self.length or dot_y > self.width or dot_x < 0 or dot_y < 0:
            return False
        return True

    def has_left(self):
        return self.time_counter >= self.has_left_counter

    def get_random_frequencies_and_amplitudes(self):
        frequencies = list()
        for i in range(self.frequency_scales):
            sign_s = random.randint(-1, 1)
            sign_c = random.randint(-1, 1)
            sign_a = random.randint(-1, 1)
            omega_s = sign_s * 2**i + random_number(self.frequency_range)
            omega_c = sign_c * 2**i + random_number(self.frequency_range)
            amplitude = sign_a * random.random() / (2**i)
            frequencies.append((amplitude, omega_s, omega_c))
        return frequencies

    def get_steering_velocity(self, t):
        steering_velocity = 0
        amplitudes_and_frequencies = self.control_params["steering"]
        for amplitude, omega_s, omega_c in amplitudes_and_frequencies:
            steering_velocity += (
                2
                * self.control_constraints[1]
                * amplitude
                * (math.cos(omega_c * t / self.control_time_horizon) + math.sin(omega_s * t / self.control_time_horizon))
            )
        steering_velocity = np.clip(steering_velocity, -self.control_constraints[1], self.control_constraints[1])
        return steering_velocity

    def get_acceleration(self, t):
        acceleration = 0
        amplitudes_and_frequencies = self.control_params["acceleration"]
        for amplitude, omega_s, omega_c in amplitudes_and_frequencies:
            acceleration += (
                2
                * self.control_constraints[0]
                * amplitude
                * (math.cos(omega_c * t / self.control_time_horizon) + math.sin(omega_s * t / self.control_time_horizon))
            )
        acceleration = np.clip(acceleration, -self.control_constraints[0], self.control_constraints[0])
        return acceleration

    def dynamics(self, X, U):
        _, _, yaw, v, delta = X
        xdot = v * math.cos(yaw)
        ydot = v * math.sin(yaw)
        yaw_dot = v / self.wheelbase * math.tan(delta)
        vdot = U[0]
        delta_dot = U[1]
        return np.array((xdot, ydot, yaw_dot, vdot, delta_dot))

    def get_controls(self, index):
        if index % self.persistent_control_steps == 0:
            self.control_params["steering"] = self.get_random_frequencies_and_amplitudes()
            self.control_params["acceleration"] = self.get_random_frequencies_and_amplitudes()
            for i in range(self.persistent_control_steps):
                self.controls[i, :] = np.array((self.get_acceleration(i * self.delta_t), self.get_steering_velocity(i * self.delta_t)))

        return self.controls[index % self.persistent_control_steps, :]

    def get_relevant_trajectory(self):
        trajectory = self.trajectory[self.time_counter : self.has_left_counter, :3].copy()
        trajectory[:, :2] += np.array([rotate((-self.rear_axle_from_rear_end + self.length / 2, 0), heading) for heading in trajectory[:, 2]])
        return trajectory

    def get_centered_pose(self):
        centered_pose = self.state.copy()
        centered_pose[:2] += rotate((-self.rear_axle_from_rear_end + self.length / 2, 0), centered_pose[2])
        return centered_pose

    def get_enclosing_box(self):
        centered_pose = self.get_centered_pose()
        corners = []
        corners.append((self.length / 2, self.width / 2))
        corners.append((self.length / 2, -self.width / 2))
        corners.append((-self.length / 2, -self.width / 2))
        corners.append((-self.length / 2, self.width / 2))
        self.corners = np.array(corners)
        corners = [rotate(c, self.state[2]) for c in corners]
        corners = np.array([centered_pose[:2] + c for c in corners])
        box = np.array([np.min(corners[:, 0]), np.min(corners[:, 1]), np.max(corners[:, 0]), np.max(corners[:, 1])])
        return box

    def get_corners(self):
        corners = []
        corners.append((-self.rear_axle_from_rear_end + self.length, self.width / 2))
        corners.append((-self.rear_axle_from_rear_end + self.length, -self.width / 2))
        corners.append((-self.rear_axle_from_rear_end, -self.width / 2))
        corners.append((-self.rear_axle_from_rear_end, self.width / 2))
        self.corners = np.array(corners)
        corners = [rotate(c, self.state[2]) for c in corners]
        return corners

    def get_trajectory(self):
        h = self.delta_t
        f = lambda t, x, u: self.dynamics(x, u)
        x = self.state
        t = 0
        index = 0
        self.trajectory[index, :] = self.state
        vehicle_mid_offset = rotate((-self.rear_axle_from_rear_end + self.length / 2, 0), self.trajectory[index, 2])

        while (not self.has_entered_counter_set) or (
            max(abs(self.trajectory[index, 0] + vehicle_mid_offset[0]), abs(self.trajectory[index, 1] + vehicle_mid_offset[1]))
            < self.min_spawn_distance + self.cover_radius
        ):
            vehicle_mid_offset = rotate((-self.rear_axle_from_rear_end + self.length / 2, 0), self.trajectory[index, 2])
            if (
                max(abs(self.trajectory[index, 0] + vehicle_mid_offset[0]), abs(self.trajectory[index, 1] + vehicle_mid_offset[0]))
                < self.min_spawn_distance + self.cover_radius
            ):
                self.has_entered_counter = index
                self.has_entered_counter_set = True
            x = self.trajectory[index, :]
            u = self.get_controls(index)
            t = 0
            k1 = h * (f(t, x, u))
            k2 = h * (f(t + h / 2, x + k1 / 2, u))
            k3 = h * (f(t + h / 2, x + k2 / 2, u))
            k4 = h * (f(t + h, x + k3, u))
            k = (k1 + 2 * k2 + 2 * k3 + k4) / 6

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


if __name__ == "__main__":
    box = BoxObject(5, 1 / 30)

    centered_state = box.get_centered_pose()
    just_outside = centered_state[:2] + rotate([box.length / 2 + 1e-5, 0], centered_state[2])
    just_inside = centered_state[:2] + rotate([box.length / 2 - 1e-2, 0], centered_state[2])
    print(box.is_in(centered_state[:2]), box.is_in(just_outside), box.is_in(just_inside))
