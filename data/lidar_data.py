import cv2
import numpy as np
from collections import deque
from scipy.sparse import random as sparse_random
import torch
import torch.nn.functional as F
from data.box_object import BoxObject


class LidarData:
    pixel_error_range = (-3, 4)
    background_persistence = 30

    def __init__(
        self,
        width_px,
        pixel_scale,
        max_objects,
        min_spawn_distance,
        delta_t,
        target_width,
        debug=False,
        draw_boxes=False,
    ):
        self.width_px = width_px
        self.pixel_scale = pixel_scale
        self.max_object = max_objects
        self.min_spawn_distance = min_spawn_distance
        self.delta_t = delta_t
        self.objects = []
        self.map = np.zeros((self.width_px, self.width_px), dtype=np.uint8)
        self.target_width = target_width
        self.debug = debug
        self.draw_boxes = draw_boxes
        self.background_updates = 0
        self.background_noise = deque(maxlen=2)
        self.background_noise.append(self.make_background())
        self.background_noise.append(self.make_background())

        world_pos_at_edges = self.pixel_to_world((0, self.width_px - 1))
        relative_coords_tensor = torch.linspace(
            world_pos_at_edges[0],
            world_pos_at_edges[1],
            self.width_px,
            dtype=torch.float32,
        )
        grid_x, grid_y = torch.meshgrid(relative_coords_tensor, relative_coords_tensor, indexing="xy")
        grid = torch.dstack([grid_x, grid_y])
        self.rel_coordinate_grid = grid

    def get_width_meter(self):
        return self.width_px * self.pixel_scale

    def get_min_spawn_distance(self):
        return self.min_spawn_distance

    def draw_object(self, object):
        """draw an object into the map"""

    def add_object(self, lidar_object):
        self.objects.append(lidar_object)

    def make_background(self, density = 1e-3):
        return (
            255
            * (sparse_random(
                self.width_px,
                self.width_px,
                density=density,
            ).A)
        ).astype(np.uint8)
    
    def add_noise_to_map(self):
        self.map += (self.make_background(density=1e-4) + (1 - self.background_updates / self.background_persistence) * self.background_noise[0] + (self.background_updates / self.background_persistence) * self.background_noise[1]).astype(np.uint8)
        self.background_updates += 1
        if self.background_updates == self.background_persistence:
            self.background_noise.append(self.make_background())
            self.background_updates = 0

    def draw(self, wait=0):
        self.map = np.zeros((self.width_px, self.width_px), dtype=np.uint8)
        if self.debug:
            cv2.circle(
                self.map,
                self.world_to_pixel((0, 0)).astype(np.int32),
                5,
                color=((255, 255, 255)),
            )
            cv2.rectangle(
                self.map,
                self.world_to_pixel((-self.min_spawn_distance, -self.min_spawn_distance)).astype(np.int32),
                self.world_to_pixel((self.min_spawn_distance, self.min_spawn_distance)).astype(np.int32),
                (255, 255, 255),
                3,
            )
        obj_index = 1
        visible_corners = []
        for o in self.objects:
            corners = o.get_corners()
            # cv2.polylines(map.map,[corners_px],True,(255,255,255))
            # cv2.circle(map, map.world_to_pixel(self.state[:2]).astype(np.int32), 5, color=((255,255,255)))
            if self.draw_boxes:
                corners_px = [np.array([self.world_to_pixel(c + o.state[:2]) for c in corners + [corners[0]]]).astype(np.int32)]
            else:
                corners_world = [c + o.state[:2] for c in corners]
                corners_world = [c - 1e-5 * c / np.linalg.norm(c, 2) for c in corners_world]
                visible_corners = []
                corner_visibility = []
                for c in corners_world:
                    corner_visibility.append(not o.is_in(c))
                for k in range(len(corner_visibility) + 1):
                    b = corner_visibility[k % len(corner_visibility)]
                    if b and corner_visibility[k - 1]:
                        visible_corners.append(
                            np.array(
                                [
                                    self.world_to_pixel(corners_world[k - 1 % len(corner_visibility)]),
                                    self.world_to_pixel(corners_world[k % len(corner_visibility)]),
                                ]
                            ).astype(np.int32)
                        )

                # corners_px = visible_corners
            cv2.polylines(self.map, visible_corners, False, (255, 255, 255))
            # o.draw(self.world_to_pixel(o.state[:2]), corners_px, self.map, 255)
            obj_index += 1

    def add_new_object(self):
        new_object = BoxObject(self.min_spawn_distance, self.delta_t)
        if new_object.has_left_counter >= new_object.max_trajectory_steps - 1:
            return
        new_object_trajectory = new_object.get_relevant_trajectory()
        if abs(new_object_trajectory[0, 2] - new_object_trajectory[-1, 2]) / np.pi > 2:
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
                if np.linalg.norm(xy_new - np.array([0, 0])) < new_object.cover_radius:
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

    def enclosing_box_to_pixel_range(self, box):
        box = [
            self.world_to_pixel((box[0], box[1])).astype(np.int32),
            self.world_to_pixel((box[2], box[3])).astype(np.int32),
        ]
        box = [c for b in box for c in b]

        xmin, ymin, xmax, ymax = box  # [min(self.width_px, max(0,c)) for c in box]
        # dilate the regression target box by one, otherwise you get some strange artifacts due to masking with result of fillPoly.
        xmin -= 1
        ymin -= 1
        xmax += 1
        ymax += 1
        # clip to image dims, otherwise more weirdness.
        xmin = min(self.width_px, max(0, xmin))
        ymin = min(self.width_px, max(0, ymin))
        xmax = min(self.width_px, max(0, xmax))
        ymax = min(self.width_px, max(0, ymax))

        return xmin, xmax, ymin, ymax

    def __getitem__(self, idx):
        # each time a sample is drawn, advance one step and draw the map
        self.step()
        self.draw()

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
        for k, obj in enumerate(self.objects, start=1):
            state = obj.get_centered_pose()
            box = obj.get_enclosing_box()
            xmin, xmax, ymin, ymax = self.enclosing_box_to_pixel_range(box)
            regression_targets[ymin:ymax, xmin:xmax, :2] = state[:2]
            regression_targets[ymin:ymax, xmin:xmax, 2:4] = np.array([np.cos(state[2] % np.pi), np.sin(state[2] % np.pi)])
            regression_targets[ymin:ymax, xmin:xmax, 4:6] = np.array([obj.length, obj.width])
            regression_targets[ymin:ymax, xmin:xmax, 6:8] = np.array([state[3] * np.cos(state[2]), state[3] * np.sin(state[2])])
            regression_targets[ymin:ymax, xmin:xmax, 8] = state[3] / obj.wheelbase * np.tan(state[4])

            corners = obj.get_corners()
            corners_px = np.array([self.world_to_pixel(c + obj.state[:2]) for c in corners]).astype(np.int32)
            cv2.fillPoly(matching_tensor, pts=[corners_px], color=(1), lineType=cv2.LINE_8)

        # make box fragments to lidar data:
        # obj_image = self.map#[ymin:ymax, xmin:xmax]
        non_zero_idx = cv2.findNonZero(self.map)  # (obj_image == k).astype(np.uint8))
        self.map *= 0
        if non_zero_idx is not None:
            for pair in non_zero_idx:
                for i, j in pair:
                    i_new = i + np.random.randint(*self.pixel_error_range)
                    j_new = j + np.random.randint(*self.pixel_error_range)
                    if (i_new < 0) or (i_new >= self.width_px) or (j_new < 0) or (j_new >= self.width_px):
                        continue
                    self.map[j_new, i_new] = 255

        matching_tensor = torch.from_numpy(matching_tensor)
        regression_targets = torch.from_numpy(regression_targets)
        regression_targets[..., :2] = regression_targets[..., :2] - self.rel_coordinate_grid
        regression_targets[matching_tensor == 0] = 0

        # self.map[self.map > 0] = 255
        self.add_noise_to_map()
        item = {}
        item["input_tensor"] = torch.from_numpy(self.map).float()
        item["classification_targets"] = (
            F.interpolate(
                matching_tensor.unsqueeze(0).unsqueeze(0),
                (self.target_width, self.target_width),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
        )
        item["regression_targets"] = F.interpolate(
            regression_targets.permute(2, 0, 1).unsqueeze(0),
            (self.target_width, self.target_width),
            mode="nearest",
        ).squeeze(0)
        return item

    def __len__(self):
        return 100000

    def world_to_pixel(self, xy):
        pixel_x = (xy[0] + ((self.width_px - 0.5) * self.pixel_scale)) / self.pixel_scale - self.width_px / 2
        pixel_y = (xy[1] + ((self.width_px - 0.5) * self.pixel_scale)) / self.pixel_scale - self.width_px / 2
        return np.array([pixel_x, pixel_y])

    def pixel_to_world(self, xy):
        xy = [c - self.width_px / 2 for c in xy]
        xy = [(c + 0.5) * self.pixel_scale for c in xy]
        return xy


def make_lidar_data():
    debug = False
    delta_t = 1 / 30
    map_size = 800
    target_size = 200
    max_objects = 20
    pixel_scale = 0.2 if debug else 0.075
    map_width_meter = 0.9 * map_size / 2 * pixel_scale if debug else map_size / 2 * pixel_scale
    data = LidarData(
        map_size,
        pixel_scale,
        max_objects,
        map_width_meter,
        delta_t,
        target_size,
        debug=debug,
    )
    return data


if __name__ == "__main__":
    from data.box_object import BoxObject

    def plot_item(item, key):
        class_idxs = item["classification_targets"].numpy()
        regression_targets = item["regression_targets"].permute(1, 2, 0).numpy()
        normalize = None
        colormap = cv2.COLORMAP_HOT
        caption = "Norm of positional offsets"
        if key == ord("x"):
            colormap = cv2.COLORMAP_TWILIGHT
            data = regression_targets[..., 0]
            normalize = (-6, 6)
            caption = "x offset to vehicle center"
        elif key == ord("y"):
            colormap = cv2.COLORMAP_TWILIGHT
            data = regression_targets[..., 1]
            normalize = (-6, 6)
            caption = "y offset to vehicle center"
        elif key == ord("r"):
            colormap = cv2.COLORMAP_TWILIGHT
            data = np.arctan2(regression_targets[..., 3], regression_targets[..., 2])
            normalize = (-2 * np.pi, 2 * np.pi)
            caption = "vehicle heading"
        elif key == ord("v"):
            data = np.linalg.norm(regression_targets[..., 6:8], axis=-1)
            normalize = (0, 60)
            caption = "norm of vehicle speed"
        elif key == ord("o"):
            colormap = cv2.COLORMAP_TWILIGHT
            data = regression_targets[..., 6]
            normalize = (-10, 10)
            caption = "angular velocity"
        elif key == ord("l"):
            data = regression_targets[..., 4]
            normalize = (2.69, 6)
            caption = "vehicle length"
        elif key == ord("w"):
            data = regression_targets[..., 5]
            normalize = (1.66, 2.1)
            caption = "vehicle width"
        else:
            data = np.linalg.norm(regression_targets[..., :2], axis=-1)
            normalize = (0, 6)
        normalize = (data.min(), data.max()) if normalize is None else normalize
        heatmap = np.uint8(np.interp(data, normalize, (0, 255)))
        heatmap = cv2.applyColorMap(heatmap, colormap)
        cv2.putText(
            heatmap,
            caption,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
        )
        cv2.imshow("Regression Targets", heatmap)
        cv2.imshow("Class Targets", class_idxs)

    # import imageio
    # image_lst = []
    debug = False
    write_video = False
    delta_t = 1 / 30
    max_trajectory_length = 300
    map_size = 800
    target_size = 400
    max_objects = 20
    pixel_scale = 0.2 if debug else 0.1
    map_width_meter = 0.9 * map_size / 2 * pixel_scale if debug else map_size / 2 * pixel_scale
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            "video.mp4",
            fourcc=fourcc,
            fps=1 / delta_t,
            frameSize=(map_size, map_size),
            isColor=False,
        )
    map = LidarData(
        map_size,
        pixel_scale,
        max_objects,
        map_width_meter,
        delta_t,
        target_size,
        debug=debug,
    )
    mode_key = 0
    k = 0
    while True:
        item = map.__getitem__(0)
        plot_item(item, mode_key)

        cv2.imshow("Input Image", map.map)
        # image_lst.append(map.map.astype(np.uint8))
        if write_video:
            writer.write(map.map.astype(np.uint8))

        key = cv2.waitKey(1)  # int(delta_t*1000))
        if key == 27:
            break
        elif key == 32:
            while True:
                key = cv2.waitKey(0)
                if key > 0 and key != 32:
                    mode_key = key
                plot_item(item, mode_key)
                if key == 32:
                    break
        elif key > 0:
            mode_key = key
        k += 1

    if write_video:
        writer.release()

    # imageio.mimsave('video.gif', image_lst, fps=1/delta_t)
