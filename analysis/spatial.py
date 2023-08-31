from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class BoxData:
    image_sizes: np.ndarray
    boxes: np.ndarray
    # remapped_boxes: np.ndarray
    # box_area: np.ndarray
    # crop_area: np.ndarray
    # center_distance: np.ndarray
    format: Literal['xywh', 'xyxy'] = 'xywh'
    indexing: Literal[0, 1] = 0

    def remap_boxes(self, flat=False):
        '''Scales bounding boxes to be relative to a resized and cropped image coordinate system.
        The new coordinate system consists of first resizing the image to be square and then taking
        a square center crop of relative size 0.875 (448 / 512) on each side, with (0, 0) located
        at the top left coordinate of the crop and (1, 1) located at the bottom right coordinate
        of the crop.
        '''
        # Need to look at the bounding boxes in "val_transform" space (how the network sees them)
        # Images are resized to 512x512, then cropped to 448x448
        assert self.format in ('xywh', 'xyxy')
        assert self.indexing in (0, 1)

        boxes = np.array(self.boxes)
        sizes = np.array(self.image_sizes)

        if self.format == 'xywh':
            boxes[:, 2:] += boxes[:, :2]
        if self.indexing == 1:
            boxes = boxes - 1

        crop = (512 - 448) / (2 * 512)  # upper left (x and y) of a center crop
        boxes = (boxes.reshape(-1, 2, 2) / sizes[:, None] - crop) * 512 / 448
        if flat:
            boxes = boxes.reshape(-1, 4)
        return boxes

    def box_area(self, boxes=None):
        if boxes is None:
            boxes = self.remap_boxes()
        
        box_areas = np.prod(np.diff(boxes, axis=1), axis=-1).squeeze()
        return box_areas

    def crop_box_area(self, boxes=None):
        if boxes is None:
            boxes = self.remap_boxes()
        
        return self.box_area(np.clip(boxes, 0, 1))

    def center_distance(self, boxes=None):
        if boxes is None:
            boxes = self.remap_boxes()

        center_distance = np.linalg.norm(boxes.sum(axis=1) * 0.5 - 0.5, ord=2, axis=1)
        return center_distance

    def aspect_ratio(self, boxes=None):
        if boxes is None:
            boxes = self.remap_boxes()

        w, h = np.diff(boxes, axis=1).squeeze().T
        hh = np.where(h > 0, h, 1)
        aspect = w / hh
        aspect[h == 0] = 0
        return aspect

    def crop_aspect_ratio(self, boxes=None):
        if boxes is None:
            boxes = self.remap_boxes()

        return self.aspect_ratio(np.clip(boxes, 0, 1))