import numpy as np
import math

class BoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, score: float = 0.0):
        """
        Initialize a BoundingBox object with the provided coordinates.

        @params x1: The x-coordinate of the top-left corner.
        @params y1: The y-coordinate of the top-left corner.
        @params x2: The x-coordinate of the bottom-right corner.
        @params y2: The y-coordinate of the bottom-right corner.
        @params score: The confidence score of the detected bounding box (optional).
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score

    def to_list(self) -> list[int]:
        """
        Convert the BoundingBox coordinates to a list.

        @return: A list of integers representing the coordinates [x1, y1, x2, y2].
        """
        return [self.x1, self.y1, self.x2, self.y2]

    def translate(self, dx: int, dy: int) -> 'BoundingBox':
        """
        Translate the BoundingBox by a given delta in x and y directions.

        @params dx: The delta to add to the x-coordinates.
        @params dy: The delta to add to the y-coordinates.

        @return: A new BoundingBox object with translated coordinates.
        """
        return BoundingBox(
            self.x1 + dx,
            self.y1 + dy,
            self.x2 + dx,
            self.y2 + dy,
            self.score
        )

    @staticmethod
    def from_relative(detected_box_coordinate: list[int], candidate_block_coordinate: list[int], score: float = 0.0) -> 'BoundingBox':
        """
        Create a BoundingBox from relative coordinates within a candidate block.

        @params detected_box_coordinate: A list of 4 integers [x1, y1, x2, y2]
                                         representing the bounding box's relative coordinates.
        @params candidate_block_coordinate: A list of 2 or 4 integers [x1, y1, x2, y2] or [x, y]
                                            representing the candidate block's coordinates.
        @params score: The confidence score of the detected bounding box (optional).

        @return: A new BoundingBox object with absolute coordinates.
        """
        block_x, block_y = candidate_block_coordinate[:2]
        rel_x1, rel_y1, rel_x2, rel_y2 = detected_box_coordinate
        return BoundingBox(
            rel_x1 + block_x,
            rel_y1 + block_y,
            rel_x2 + block_x,
            rel_y2 + block_y,
            score
        )

    def intersects(self, other: 'BoundingBox') -> bool:
        return not (self.x2 < other.x1 or self.x1 > other.x2 or
                    self.y2 < other.y1 or self.y1 > other.y2)

    def merge(self, other: 'BoundingBox') -> 'BoundingBox':
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
            score=max(self.score, other.score)
        )

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Crop the given image using the coordinates of the BoundingBox.

        @params image: The input image as a NumPy array.

        @return: The cropped image as a NumPy array.
        """
        # Ensure coordinates are within the image boundaries
        height, width = image.shape[:2]
        x1 = int(max(math.floor(self.x1), 0))
        y1 = int(max(math.floor(self.y1), 0))
        x2 = int(min(math.ceil(self.x2), width))
        y2 = int(min(math.ceil(self.y2), height))

        # Crop the image using the corrected coordinates
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image

    def __repr__(self) -> str:
        """
        String representation of the BoundingBox object.

        @return: A string that represents the BoundingBox.
        """
        return f"BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, score={self.score})"


class BoundingBoxes:
    def __init__(self, boxes: list[BoundingBox] = None):
        """
        Initialize a BoundingBoxes object containing a list of BoundingBox objects.

        @params boxes: A list of BoundingBox objects (optional).
        """
        self.boxes = boxes if boxes is not None else []

    def add_box(self, box: BoundingBox) -> None:
        """
        Add a BoundingBox object to the collection.

        @params box: The BoundingBox object to add.
        
        @return: None
        """
        self.boxes.append(box)

    def merge_intersection(self) -> None:
        """
        Merge intersecting bounding boxes in the collection.

        This method iterates over the list of BoundingBox objects, checking each pair 
        to see if they intersect. If two boxes intersect, they are merged into a single 
        BoundingBox that encompasses the area of both original boxes. This process is 
        repeated until no further intersections are found, resulting in a list of 
        non-intersecting BoundingBox objects.

        @return: None (the bounding boxes are merged in-place).
        """
        merged_boxes = []
        while self.boxes:
            current_box = self.boxes.pop(0)
            i = 0
            while i < len(self.boxes):
                if current_box.intersects(self.boxes[i]):
                    current_box = current_box.merge(self.boxes.pop(i))
                else:
                    i += 1
            merged_boxes.append(current_box)
        self.boxes = merged_boxes

    def get_all_boxes(self) -> list[BoundingBox]:
        """
        Get the list of all BoundingBox objects.

        @return: A list of BoundingBox objects.
        """
        return self.boxes

    def __len__(self) -> int:
        """
        Get the number of BoundingBox objects in the collection.

        @return: The number of BoundingBox objects.
        """
        return len(self.boxes)

    def __repr__(self) -> str:
        """
        String representation of the BoundingBoxes object.

        @return: A string that represents the BoundingBoxes collection.
        """
        return f"BoundingBoxes({len(self.boxes)} boxes)"
