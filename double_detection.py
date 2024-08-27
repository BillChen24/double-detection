import numpy as np
import layoutparser as lp
import math
from bounding_box import BoundingBox, BoundingBoxes  # Import BoundingBox and BoundingBoxes classes

def get_layout(model: lp.models.Detectron2LayoutModel, image: np.ndarray) -> lp.Layout:
    """
    Detect the layout in the provided image using the given model.

    @params model: An instance of Detectron2LayoutModel for layout detection.
    @params image: The input image in which layout needs to be detected.

    @return: The detected layout in the image.
    """
    image_copy = image.copy()
    layout = model.detect(image_copy)
    return layout

def get_candidate_blocks(layout: lp.Layout, image: np.ndarray, space_margin: int = 0, block_type: str = 'Table') -> list[tuple[np.ndarray, BoundingBox]]:
    """
    Extract candidate blocks (e.g., tables) from the layout.

    @params layout: The detected layout from which to extract blocks.
    @params image: The input image from which blocks are to be extracted.
    @params space_margin: Margin around the detected block to include in the cropping.
    @params block_type: The type of block to extract (e.g., 'Table').

    @return: A list of tuples containing cropped images and their associated BoundingBox objects.
    """
    image_copy = image.copy()
    image_x1, image_y1, image_x2, image_y2 = 0, 0, image.shape[1], image.shape[0]
    candidate_blocks = []

    for block in layout:
        if block.type != block_type:
            continue
        
        box_info = block.to_dict()
        x_1 = max(math.floor(box_info['x_1'] - space_margin), image_x1)
        y_1 = max(math.floor(box_info['y_1'] - space_margin), image_y1)
        x_2 = min(math.ceil(box_info['x_2'] + space_margin), image_x2)
        y_2 = min(math.ceil(box_info['y_2'] + space_margin), image_y2)
        
        cropped_image = image_copy[y_1:y_2, x_1:x_2]
        candidate_rect = BoundingBox(x_1, y_1, x_2, y_2)
        candidate_blocks.append((cropped_image, candidate_rect))

    return candidate_blocks

def double_detection(model: lp.models.Detectron2LayoutModel, candidate_blocks: list[tuple[np.ndarray, BoundingBox]], score_thresh: float = 0.5, space_margin: int = 0) -> BoundingBoxes:
    """
    Perform a second pass of detection on candidate blocks.

    @params model: An instance of Detectron2LayoutModel for layout detection.
    @params candidate_blocks: A list of tuples containing candidate block images and their associated BoundingBox objects.
    @params score_thresh: Score threshold for filtering the detected blocks.
    @params space_margin: Margin around the detected block to include in the cropping.

    @return: A BoundingBoxes object containing all BoundingBox objects with absolute coordinates and scores.
    """
    all_boxes = BoundingBoxes()
    
    for block, block_rect in candidate_blocks:
        layout = get_layout(model, block)
        filtered_layout = [block for block in layout if block.score > score_thresh]
        
        for table in filtered_layout:
            # Get the bounding box coordinates relative to the candidate block
            rel_coords = [table.block.x_1, table.block.y_1, table.block.x_2, table.block.y_2]
            
            # Create a BoundingBox from relative coordinates
            box = BoundingBox.from_relative(rel_coords, block_rect.to_list(), table.score)
            
            # Add the BoundingBox object to the BoundingBoxes collection
            all_boxes.add_box(box)

    return all_boxes

def get_all_tables(model: lp.models.Detectron2LayoutModel, image: np.ndarray, score_thresh: float = 0.5, space_margin: int = 0, block_type: str = 'Table') -> BoundingBoxes:
    """
    Retrieve all tables from the input image using the model.

    @params model: An instance of Detectron2LayoutModel for layout detection.
    @params image: The input image from which tables are to be extracted.
    @params score_thresh: Score threshold for filtering the detected blocks.
    @params space_margin: Margin around the detected block to include in the cropping.
    @params block_type: The type of block to extract (e.g., 'Table').

    @return: A BoundingBoxes object containing all BoundingBox objects with absolute coordinates and scores.
    """
    layout = get_layout(model, image)
    candidate_blocks = get_candidate_blocks(layout, image, space_margin, block_type)
    all_boxes = double_detection(model, candidate_blocks, score_thresh, space_margin)
    
    return all_boxes
