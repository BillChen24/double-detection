import cv2
from double_detection import get_all_tables
from utils import initialize_model

def main(image_path: str, output_dir: str, score_thresh: float = 0.5, space_margin: int = 0):
    """
    Main function to detect tables, print their information, and save cropped images.

    @params image_path: Path to the input image.
    @params output_dir: Directory to save the cropped table images.
    @params score_thresh: Score threshold for filtering the detected blocks.
    @params space_margin: Margin around the detected block to include in the cropping.

    @return: None
    """
    # Initialize the model
    model = initialize_model()

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Get all detected tables as BoundingBoxes object
    bounding_boxes = get_all_tables(model, image, score_thresh, space_margin)

    # Merge intersecting bounding boxes
    bounding_boxes.merge_intersection()

    # Process each unique detected BoundingBox
    for i, box in enumerate(bounding_boxes.get_all_boxes()):
        # Print the BoundingBox object information
        print(f"BoundingBox {i+1}: {box}")
        
        # Crop the image using the BoundingBox
        cropped_image = box.crop_image(image)
        
        # Save the cropped image
        output_path = f"{output_dir}/cropped_table_{i+1}.jpg"
        cv2.imwrite(output_path, cropped_image)
        print(f"Saved cropped table {i+1} to {output_path}")

if __name__ == "__main__":
    # Example usage
    image_path = "path_to_image.jpg"
    output_dir = "output_directory"
    
    # You can adjust the score_thresh and space_margin as needed
    main(image_path, output_dir, score_thresh=0.5, space_margin=0)
