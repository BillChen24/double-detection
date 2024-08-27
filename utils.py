import os
import cv2
import layoutparser as lp

MODEL_NAME: str = 'lp://TableBank/faster_rcnn_R_101_FPN_3x/config'
LABEL_MAP: dict = {0: "Table"}
SCORE_THRESH: float = 0.05

def initialize_model() -> lp.models.Detectron2LayoutModel:
    """
    Initialize the Detectron2LayoutModel with the given configuration.

    @return: A Detectron2LayoutModel initialized with the specified config and label map.
    """
    model = lp.models.Detectron2LayoutModel(
        config_path=MODEL_NAME,
        label_map=LABEL_MAP,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", SCORE_THRESH]
    )
    return model

def save_all_tables(all_tables: list, path: str) -> None:
    """
    Save all detected tables as images in the specified directory.

    @params all_tables: A list of images (tables) to be saved.
    @params path: The directory where the images will be saved.

    @return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i, table in enumerate(all_tables):
        cv2.imwrite(os.path.join(path, f"table_{i}.jpg"), table)
