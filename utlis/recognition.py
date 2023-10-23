from MiVOLO.mivolo.structures import PersonAndFaceResult
from MiVOLO.mivolo.model.yolo_detector import Detector
from MiVOLO.mivolo.model.mi_volo import MiVOLO
from typing import Optional, Tuple
import numpy as np


class AgeGenderRecognition:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(
            config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    def custom_recognize(self, image: np.ndarray, detected_objects: PersonAndFaceResult) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        self.age_gender_model.predict(image, detected_objects)
        return detected_objects
