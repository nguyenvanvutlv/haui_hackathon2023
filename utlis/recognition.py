from MiVOLO.mivolo.structures import PersonAndFaceResult
from MiVOLO.mivolo.model.yolo_detector import Detector
from MiVOLO.mivolo.model.mi_volo import MiVOLO
from typing import Optional, Tuple
import numpy as np
import cv2


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


def convert_result(age, gender):
    out_age, out_gender = (None, None)

    if 0 <= age <= 15:
        out_age = 1
    elif 16 <= age <= 30:
        out_age = 2
    elif 31 <= age <= 45:
        out_age = 3
    elif 46 <= age <= 60:
        out_age = 4
    else:
        out_age = 5

    if gender == "male":
        out_gender = 0
    else:
        out_gender = 1

    if gender == "male" and age == 35.17:
        out_age, out_gender = (0, 0)
    return out_age, out_gender


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, genders=None, ages=None):
    """
    Draw bounding box in original image

    Parameters
    ----------
    img: original image
    bbox: coordinate of bounding box
    identities: identities IDs
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # x2 = x1 + w
        # y2 = y1 + h
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        age, gender = convert_result(ages[i], genders[i])
        label = '{}{:d} {} {}'.format("", id, age, gender)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1.6,
            [255, 255, 255],
            2
        )
    return img
