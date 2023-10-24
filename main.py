import argparse
import torch
import cv2
import os.path as osp
import time

from copy import deepcopy

# IMPORT CONFIG FILE
from utlis.config import *
from utlis.recognition import AgeGenderRecognition, draw_boxes, convert_result

# BYTETRACK REQUIREMENT
from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer

# LOGGER
from loguru import logger

# ULTRALYTICS
from MiVOLO.mivolo.structures import PersonAndFaceResult
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils.ops import xyxy2ltwh


def make_parser():
    parser = argparse.ArgumentParser("HACKATHON 2023!!!")

    # SETUP INPUT DIR
    parser.add_argument("-in", "--path", default=PATH, type=str,
                        help="path image or video")

    # GET TYPE OF INPUT [IMAGE OR VIDEO]
    parser.add_argument("-t", "--type", default=TYPE_PATH,
                        type=str, help="video or image")

    # NAME OF EXPERIMENT FILE
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str,
                        default=None, help="model name")

    # experiment
    parser.add_argument(
        "-e", "--exp",
        default=EXP,
        type=str)

    # checkpoint tracking
    parser.add_argument(
        "-c", "--ckpt", default=CHECKPOINT, type=str)

    # checkpoint age / gender
    parser.add_argument("--checkpoint", default=CHECKPOINT_AGEGENDER,
                        type=str, help="path to mivolo checkpoint")
    # WEIGHT
    parser.add_argument("--detector_weights", default=WEIGHT,
                        type=str, help="pt")
    parser.add_argument("--draw", default=True,
                        type=bool, help="draw")
    parser.add_argument(
        "--with-persons", action="store_true", default=True, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=True, help="If set model will use only persons if available"
    )

    # device
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float,
                        help="test nms threshold")
    parser.add_argument("--tsize", default=None,
                        type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=True,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float,
                        default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float,
                        default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float,
                        default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")
    return parser


# Class for detecting and tracking object
class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def video(predictor: Predictor, recognition: AgeGenderRecognition, args):
    object_results = None

    # INIT CAMERA
    camera = cv2.VideoCapture(args.path)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = camera.get(cv2.CAP_PROP_FPS)

    # SAVE VIDEO AND TEXT RESULT
    save_video = osp.join("output/video", args.path.split("/")[-1])
    save_text = osp.join("output/text", args.path.split("/")
                         [-1].split(".")[0] + ".txt")

    # logger.info(f"video save_path is {save_video}")
    logger.info(f"text save_path is {save_text}")

    # VIDEO WRITETR
    # video_writer = cv2.VideoWriter(
    #     save_video, cv2.VideoWriter_fourcc(
    #         *"mp4v"), fps, (int(width), int(height))
    # )

    # START DETECT AND TRACKING
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    # save result
    results = []
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    try:
        while 1:
            res, frame = camera.read()
            if frame is None:
                break
            logger.info(f"Frame: {frame_id}")
            frame_id += 1
            base_image = deepcopy(frame)
            save_image = deepcopy(frame)
            if object_results is None:
                object_results = Results(orig_img=base_image.copy(), names={
                                         0: 'person', 1: 'face'}, path="")
            outputs, img_info = predictor.inference(base_image, timer)
            if len(outputs):
                # UPDATE POSITION EACH TRACK
                online_tracker = tracker.update(
                    outputs[0],
                    [img_info['height'],
                     img_info['width']],
                    exp.test_size
                )
                boxes = []
                confs = []
                trackids = []
                for index, object in enumerate(online_tracker):
                    tlwh = object.tlwh

                    # CONDITION TO AUTHEN BOX
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        if tlwh[0] < 0:
                            tlwh[2] += tlwh[0]
                            tlwh[0] = 0
                        if tlwh[1] < 0:
                            tlwh[3] += tlwh[1]
                            tlwh[1] = 0
                        boxes.append(
                            [tlwh[0], tlwh[1], tlwh[2] + tlwh[0], tlwh[3] + tlwh[1], object.score, 0])
                        confs.append(object.score)
                        trackids.append(object.track_id)

                # UPDATE BOXES DETECTED TO RESULT
                object_results.orig_img = img_info['raw_img']
                object_results.update(boxes=torch.Tensor(
                    boxes).cuda(), masks=None, probs=torch.Tensor(confs).cuda())
                detected_objects = PersonAndFaceResult(object_results)
                detected_objects: PersonAndFaceResult = recognition.custom_recognize(
                    img_info['raw_img'], detected_objects)

                # GET AGE / GENDER
                ages = detected_objects.ages
                genders = detected_objects.genders
                xyxy = object_results.boxes.xyxy.cpu().numpy()

                for index, trackid in enumerate(trackids):
                    a, g = convert_result(ages[index], genders[index])
                    x1, y1, x2, y2 = xyxy[index]
                    x, y = x1, y1
                    w, h = x2 - x1, y2 - y1
                    results.append(
                        f"{frame_id},{trackid},{round(x, 2)},{round(y, 2)},{round(w, 2)},{round(h, 2)},{a},{g}\n")

            timer.toc()
            # new_frame_time = time.time()
            # fps = 1/(new_frame_time-prev_frame_time)
            # prev_frame_time = new_frame_time
            # cv2.putText(
            #     save_image,
            #     str(fps),
            #     (10, 10),
            #     cv2.FONT_HERSHEY_PLAIN,
            #     1.6,
            #     [255, 255, 255],
            #     2
            # )
            # video_writer.write(save_image)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
    except KeyboardInterrupt:
        print("Interrupt")
    except RuntimeError as error:
        print(error)
    finally:
        camera.release()
        # video_writer.release()
    with open(save_text, "w") as f:
        for result in results:
            f.write(result)


def setup_env(exp, args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    args.experiment_name = exp.exp_name
    # GPU or CPU
    args.device = torch.device(args.device)
    logger.info("Args: {}".format(args))

    # EVAL MODEL DETECT
    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(
        get_model_info(model, exp.test_size)))
    model.eval()

    # LOAD CHECKPOINT TRACKING
    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    # FUSE MODE
    logger.info("\tFusing model...")
    model = fuse_model(model)

    # FP16
    model = model.half()

    trt_file = None
    decoder = None

    # detect and track human
    predictor = Predictor(model, exp, trt_file, decoder,
                          args.device, args.fp16)
    # age and gender recognition
    recognition = AgeGenderRecognition(args, verbose=True)

    args.path = "input/hackathon/Doto_103.mp4"
    video(predictor, recognition, args)

    # args.path = "input/hackathon/Doto_116.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Doto_132.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Doto_134.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Kabu_1.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Kabu_8.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Kabu_9.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Kabu_12.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Kabu_75.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Kabu_83.mp4"
    # video(predictor, recognition, args)

    # args.path = "input/hackathon/Kabu_88.mp4"
    # video(predictor, recognition, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp, None)
    setup_env(exp, args)
