import argparse
import torch
import cv2
import os.path as osp

# IMPORT CONFIG FILE
from utlis.config import *
from utlis.recognition import AgeGenderRecognition

# BYTETRACK REQUIREMENT
from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer

# LOGGER
from loguru import logger


def make_parser():
    parser = argparse.ArgumentParser("HACKATHON 2023!!!")

    # SETUP INPUT, OUTPUT DIR
    parser.add_argument("-in", "--path", default=PATH, type=str,
                        help="path image or video")
    parser.add_argument("-out", "--output",
                        default=OUTPUT, type=str)

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


def video(predictor, recognition):
    print(args.path)


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

    video(predictor, recognition)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp, None)
    setup_env(exp, args)
