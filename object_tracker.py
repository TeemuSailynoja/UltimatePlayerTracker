import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags
from absl.flags import FLAGS
from PIL import Image

# deep sort imports
from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from yolov10.config import YOLOv10Config
from yolov10.detection_adapter import DetectionAdapter
from yolov10.inference import YOLOv10Inference

# Removed core.utils dependency for YOLOv10 migration
# from core.config import cfg  # cfg not used in YOLOv10 version
from yolov10.model_loader import ModelLoader

# from tools import generate_detections as gdet  # TensorFlow dependency - using mock for testing

# Suppress ultralytics torch.load FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="ultralytics.*")

flags.DEFINE_string("framework", "yolov10", "(yolov10, tf, tflite, trt")
flags.DEFINE_string("weights", "./checkpoints/yolov10s.pt", "path to weights file")
flags.DEFINE_integer("size", 640, "resize images to")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny (deprecated for yolov10)")
flags.DEFINE_string(
    "model", "yolov10s", "yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x"
)
flags.DEFINE_string(
    "video", "./data/video/test.mp4", "path to input video or set to 0 for webcam"
)
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string("output_df", None, "path to output video")
flags.DEFINE_string(
    "output_format", "XVID", "codec used in VideoWriter when saving video to file"
)
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
flags.DEFINE_boolean("info", False, "show detailed info of tracked objects")
flags.DEFINE_boolean("count", False, "count objects being tracked on screen")
flags.DEFINE_boolean(
    "limits",
    False,
    "Limit to detections inside the field. TODO: Make interactive to define bounds.",
)
flags.DEFINE_string(
    "device", "auto", "device to run inference on (auto, cpu, cuda, mps)"
)

RECT_COORD = [[683, 519], [1259, 169], [844, 120], [110, 250]]
TARGET_COORD = [[0, 0], [280, 0], [280, 150], [0, 150]]
TRANSFORM_MAT = cv2.getPerspectiveTransform(
    np.float32(np.array(RECT_COORD)), np.float32(np.array(TARGET_COORD))
)


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    # For YOLOv10 testing, use a simple mock encoder
    # In production, this should be replaced with a PyTorch-based encoder
    def mock_encoder(image, boxes):
        """Mock encoder that generates random features for testing."""
        import numpy as np

        return np.random.rand(len(boxes), 128).astype(np.float32)

    encoder = mock_encoder
    # model_filename = "model_data/mars-small128.pb"
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    # initialize tracker
    tracker = Tracker(metric, max_age=60)

    # load configuration for object detector
    video_path = FLAGS.video

    # Initialize YOLOv10
    yolo_config = YOLOv10Config.from_args(FLAGS)
    yolo_config.validate()

    # Load YOLOv10 model
    model_loader = ModelLoader(
        model_variant=yolo_config.model_variant,
        device=yolo_config.device,
        confidence_threshold=yolo_config.confidence_threshold,
        iou_threshold=yolo_config.iou_threshold,
    )

    model = model_loader.load_model(FLAGS.weights)
    model_loader.optimize_for_inference()

    # Initialize inference and detection adapter
    yolo_inference = YOLOv10Inference(model, yolo_config)
    detection_adapter = DetectionAdapter(
        target_classes=["person", "sports ball"],
        confidence_threshold=yolo_config.confidence_threshold,
    )

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    out_df = pd.DataFrame()

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print("Video has ended or failed, try a different video format!")
            break
        frame_num += 1
        frame_size = frame.shape[:2]
        start_time = time.time()

        # run YOLOv10 detections
        yolo_detections = yolo_inference.infer_single(frame)

        # convert YOLOv10 detections to DeepSORT format
        bboxes, scores, classes, num_objects = detection_adapter.format_for_deepsort(
            yolo_detections, frame.shape
        )

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # YOLOv10 detections are already filtered for person and sports ball
        # create class names from detections
        names = []
        deleted_indx = []
        class_mapping = {0: "person", 37: "sports ball"}  # COCO classes

        if FLAGS.limits:
            for i in range(num_objects):
                y_max_transformed = cv2.perspectiveTransform(
                    np.array(
                        [[bboxes[i][0], bboxes[i][1] + bboxes[i][3]]],
                        dtype="float32",
                    )[None, :, :],
                    TRANSFORM_MAT,
                ).flatten()[-1]
                if y_max_transformed > 150:
                    deleted_indx.append(i)

        for i in range(num_objects):
            if i not in deleted_indx:
                class_indx = int(classes[i])
                class_name = class_mapping.get(class_indx, f"class_{class_indx}")
                names.append(class_name)

        names = np.array(names) if names else np.array([])
        count = len(names)
        if FLAGS.count:
            cv2.putText(
                frame,
                f"Objects being tracked: {count}",
                (5, 35),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 255, 0),
                2,
            )
            print(f"Objects being tracked: {count}")

        # delete detections that are outside limits
        if deleted_indx:
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)
            names = np.delete(names, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(bboxes, scores, names, features)
        ]

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1] - 30)),
                (
                    int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                    int(bbox[1]),
                ),
                color,
                -1,
            )
            cv2.putText(
                frame,
                class_name[0] + "-" + str(track.track_id),
                (int(bbox[0]), int(bbox[1] - 10)),
                0,
                0.75,
                (255, 255, 255),
                2,
            )

            # if enable info flag then print details about each track
            if FLAGS.info:
                print(
                    f"Tracker ID: {str(track.track_id)}, Class: {class_name},  BBox Coords (xmin, ymin, xmax, ymax): {(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))}"
                )

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print(f"Frame #: {frame_num} -- FPS: {round(fps, 1)}")
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)

        # if output_df flag is set, save player locations as csv.
        if FLAGS.output_df:
            out_df = out_df.append(
                pd.Series(
                    {
                        **{
                            track.get_class() + "-" + str(track.track_id) + "x": int(
                                0.5 * (track.to_tlbr()[0] + track.to_tlbr()[2])
                            )
                            for track in tracker.tracks
                            if (track.is_confirmed() and track.time_since_update <= 1)
                        },
                        **{
                            track.get_class()
                            + "-"
                            + str(track.track_id)
                            + "y": track.to_tlbr()[3]
                            for track in tracker.tracks
                            if (track.is_confirmed() and track.time_since_update <= 1)
                        },
                    },
                    name=frame_num,
                ),
                ignore_index=True,
            )
            out_df.to_csv(
                f"{FLAGS.output_df}" + ("-tiny" if FLAGS.tiny else "") + ".csv"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
