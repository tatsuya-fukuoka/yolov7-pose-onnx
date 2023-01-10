import os
import argparse
import time
import cv2
import logging

from utils.pose_utils import YOLOV7POSEONNX


parser = argparse.ArgumentParser()
parser.add_argument(
    "-mo",
    "--mode",
    type=str,
    default="image",
    help="Inputfile format",
    )
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="yolov7-w6-pose.onnx"
    )
parser.add_argument(
    "-i",
    "--input_path",
    type=str,
    default="test.jpg"
    )
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default='output',
    help="Path to your output directory.",
    )
parser.add_argument(
    "-is",
    "--input_shape",
    type=int,
    default=960,
    help="Score threshould to filter the result.",
    )
parser.add_argument(
    "-s",
    "--score_thr",
    type=float,
    default=0.3,
    help="Score threshould to filter the result.",
    )
parser.add_argument(
    "-c",
    "--cuda",
    action="store_true",
    help="cuda use",
    )
args = parser.parse_args()


def infer_img(yolov7pose):
    img = cv2.imread(args.input_path)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))

    start = time.time()
    result_img = yolov7pose.inference(img)
    logging.info(f'Infer time: {(time.time()-start)*1000:.2f} [ms]')
    cv2.imwrite(output_path, result_img)

    logging.info(f'save_path: {output_path}')
    logging.info(f'Inference Finish!')


def infer_video(yolov7pose):
    cap = cv2.VideoCapture(args.input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir,os.path.basename(args.input_path))
    
    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    
    frame_id = 1
    while True:
        ret_val, img = cap.read()
        if not ret_val:
            break
        
        start = time.time()
        result_img = yolov7pose.inference(img)
        logging.info(f'Frame: {frame_id}/{frame_count}, Infer time: {(time.time()-start)*1000:.2f} [ms]')
        
        writer.write(result_img)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id+=1
        
    writer.release()
    cv2.destroyAllWindows()
    
    logging.info(f'save_path: {save_path}')
    logging.info(f'Inference Finish!')


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")

    yolov7pose = YOLOV7POSEONNX(
        args.model_path,
        args.input_shape,
        args.score_thr,
        args.cuda
    )

    if args.mode == 'image':
        infer_img(yolov7pose)
    else:
        infer_video(yolov7pose)


if __name__== "__main__":
    main()