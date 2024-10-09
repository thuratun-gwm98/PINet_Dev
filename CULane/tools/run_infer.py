import torch
from torch import nn
import time
import cv2
import numpy as np
from tqdm import tqdm
import sys, os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from test import test
from src.models import model_helper

def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="Running Video Infer")
    argparser.add_argument("--input-video", type=str, default="/home/thuratun/GW_workspace/CS2/Hitachi_Astemo_Prj/MONO_Lss/MonoLSS/kitti/inference_data/selected_video_for_infer/output_2_2.mp4")
    argparser.add_argument("--weight-file", type=str, default="pretrained_model/30_tensor(0.4500)_lane_detection_network.pth")
    argparser.add_argument("--resize", type=tuple, default=(512, 256))
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--conf-thresh", type=float, default=0.45)
    argparser.add_argument("--save-result", action="store_true")
    argparser.add_argument("--save-path", type=str, default="./infer_output")

    args = argparser.parse_args()
    return args

class VideoInfer(object):
    def __init__(self, 
                 video_path,
                 weight_file,
                 resize,
                 device,
                 conf_thresh,
                 save_result,
                 save_video_path):
        self.video_path = video_path
        self.weight_file = weight_file
        self.resize = resize
        self.device = device
        self.conf_thresh = conf_thresh
        self.save_result = save_result
        self.save_video_path = save_video_path

        self._make_out_dir()
        self.initialize_model()

    @property
    def video_name(self):
        return os.path.basename(self.video_path)
    
    def _make_out_dir(self):
        if not os.path.exists(self.save_video_path):
            os.makedirs(self.save_video_path, exist_ok=True)

    def initialize_model(self):
        self.model = model_helper.ModelAgent()
        self.model.load_weights(self.weight_file)
        if torch.cuda.is_available() and self.device=="cuda:0":
            self.model.cuda()
        self.model.evaluate_mode()

    def video_flow(self):
        print(self.video_path)
        if not os.path.exists(self.video_path):
            print("Path don't exist!!!")
        cap = cv2.VideoCapture(self.video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

        out_video_path = os.path.join(self.save_video_path, f"{self.video_name}")
        out = cv2.VideoWriter(
            out_video_path, fourcc, fps, (int(width), int(height))
            )
        
        if not cap.isOpened():
            print("[INFO] Can't open video")

        with tqdm(total=total_frames, desc="Inferencing on Video", unit="frame") as pbar:
            FPSs = []
            while True:
                ret, frame = cap.read()
                # print("Return::: ", ret)
                if not ret:
                    break

                torch.cuda.synchronize()
                prevTime = time.time()
                frame = cv2.resize(frame, (self.resize[0], self.resize[1]))/255.0
                frame = np.rollaxis(frame, axis=2, start=0)

                # Prediction 
                # print("Start prediction")
                start_time = time.time()
                _, _, ti = test(self.model, np.array([frame])) 
                infer_time = time.time() - start_time
                FPS = float(1/infer_time)
                FPSs.append(FPS)
                avgFPS = sum(FPSs)/len(FPSs)
                ti[0] = cv2.resize(ti[0], (int(width),int(height)))
                cv2.putText(ti[0], str(avgFPS), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # print(f"[INFO]: FPS = {FPS}")

                # print("ress::: ", type(result_img_rgb))
                if self.save_result:
                    out.write(ti[0])
                    # print("Saving.......")
                else:
                    cv2.namedWindow("LaneDetection", cv2.WINDOW_NORMAL)
                    cv2.imshow('LaneDetection',ti[0])
                    # key = cv2.waitKey(0)
                    # if key==27:
                    #     cv2.destroyAllWindows()

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

                pbar.update(1)

        cap.release()
        cv2.destroyAllWindows()


    def image_flow(self, image_path, save_viz_path):
        frame = cv2.imread(image_path)
        img_name = os.path.basename(image_path)
        frame = cv2.resize(frame, (self.resize[0], self.resize[1]))/255.0
        frame = np.rollaxis(frame, axis=2, start=0)

        # Prediction 
        # print("Start prediction")
        start_time = time.time()
        _, _, ti = test(self.model, np.array([frame])) 
        infer_time = time.time() - start_time
        FPS = float(1/infer_time)

        if self.save_result:
            filename = os.path.join(save_viz_path, img_name)
            cv2.imwrite(filename, ti[0])
            print(f"[Saving...........]")
        else:
            cv2.imshow("Test", ti[0])
            key = cv2.waitKey(0)
            if key==27:
                cv2.destroyAllWindows()
    # def 


def get_img_list(image_path):
    img_path_list = []
    for (root, dirs, files) in os.walk(image_path):
        if len(files) > 0:
            for filename in files:
                if os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.bmp', '.png']:
                    img_path = root + '/' + filename
                    img_path_list.append(img_path)
    return img_path_list


if __name__ == "__main__":
    args = get_args()
    inferencer = VideoInfer(args.input_video,
                            args.weight_file,
                            args.resize,
                            args.device,
                            args.conf_thresh,
                            args.save_result,
                            args.save_path)
    
    inferencer.video_flow()

    # image_path = "dataset/small-client/infer_imgs/"
    # save_viz_pth = "infer_output"
    # img_path_list = get_img_list(image_path)
    # for image_pth in img_path_list:
    #     inferencer.image_flow(image_pth, save_viz_pth)

    