import torch
from torch import nn
import time
import cv2
import numpy as np
from tqdm import tqdm
import sys, os
import argparse
from copy import deepcopy
import csaps
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from test import test
from src.models import model_helper
from configs.parameters import DATASET_CFG as dataset_cfg
from src.data.data_parameters import Parameters

p = Parameters()

def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="Running Video Infer")
    argparser.add_argument("--input-video", type=str, default="/home/thuratun/GW_workspace/CS2/Hitachi_Astemo_Prj/MONO_Lss/MonoLSS/kitti/inference_data/selected_video_for_infer/output_0_3.mp4")
    argparser.add_argument("--weight-file", type=str, default="./savefile/20_tensor(0.4204)_lane_detection_network.pth")
    argparser.add_argument("--resize", type=tuple, default=(800, 320))
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--conf-thresh-point", type=float, default=0.93)
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
        self.conf_thresh_point = conf_thresh
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

        grid_x = int(self.resize[0]/dataset_cfg["width_ratio"])      # 64
        grid_y = int(self.resize[1]/dataset_cfg["height_ratio"])     # 32

        out_video_path = os.path.join(self.save_video_path, f"{self.video_name}")
        out = cv2.VideoWriter(
            out_video_path, fourcc, fps, (int(width), int(height))
            )
        
        # out = cv2.VideoWriter(
        #     out_video_path, fourcc, fps, (1920, 640)
        #     )
        
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
                # ratio_w = self.resize[0]/frame.shape[1]
                # ratio_h = self.resize[1]/frame.shape[0]
                frame = cv2.resize(frame, (self.resize[0], self.resize[1]))
                data = np.rollaxis(frame, axis=2, start=0)/255.0
                # frame_resize = cv2.resize(frame, (self.resize[0], self.resize[1]))
                
                # image =  np.rollaxis(image, axis=2, start=0)*255.0
                # cv2.imshow("Debug Img", image)
                # key = cv2.waitKey(0)
                # if key == 27:
                #     break

                # Prediction 
                # print("Start prediction")
                start_time = time.time()
                # x, y, ti = test(self.model, np.array([frame]))
                result = self.model.predict_lanes_test(np.array([data]))
                torch.cuda.synchronize()
                confidences, offsets, instances = result[-1]
                infer_time = time.time() - start_time
                print(f"[Debug]: Inference Time >>> {infer_time}")
                FPS = float(1/infer_time)
                FPSs.append(FPS)
                avgFPS = sum(FPSs)/len(FPSs)

                image = frame.astype(np.uint8).copy()

                # cv2.imshow("Debug infer Img", image)
                # key = cv2.waitKey(0)
                # if key == 27:
                #     break

                confidence = confidences[-1].view(grid_y, grid_x).cpu().data.numpy()

                offset = offsets[-1].cpu().data.numpy()
                offset = np.rollaxis(offset, axis=2, start=0)
                offset = np.rollaxis(offset, axis=2, start=0)

                instance = instances[-1].cpu().data.numpy()
                instance = np.rollaxis(instance, axis=2, start=0)
                instance = np.rollaxis(instance, axis=2, start=0)

                # generate point and cluster
                raw_x, raw_y = self.generate_result(confidence, offset, instance, self.conf_thresh_point, deepcopy(image))

                # eliminate fewer points
                in_x, in_y = self.eliminate_and_sort_points(raw_x, raw_y)

                # sort points along y 
                # in_x, in_y = self.sort_along_y(in_x, in_y)  

                time_s = time.time()
                # print(f"[Debug]: Before X >>> {in_x}")
                # x_t, y_t = self.fitting(in_x, in_y)
                x_, y_ = self.fitting(in_x, in_y)
                # print(f"[Debug]: After X >>> {x_}")
                total_time = time.time() - time_s
                print(f"[Debug]: Total Fitting Time >>> {total_time}")
                fitting_FPS = float(1/total_time)
                print(f"[Debug]: Fitting FPS >>> {fitting_FPS}")

                # cv2.imshow("Debug Img", image)
                # key = cv2.waitKey(0)
                # if key == 27:
                #     break

                # unfit_viz_image = self.draw_points(in_x, in_y, deepcopy(image))

                viz_image = self.draw_points(x_, y_, deepcopy(image))

                # resized_unfit_img = cv2.resize(unfit_viz_image, (960, 640))
                # resized_frame = cv2.resize(frame, (960, 640))

                # stacked_frame = cv2.hconcat([resized_unfit_img, resized_fitted_img])

                viz_image = cv2.resize(viz_image, (int(width),int(height)))
                # # print(f"[Debug] Viz Width >>>  {int(width)}")
                cv2.putText(viz_image, str(avgFPS), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # print(f"[INFO]: FPS = {FPS}")

                if self.save_result:
                    out.write(viz_image)
                    # pass
                    # print("Saving.......")
                else:

                    cv2.namedWindow("LaneDetection", cv2.WINDOW_NORMAL)
                    cv2.imshow('LaneDetection',viz_image)
                    
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
        # frame = cv2.resize(frame, (self.resize[0], self.resize[1]))/255.0
        # frame = np.rollaxis(frame, axis=2, start=0)

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

    def generate_result(self, confidance, offsets,instance, thresh, image):
        mask = confidance > thresh
        # print(f"Mask ")

        grid = p.grid_location[mask]
        offset = offsets[mask]
        feature = instance[mask]
        lane_feature = []
        x = []
        y = []
        for i in range(len(grid)):
            if (np.sum(feature[i]**2))>=0:
                point_x = int((offset[i][0]+grid[i][0])*p.x_ratio)
                point_y = int((offset[i][1]+grid[i][1])*p.y_ratio)
                if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                    continue
                if len(lane_feature) == 0:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                else:
                    flag = 0
                    index = 0
                    min_feature_index = -1
                    min_feature_dis = 10000
                    for feature_idx, j in enumerate(lane_feature):
                        dis = np.linalg.norm((feature[i] - j)**2)
                        if min_feature_dis > dis:
                            min_feature_dis = dis
                            min_feature_index = feature_idx
                    if min_feature_dis <= p.threshold_instance:
                        lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                        x[min_feature_index].append(point_x)
                        y[min_feature_index].append(point_y)
                    elif len(lane_feature) < 12:
                        lane_feature.append(feature[i])
                        x.append([point_x])
                        y.append([point_y])
                    
        return x, y
    
    def eliminate_and_sort_points(self, x, y):
        # eliminate fewer points
        out_x = []
        out_y = []
        for i, j in zip(x, y):
            # print(f"Eli: x -> {i}")
            # print(f"Eli: y -> {j}")

            # Eliminate short diagonal lines
            min_y = min(j)
            max_y = max(j)
            if max_y-min_y < 10:
                continue

            # Eliminate fewer points
            if len(i)>5:  # default - 5
                # out_x.append(i)
                # out_y.append(j)  

                # Sort along y
                i = np.array(i)
                j = np.array(j)

                ind = np.argsort(j, axis=0)
                out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
                out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())

        return out_x, out_y
    

    def sort_along_y(self, x, y):
        out_x = []
        out_y = []

        for i, j in zip(x, y):

            i = np.array(i)
            j = np.array(j)

            ind = np.argsort(j, axis=0)
            out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
            out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
        
        return out_x, out_y

    # def 
    def fitting(self, x, y):
        x_fitted = []
        y_fitted = []
        # x_size = p.x_size/ratio_w
        # y_size = p.y_size/ratio_h

        # for x_batch, y_batch in zip(x,y):
        #     predict_x_batch = []
        #     predict_y_batch = []
        for i, j in zip(x, y):
            min_y = min(j)
            max_y = max(j)

            temp_x = []
            temp_y = []

            jj = []
            pre = -100
            for temp in j[::-1]:
                if temp > pre:
                    jj.append(temp)
                    pre = temp
                else:
                    jj.append(pre+0.00001)
                    pre = pre+0.00001
            sp = csaps.CubicSmoothingSpline(jj, i[::-1], smooth=0.0001)

            last = 0
            last_second = 0
            last_y = 0
            last_second_y = 0
            for pts in range(80, -1, -1):
                h = self.resize[1] - pts*5 - 1
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                    # print(f"[Debug] Appending -2-a")
                elif min_y <= h and h <= max_y:
                    temp_x.append( sp([h])[0] )
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    # print(f"[Debug]: Last y")
                    if len(temp_x)<2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                        # print(f"[Debug]: Last Second Y -1")
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                        # print(f"[Debug]: Last Second Y -2")
                else:
                    print(f"[Debug] Last >>> {last}")
                    print(f"[Debug] Last Second >>> {last_second}")
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > self.resize[0] or l < 0 :
                            # print(f"[Debug] Appending 1")
                            temp_x.append(-2)
                        else:
                            # print(f"[Debut] Appending l-1 >> {l}")
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        print(f"L >>>> {l}")
                        if l > self.resize[0] or l < 0 :
                            temp_x.append(-2)
                            # print(f"[Debug] Appending 2")
                        elif l < self.resize[0]:
                            temp_x.append(l)
                            print(f"[Debug] Appending l-2 >> {l}")


            x_fitted.append(temp_x[::-1])
            y_fitted.append(temp_y[::-1]) 

        return x_fitted, y_fitted
    
    @staticmethod
    def draw_points(x, y, image):
        color_index = 0
        for i, j in zip(x, y):
            # color_index += 1
            # if np.sum(np.array(i)>=0) <= 1:
            #     continue
            # print(f"[Debug] In i >>> {i}")
            # print(f"[Debug] In j >>> {j}")

            filter_x_y = [(x, y) for x, y in zip(i, j) if x >= 0]
            # filter_y = [y for y in j if y >= 0]
            # filter_y = 
            for index in range(len(filter_x_y)-1):
                # print(f"i >>> {i}")
                # image = cv2.circle(image, (int(i[index]), int(j[index])), 5, p.color[color_index], -1)
                # cv2.putText(viz_image, str(avgFPS), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # if int(i[index]) == -2:
                #     print(f">>>>>> -2")
                #     continue
                
                # image = cv2.circle(image, (int(filter_x_y[index][0]), int(filter_x_y[index][1])), 4, (255, 0, 0), -1)
                # cv2.putText(image, str(int(filter_x_y[index][0])), (int(filter_x_y[index][0]), int(filter_x_y[index][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # print(f"[Debug]: X >>>> {(int(filter_x_y[index][0]), int(filter_x_y[index][1]))}")
                image = cv2.line(image, (int(filter_x_y[index][0]), int(filter_x_y[index][1])), (int(filter_x_y[index+1][0]), int(filter_x_y[index+1][1])), (0, 255, 0), 2)
                # viz_img = cv2.resize(image, (512, 256))
                # cv2.imshow("Debug Img", viz_img)
                # key = cv2.waitKey(0)
                # if key == 27:
                #     break
        return image


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
                            args.conf_thresh_point,
                            args.save_result,
                            args.save_path)
    
    inferencer.video_flow()

    # image_path = "dataset/small-client/infer_imgs/"
    # save_viz_pth = "infer_output"
    # img_path_list = get_img_list(image_path)
    # for image_pth in img_path_list:
    #     inferencer.image_flow(image_pth, save_viz_pth)

    