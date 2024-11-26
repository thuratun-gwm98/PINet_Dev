import torch
from torch import nn
import time
import cv2
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree, distance
import imutils
from collections import defaultdict

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
    argparser.add_argument("--weight-file", type=str, default="./pretrained_model/50_tensor_0.3877_lane_detection_network.pth")
    argparser.add_argument("--resize", type=tuple, default=(1920, 768))
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--conf-thresh-point", type=float, default=0.81)
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
                total_time_s = time.time()
                start_time = time.time()
                # x, y, ti = test(self.model, np.array([frame]))
                result = self.model.predict_lanes_test(np.array([data]))
                torch.cuda.synchronize()
                confidences, offsets, instances = result[-1]
                infer_time = time.time() - start_time
                # print(f"[Debug]: Inference Time >>> {infer_time}")
                FPS = float(1/infer_time)

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
                # print(f"[Debug]: Offset >> {offset}")
                raw_x, raw_y = self.generate_result(confidence, offset, instance, self.conf_thresh_point, deepcopy(image))

                # raw_x, raw_y = self.generate_result_2(confidence, offset, instance, self.conf_thresh_point, deepcopy(image))
                # print(f"[Debug]: Raw x >> {raw_x}")

                # eliminate fewer points
                in_x, in_y = self.eliminate_and_sort_points(raw_x, raw_y) 
                
                # print(f"[Debug]: Before X >>> {in_x}")
                # x_, y_ = self.fitting(in_x, in_y)
                # print(f"[Debug]: After X >>> {x_}")

                # unfit_viz_image = self.draw_points(in_x, in_y, deepcopy(image))
                total_time = time.time() - total_time_s
                # print(f"[Debug]: Total Fitting Time >>> {total_time}")
                post_pro_FPS = float(1/total_time)
                # print(f"[Debug]: Fitting FPS >>> {post_pro_FPS}")

                viz_image = self.draw_points(in_x, in_y, deepcopy(image))
                

                # resized_unfit_img = cv2.resize(unfit_viz_image, (960, 640))
                # resized_frame = cv2.resize(frame, (960, 640))

                # stacked_frame = cv2.hconcat([resized_unfit_img, resized_fitted_img])

                viz_image = cv2.resize(viz_image, (int(width),int(height)))

                FPSs.append(post_pro_FPS)
                avgFPS = sum(FPSs)/len(FPSs)
                # # print(f"[Debug] Viz Width >>>  {int(width)}")
                cv2.putText(viz_image, str(avgFPS), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # print(f"[INFO]: FPS = {FPS}")

                if self.save_result:
                    out.write(viz_image)
                    
                else:

                    cv2.namedWindow("LaneDetection", cv2.WINDOW_NORMAL)
                    cv2.imshow('LaneDetection',viz_image)

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

                pbar.update(1)

        cap.release()
        cv2.destroyAllWindows()


    def image_flow(self, image_path, text_path, save_viz_path):
        in_img = cv2.imread(image_path)
        img_name = os.path.basename(image_path)
        frame = cv2.resize(in_img, (self.resize[0], self.resize[1]))
        # frame = np.rollaxis(frame, axis=2, start=0)

        gt_xy_points, gt_poly_points = self.read_gt_data(text_path)
        print(f"[Debug]: Lines Per Frame >>>")

        # Resize XY & Poly Points for segmentation mask
        resized_gt_lines = self.resize_x_y_points(gt_xy_points, in_img)

        resized_gt_pts = self.resize_poly_points(gt_poly_points, in_img)

        #### Points Prediction ####
        data = np.rollaxis(frame, axis=2, start=0)/255.0
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

        image = frame.astype(np.uint8).copy()

        grid_x = int(self.resize[0]/dataset_cfg["width_ratio"])      # 64
        grid_y = int(self.resize[1]/dataset_cfg["height_ratio"])     # 32

        confidence = confidences[-1].view(grid_y, grid_x).cpu().data.numpy()

        offset = offsets[-1].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)

        instance = instances[-1].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = self.generate_result(confidence, offset, instance, self.conf_thresh_point, deepcopy(image))

        # raw_x, raw_y = self.generate_result(confidence, offset, instance, self.conf_thresh_point, deepcopy(image))

        # eliminate fewer points
        in_x, in_y = self.eliminate_and_sort_points(raw_x, raw_y)

        # x_, y_ = self.fitting(in_x, in_y)

        # viz_image = self.draw_points(in_x, in_y, deepcopy(image))
        seg_pt_viz_img = self.draw_seg_points(resized_gt_pts, in_x, in_y, deepcopy(image))

        seg_line_viz_img = self.draw_seg_lines(resized_gt_lines, in_x, in_y, deepcopy(image))

        if self.save_result:
            filename = os.path.join(save_viz_path, img_name)
            cv2.imwrite(filename, seg_line_viz_img)
            print(f"[Saving...........]")
        else:
            cv2.imshow("Test", seg_line_viz_img)
            key = cv2.waitKey(0)
            if key==27:

                cv2.destroyAllWindows()

    def generate_result(self, confidance, offsets, instance, thresh, image):
        viz_image = image.copy()
        mask = confidance > thresh
        # print(f"[Debug] Mask >> {mask} ")

        grid = p.grid_location[mask]
        # print(f"[Debug] Grid Location >> {p.grid_location.shape}")
        # print(f"[Debug] Grid >> {grid}")
        offset = offsets[mask]
        # print(f"[Debug] Offset >> {offset}")
        # print(f"Confidance >> {confidance}")
        feature = instance[mask]
        conf = confidance[mask]
        # print(f"Conf >> {conf}")
        # print(f"Feature >> {feature}")
        
        lane_feature = []
        x = []
        y = []
        for i in range(len(grid)):
            # print(f"Feature of idx >>> {feature[i]}")
            # print(f"Feature sqrt >> {feature[i]**2}")
            if (np.sum(feature[i]**2))>=0:
                point_x = int((offset[i][0]+grid[i][0])*p.x_ratio)
                point_y = int((offset[i][1]+grid[i][1])*p.y_ratio)
                # cv2.circle(viz_image, (point_x, point_y), 5, (0, 0, 255), -1)
                conf_txt = round(conf[i], 3)
                # print(f"Confidence >> {conf_txt}")
                # cv2.putText(viz_image, f"Conf: {conf_txt}", (point_x, point_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                    continue
                if len(lane_feature) == 0:              # initially, Lane Feature is Zero
                    lane_feature.append(feature[i])         # Then, APPEND very-first feature[i] 
                    x.append([point_x])
                    y.append([point_y])
                else:
                    flag = 0
                    index = 0
                    min_feature_index = -1
                    min_feature_dis = 10000 #10000
                    for feature_idx, j in enumerate(lane_feature):
                        dis = np.linalg.norm((feature[i] - j)**2)  # Find the distance between prev & current feature
                        # print(f"Distance >>> {dis}")
                        if min_feature_dis > dis:    # If the min_feature_dis is greater than the new_distance (dis)
                            min_feature_dis = dis           # The new_distance will be min_feature_dis
                            min_feature_index = feature_idx                 # Record the min_feature_index with the index of new_distance
                    # print(f"Min Feature Distance >>> {min_feature_dis}")
                    # print(f"Lane Feature Length >>> {len(lane_feature)}")

                    if min_feature_dis <= p.threshold_instance:
                        cv2.circle(viz_image, (point_x, point_y), 3, (0, 255, 0), -1)
                        # print(f"Lane Feature >>> {lane_feature[min_feature_index]}")
                        # print(f"X min_feat_idx >>> {len(x[min_feature_index])}")
                        # print(f"Value >>> {(lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)}")

                        lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                        x[min_feature_index].append(point_x)
                        y[min_feature_index].append(point_y)
                    elif len(lane_feature) < 12:  #12
                        # print(f"Normal Clustering")
                        lane_feature.append(feature[i])
                        x.append([point_x])
                        y.append([point_y])

                # cv2.imshow("Raw Viz Image", viz_image)
                # key = cv2.waitKey(0)
                # if key==27:
                #     cv2.destroyAllWindows()
                #     break
                    
        return x, y

    def resize_poly_points(self, gt_poly_pts, in_img):
        orig_height, orig_width = in_img.shape[:2]

        scale_x = self.resize[0] / orig_width
        scale_y = self.resize[1] / orig_height

        scaled_points = []

        for points in gt_poly_pts:

            points = np.array(points).reshape(-1, 2)
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y

            # convert points to interget coordinates
            points = points.astype(np.int32)

            scaled_points.append(points)

        # print(f"[Debug]: Scaled Points >>> {scaled_points}")
        return scaled_points

    def resize_x_y_points(self, gt_xy_pts, in_img):

        orig_height, orig_width = in_img.shape[:2]

        scale_x = self.resize[0] / orig_width
        scale_y = self.resize[1] / orig_height
        resized_lines_pts = []
        for xy_pt in gt_xy_pts:
            resized_single_line = []
            for i in range(0, len(xy_pt)-1, 2):
                # print(f"[Debug]: Pt >> {xy_pt[i]}")
                x_pts = xy_pt[i] * scale_x
                y_pts = xy_pt[i+1] * scale_y
                resized_single_line.extend([x_pts, y_pts])

            # print(f"[Debug]: Resize xy pts >>> {resized_single_line}")

            resized_lines_pts.append(resized_single_line)

        return resized_lines_pts
        

        # resized_xy_points = []


    def read_gt_data(self, ann_txt_pth):
        lines_per_frame = []
        poly_points_list = []

        with open(ann_txt_pth) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                lines_per_frame.append([int(eval(x)) for x in l[:]])
                # print(f"[Debug]: Lines Per Frame >>> {lines_per_frame}")

                points = list(map(float, line.split()))
                # points = np.array(points).reshape(-1, 2).astype(np.int32)
                poly_points_list.append(points)
                # print(f"[Debug]: Points >>> {points}")

        return lines_per_frame, poly_points_list

    # def gen_lane_mask()
    
    def generate_result_2(self, confidance, offsets,instance, thresh, image):

        mask = confidance > thresh

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
    

    # def gt_to_seg(self, image, txt_file):


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
                    # print(f"[Debug] Last >>> {last}")
                    # print(f"[Debug] Last Second >>> {last_second}")
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
                        # print(f"L >>>> {l}")
                        if l > self.resize[0] or l < 0 :
                            temp_x.append(-2)
                            # print(f"[Debug] Appending 2")
                        elif l < self.resize[0]:
                            temp_x.append(l)
                            # print(f"[Debug] Appending l-2 >> {l}")


            x_fitted.append(temp_x[::-1])
            y_fitted.append(temp_y[::-1]) 

        return x_fitted, y_fitted

    @staticmethod
    def draw_seg_points(poly_pts_gt, x, y, image):

        mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        for points in poly_pts_gt:
            # print(f"[Debug]: Points >>> {points}")
            cv2.fillPoly(mask, [points], (0, 255, 255))
            
        # mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(image, 0.8, mask, 0.2, 0)

        # cv2.imshow("Seg Img", overlay)
        # key = cv2.waitKey(0)

        # if key==27:
        #     cv2.destroyWindow()
        return overlay
    
    def modified_instances(self, x_seg, y_seg, x_kps, y_kps, distance_threshold=15):

        new_x = []
        new_y = []

        modified_lanes = defaultdict(list)

        for xpt, ypt in zip(x_seg, y_seg):
            seg_pts = [xpt, ypt]

            lane_instance = 1
            another_instances = len(x_kps) + 1
            for i, j in zip(x_kps, y_kps):
                filter_kp = [(x, y) for x, y in zip(i, j) if x >= 0]

                new_i = i
                new_j = j

                different_i = []
                different_j = []

                for x, y in zip(i, j):
                    pred_kp = [x, y]
                    eucli_distance = distance.euclidean(pred_kp, seg_pts)

                    if eucli_distance < distance_threshold:
                        modified_lanes[lane_instance].append((xpt, ypt))
                        # new_i.append(xpt)
                        # new_j.append(ypt)
                    else:
                        # different_i.append(xpt)
                        # different_j.append(ypt)
                        modified_lanes[another_instances].append((xpt, ypt))

                lane_instance += 1

            new_x.append(new_i)
            new_y.append(new_j)

            new_x.append(different_i)
            new_y.append(different_j)

        return new_x, new_y



    def draw_seg_lines(self, resized_lines, x, y, image):

        mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Dummy Segmentation
        for line in resized_lines:
            # print(f"[Debug]: Points >>> {points}")
            # cv2.fillPoly(mask, [points], (0, 255, 255))
            for i in range(0, len(line)-2, 2):
                start_pt = (int(line[i]), int(line[i+1]))
                end_pt = (int(line[i+2]), int(line[i+3]))

                cv2.line(mask, start_pt, end_pt, (255, 120, 203), 10)
            
        # mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # print(f"Mask Output >> {mask}")

        ### Adding Dummy Points to Segment###
        self.points_with_numpy(mask)
        _, xpts, ypts = self.points_with_contour(mask)


        # new_x, new_y = self.modified_instances(xpts, ypts, x, y)


        # self.points_with_skele(mask)
        # self.points_wiht_houghline(mask)
        # self.filter_pts

        overlay = cv2.addWeighted(image, 0.8, mask, 0.5, 0)

        kp_img = overlay.copy()
        seg_pt_img = overlay.copy()
        colors = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]
        
        color_idx = 0
        for i, j in zip(x, y):
            # self.filter_pts()
            color_idx += 1

            filter_kp = [(x, y) for x, y in zip(i, j) if x >= 0]
            for index in range(len(filter_kp)-1):
                cv2.circle(kp_img, (int(filter_kp[index][0]), int(filter_kp[index][1])), 5, colors[color_idx], -1)

            f_xpt, f_ypt, i_xpt, i_ypt = self.filter_pts(i, j, xpts, ypts)
            filter_x_y = [(x, y) for x, y in zip(f_xpt, f_ypt) if x >= 0]

            for index in range(len(filter_x_y)-1):
                cv2.circle(seg_pt_img, (int(filter_x_y[index][0]), int(filter_x_y[index][1])), 5, colors[color_idx], -1)

        self.display_img(kp_img, "Kp Img")
        self.display_img(seg_pt_img, "OverLay Img")

        return overlay

    
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

    def points_with_numpy(self, mask):
        mask_copy = mask.copy()
        s_time = time.time()
        lane_lines_points = np.column_stack(np.where(mask_copy > 0))
        sampled_points = lane_lines_points[::300]
        for y_, x_, z_ in sampled_points:
            cv2.circle(mask_copy, (x_, y_), radius=3, color=(255, 255, 255), thickness=-1)

        total_time = time.time() - s_time
        FPS = float(1/total_time)
        print(f"[Debug]: Points With Numpy FPS >> {FPS}")

        self.display_img(mask_copy, "Numpy Pt Method")

    def points_with_contour(self, mask):
        mask_copy = cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY)
        mask_copy_ = mask.copy()
        s_time = time.time()
        contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours(contours)
        print(f"[Debug]: Contours >>> {contours}")
        # smooth_contour_pts = cv2.approxPolyDP(contours, epsilon=1, closed=False)
        contour_points = np.vstack([contour[:, 0, :] for contour in contours])
        
        lane_line_points = []
        print(f"Stacked Contours >>> {contour_points}")
        # for contour in contour_points:
        #     print(f"Contour >> {contour}")
        #     for point in contour:
        #         x, y = point[0]
        #         lane_line_points.append((x, y))
        for point in contour_points:
            x, y = point[0], point[1]
            lane_line_points.append((x, y))
        print(f"Lane Line Points >> {lane_line_points}")

        sampled_points = lane_line_points[::10]

        # Create Refined Midline by averaging or smoothing filtered pts
        # smooth_sampled_pts = cv2.approxPolyDP(sampled_points, epsilon=1, )
        
        x_pts = []
        y_pts = []
        for pt in sampled_points:
            x_pts.append(pt[0])
            y_pts.append(pt[1])
            cv2.circle(mask_copy_, (pt[0], pt[1]), 5, (255, 255, 255), -1)
        # for x_s, y_s in zip(x_smooth, y_smooth):
        #     cv2.circle(mask_copy_, (int(x_s), int(y_s)), 5, (255, 255, 255), -1)
        
        total_time = time.time() - s_time
        fps = float(1/total_time)
        print(f"[Debug]: Points with Contour  FPS >> {fps}")

        self.display_img(mask_copy_, "Contour Pt Method")

        return mask_copy_, x_pts, y_pts

    def filter_pts(self, kp_x, kp_y, x_pts, y_pts, distance_threshold=15):
        # filtered_xpts = []
        # filtered_ypts = []
        filtered_x_pt = []
        filtered_y_pt = []
        lanes_line = 
        other_instance_xpt = []
        other_instance_ypt = []
        for kpx, kpy in zip(kp_x, kp_y):
            # close_pts = [pt=[x_pt, y_pt] for x_pt, y_pt in zip(x_pts, y_pts) if ]
            pred_kp = [kpx, kpy]
            for x_pt, y_pt in zip(x_pts, y_pts): 
                ctn_pts = [x_pt, y_pt]
                eucli_dist = distance.euclidean(ctn_pts, pred_kp)

                if eucli_dist < distance_threshold:
                    filtered_x_pt.append(x_pt)
                    filtered_y_pt.append(y_pt)

                else:
                    other_instance_xpt.append(x_pt)
                    other_instance_ypt.append(y_pt)


        filtered_x_pt.extend(kp_x)
        filtered_y_pt.extend(kp_y)

        return filtered_x_pt, filtered_y_pt, other_instance_xpt, other_instance_ypt


    def points_with_skele(self, mask):
        mask_copy = mask.copy()
        s_time = time.time()
        mask_binary = (mask > 0).astype(np.uint8)
        skeleton = skeletonize(mask_binary).astype(np.uint8)
        lane_lines_points = np.column_stack(np.where(skeleton > 0))
        sampled_points = lane_lines_points[::2]
        for y_, x_, z_ in sampled_points:
            cv2.circle(mask_copy, (x_, y_), radius=5, color=(255, 255, 255), thickness=-1)

        total_time = time.time() - s_time
        fps = float(1/total_time)
        print(f"[Debug]: Points with Skeletonize+Numpy FPS >> {fps}")

        self.display_img(skeleton, "SkeletonIMg")

    
    def points_wiht_houghline(self, mask):
        mask_copy = cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY)
        mask_copy_ = mask.copy()
        time_s = time.time()
        lines = cv2.HoughLinesP(mask_copy, rho=1, theta=np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
        line_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_points.append((x1, y1))
                line_points.append((x2, y2))

        sampled_points = line_points[::10]

        for pt in sampled_points:
            cv2.circle(mask_copy_, (pt[0], pt[1]), 5, (255, 255, 255), thickness=-1)

        total_time = time.time() - time_s
        fps = float(1/total_time)

        print(f"[Debug]: HoughLine Pts >> {sampled_points}")

        print(f"[Debug]: Points with HoughLine FPS >> {fps}")

        self.display_img(mask_copy_, "HoughLine Viz")


    @staticmethod
    def display_img(image, image_name="Debug Display"):
        resized_img = cv2.resize(image, (768, 340))
        cv2.imshow(image_name, resized_img)

        key = cv2.waitKey(0)

        if key==27:
            cv2.destroyAllWindows()




def get_img_and_txt(image_path):
    img_path_list = []
    text_path_list = []
    for (root, dirs, files) in os.walk(image_path):
        if len(files) > 0:
            for filename in files:
                if os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.bmp', '.png']:
                    img_path = root + '/' + filename
                    txt_filename = f"{os.path.splitext(filename)[0]}.lines.txt"
                    txt_path = root + '/' + txt_filename
                    img_path_list.append(img_path)
                    text_path_list.append(txt_path)
    return img_path_list, text_path_list





if __name__ == "__main__":
    args = get_args()
    inferencer = VideoInfer(args.input_video,
                            args.weight_file,
                            args.resize,
                            args.device,
                            args.conf_thresh_point,
                            args.save_result,
                            args.save_path)
    
    # inferencer.video_flow()

    image_path = "dataset/infer_branch_img/"
    save_viz_pth = "infer_output"
    img_path_list, text_path_list = get_img_and_txt(image_path)
    for image_pth, text_pth in zip(img_path_list, text_path_list):
        inferencer.image_flow(image_pth, text_pth, save_viz_pth)

    