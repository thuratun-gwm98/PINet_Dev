#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
import numpy as np
from copy import deepcopy
import time
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models import model_helper
from src.data.data_parameters import Parameters
from src.data import util
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from patsy import cr
import csaps
from configs.parameters import DATASET_CFG
from src.data.data_loader import DataGenerator

p = Parameters()
dataset_cfg = DATASET_CFG
###############################################################
##
## Training
## 
###############################################################
def Testing():
    print('Testing')
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = DataGenerator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = model_helper.ModelAgent()
    else:
        lane_agent = model_helper.ModelAgent()
        lane_agent.load_weights(p.model_path)
	
    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## testing
    ##############################
    print('Testing loop')
    lane_agent.evaluate_mode()
    # print(f"[INFO]: Current Mode is >>> {p.mode}")
    if p.mode == 0 : # check model with test data 
        # print(f"[INFO]: Mode-0")
        for _, _, _, test_image, _ in loader.Generate():
            _, _, ti = test(lane_agent, np.array([test_image]))
            cv2.imshow("test", ti[0])
            cv2.waitKey(0) 

    elif p.mode == 1: # check model with video
        cap = cv2.VideoCapture("/home/thuratun/GW_workspace/CS2/Hitachi_Astemo_Prj/MONO_Lss/MonoLSS/kitti/inference_data/testing_videos/output_0_3.mp4")
        while(cap.isOpened()):
            ret, frame = cap.read()
            torch.cuda.synchronize()
            prevTime = time.time()
            frame = cv2.resize(frame, (512, 256))/255.0
            frame = np.rollaxis(frame, axis=2, start=0)
            _, _, ti = test(lane_agent, np.array([frame])) 
            curTime = time.time()
            sec = curTime - prevTime
            fps = 1/(sec)
            s = "FPS : "+ str(fps)
            ti[0] = cv2.resize(ti[0], (1280,800))
            cv2.putText(ti[0], s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow('frame',ti[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif p.mode == 2: # check model with a picture
        #test_image = cv2.imread(p.test_root_url+"clips/0530/1492720840345996040_0/20.jpg")
        test_image = cv2.imread("./aa.png")
        test_image = cv2.resize(test_image, (512,256))/255.0
        test_image = np.rollaxis(test_image, axis=2, start=0)
        _, _, ti = test(lane_agent, np.array([test_image]))
        cv2.imshow("test", ti[0])
        cv2.waitKey(0)   

    elif p.mode == 3: #evaluation
        print("evaluate")
        evaluation(loader, lane_agent)

############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(loader, lane_agent, thresh = p.threshold_point, index= -1, name = None):
    progressbar = tqdm(range(loader.size_test//4))
    for test_image, ratio_w, ratio_h, path, target_h, target_lanes in loader.Generate_Test():
        # print(f"Path >>> {len(path)}")
        # print(f"Test Image >>> {test_image.shape}")
        x, y, out_images = test(lane_agent, test_image, thresh, index= index)
        print(f"Out Images >>> {len(out_images)}")
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)
        #x_, y_ = find_target(x_, y_, ratio_w, ratio_h)
        x_, y_ = fitting(x_, y_, ratio_w, ratio_h)
        print(f"[Debug]: X_ >>> {x_}")
        # print(f"[Debug]: Y_ >>> {y_}")

        for idx, pth in enumerate(path):
            print(f"Path >>> {pth}")
            image_path = dataset_cfg["dataset_root_dir"] + '/' + pth
            image = cv2.imread(image_path)

            # image = deepcopy(test_image[idx])
            # image =  np.rollaxis(image, axis=2, start=0)
            # image =  np.rollaxis(image, axis=2, start=0)*255.0
            # image = image.astype(np.uint8).copy()
            
            # print(f"X_ >> {x_}")
            for x_values, y_values in zip(x_[idx], y_[idx]):
                # print(f"[Debug]: X >> {x_values}")
                # print(f"[Debug]: Y >> {y_values}")
                count = 0
                
                if np.sum(np.array(x_values)>=0) > 1 : ######################################################
                    f_x_values = []
                    f_y_values = []
                    for x_value, y_value in zip(x_values, y_values):
                        if x_value >= 0:
                            f_x_values.append(x_value)
                            f_y_values.append(y_value)
                            count+=1
                    print(f"F x vals >> {f_x_values}")
                    # print(f"F y value >> {f_y_values}")
                    for i in range(len(f_x_values)-1):
                        # viz_image = cv2.circle(image, (int(f_x_values[i]), int(f_y_values[i])), 10, (0, 255, 0), -1)
                        viz_image = cv2.line(image, (int(f_x_values[i]), int(f_y_values[i])), (int(f_x_values[i+1]), int(f_y_values[i+1])), (0, 255, 0), 3)
            viz_image = cv2.resize(viz_image, (1920, 768))
            testing_img_pth = "test_result/images"
            if not os.path.exists(testing_img_pth):
                os.makedirs(testing_img_pth, exist_ok=True)
            image_name = os.path.basename(pth)
            cv2.imwrite(f'{testing_img_pth}/{image_name}', viz_image)

        result_data = write_result(x_, y_, path)
        progressbar.update(1)
    progressbar.close()

############################################################################
## linear interpolation for fixed y value on the test dataset
############################################################################
def find_target(x, y, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []
            for h in range(100, 590, 10):
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    for k in range(len(j)-1):
                        if j[k] >= h and h >= j[k+1]:
                            #linear regression
                            if i[k] < i[k+1]:
                                temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                            else:
                                temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                            break
                else:
                    temp_x.append(-2)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch)            
    
    return out_x, out_y

def fitting(x, y, ratio_w, ratio_h):
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    print(f"[Debug] X Size >>> {x_size}")

    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
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
            for pts in range(300, -1, -1):
                h = 2160 - pts*5 - 1
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    temp_x.append( sp([h])[0] )
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    if len(temp_x)<2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                else:
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x[::-1])
            predict_y_batch.append(temp_y[::-1])
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch) 

    return out_x, out_y

############################################################################
## write result
############################################################################
def write_result(x, y, path):
    
    batch_size = len(path)
    save_path = "test_result"
    for i in range(batch_size):
        # print(f"[Debug]: paht >>> {path}")
        path_detail = path[i].split("/")
        # print(f"[Debug]: path_detal >>> {path_detail}")
        first_folder = path_detail[0]
        print(f"[Debug]: First Folder >>> {first_folder}")
        second_folder = path_detail[1]
        print(f"[Debug]: Second Folder >>> {second_folder}")
        file_name = path_detail[1].split(".")[0]+".lines.txt"
        save_test_path = save_path+"/"+first_folder
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        # if not os.path.exists(save_path+"/"+first_folder):
        #     os.makedirs(save_path+"/"+first_folder)      
        with open(save_test_path+"/"+file_name, "w") as f:  
            for x_values, y_values in zip(x[i], y[i]):
                # print(f"[Debug]: X >> {x_values}")
                # print(f"[Debug]: Y >> {y_values}")
                count = 0
                if np.sum(np.array(x_values)>=0) > 1 : ######################################################
                    for x_value, y_value in zip(x_values, y_values):
                        if x_value >= 0:
                            f.write(str(x_value) + " " + str(y_value) + " ")
                            count += 1
                    if count>1:
                        f.write("\n")


############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh = p.threshold_point, index= -1):

    grid_x = dataset_cfg["img_width"]//dataset_cfg["width_ratio"]       # 64
    grid_y = dataset_cfg["img_height"]//dataset_cfg["height_ratio"]     # 32

    result = lane_agent.predict_lanes_test(test_images)
    torch.cuda.synchronize()
    confidences, offsets, instances = result[index]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []
    
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image =  np.rollaxis(image, axis=2, start=0)
        image =  np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        # cv2.imshow("Debug Test Img", image)
        # key = cv2.waitKey(0)
        # if key == 27:
        #     break

        confidence = confidences[i].view(grid_y, grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh, deepcopy(image))

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)

    return out_x, out_y, out_images

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        # print(f"Eli: x -> {i}")
        # print(f"Eli: y -> {j}")
        if len(i)>5:  # default - 5
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh, image):

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

if __name__ == '__main__':
    Testing()
