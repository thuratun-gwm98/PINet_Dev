#########################################################################
##
##  Data loader source code for TuSimple dataset
##
#########################################################################

import os
import math
import numpy as np
import cv2
import json
import random
from pathlib import Path
from copy import deepcopy
from configs.parameters import DATASET_CFG

#########################################################################
## some iamge transform utils
#########################################################################
def Translate_Points(point,translation): 
    point = point + translation 
    
    return point

def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


#########################################################################
## Data loader class
#########################################################################
class DataGenerator(object):
    ################################################################################
    ## initialize (load data set from url)
    ################################################################################
    def __init__(self):
        # self.p = Parameters()
        self.dataset_cfg = DATASET_CFG
        self.dataset_root = self.dataset_cfg['dataset_root_dir']
        print(f"[Debug]: Dataset Root >>> {self.dataset_root}")

        # Train & Test Set
        self.train_set = os.path.join(self.dataset_root, f"list/train.txt")
        # print(f"Training Set >> {self.train_set}")
        self.test_set = os.path.join(self.dataset_cfg["dataset_root_dir"], f"list/test.txt")

        # load training set
        self.train_data = []
        
        with open(self.train_set) as f:
            self.train_data = f.readlines()

        self.size_train = len(self.train_data)
        # print(f"[Debug] Train data >> {self.train_data}")

        # load test set
        self.test_data = []
        with open(self.test_set) as f:
            self.test_data = f.readlines()

        self.size_test = len(self.test_data)

    #################################################################################################################
    ## Generate data as much as batchsize and augment data (filp, translation, rotation, gaussian noise, scaling)
    #################################################################################################################
    def Generate(self, sampling_list = None): 
        cuts = [(b, min(b + self.dataset_cfg["batch_size"], self.size_train)) for b in range(0, self.size_train, self.dataset_cfg["batch_size"])]
        print(f"Cuts >>> {cuts}")
        random.shuffle(self.train_data)
        random.shuffle(self.train_data)
        random.shuffle(self.train_data)
        for start, end in cuts:
            print(f"Start >>> {start}")
            print(f"End >>> {end}")
            # resize original image to 512*256
            self.inputs, self.target_lanes, self.target_h, self.test_image, self.data_list = self.Resize_data(start, end, sampling_list)
            # print(f"INputsss >>> {type(self.inputs)}")
            self.actual_batchsize = self.inputs.shape[0]
            self.Flip()
            self.Translation()
            self.Rotate()
            self.Gaussian()
            self.Change_intensity()
            self.Shadow()

            print(f"Inputtss >>> {self.inputs[0].shape}")

            yield self.inputs/255.0, self.target_lanes, self.target_h, self.test_image/255.0, self.data_list  # generate normalized image

    #################################################################################################################
    ## Generate test data
    #################################################################################################################
    def Generate_Test(self): 
        cuts = [(b, min(b + self.dataset_cfg["batch_size"], self.size_test)) for b in range(0, self.size_test, self.dataset_cfg["batch_size"])]
        for start, end in cuts:
            test_image, path, ratio_w, ratio_h, target_h, target_lanes = self.Resize_data_test(start, end)
            yield test_image/255.0, ratio_w, ratio_h, path, target_h, target_lanes

    #################################################################################################################
    ## resize original image to 512*256 and matching correspond points
    #################################################################################################################

    def Resize_data_test(self, start, end):
        inputs = []
        path = []
        target_lanes = []
        target_h = []

        for i in range(start, end):
            test_img = self.test_data[i]
            # print(f"[Debug] Data:: {data}")
            # image_path = os.path.join(self.dataset_root, Path(data))
            test_image_path = self.dataset_root + '/' + test_img[0:-1]
            # image_path = os.path.join(self.dataset_root, test_img[0:-1])
            # img_pth = self.p.test_root_url+data[1:-1]
            # print(os.path.exists(img_pth))
            # temp_image_d = cv2.imread(self.p.test_root_url+data[0:-1])
            # print(temp_image_d.shape)
            temp_image = cv2.imread(test_image_path)
            original_size_x = temp_image.shape[1]
            original_size_y = temp_image.shape[0]
            ratio_w = self.dataset_cfg['img_width']*1.0/temp_image.shape[1]
            ratio_h = self.dataset_cfg['img_height']*1.0/temp_image.shape[0]
            temp_image = cv2.resize(temp_image, (self.dataset_cfg['img_width'], self.dataset_cfg['img_height']))
            inputs.append( np.rollaxis(temp_image, axis=2, start=0) )
            path.append(test_img[0:-1])

            temp_lanes = []
            temp_h = []

            # annoatation = self.p.test_root_url+test_img[0:-4]+"lines.txt"
            # annoatation = os.path.join(self.dataset_root, f"{test_img[0:-4]}lines.txt")
            annotation = self.dataset_root + '/' + f"{test_img[0:-4]}lines.txt"
            # print(f"[debug]: Annotation >> {annoatation}")

            # print(f"[Debug]: annotation >>> {annoatation}")
            with open(annotation) as f:
                annoatation_data = f.readlines()
            # print(f"[Debug]: test annotation data_j >>> {annoatation_data}")
            for j in annoatation_data:
                # print(f"[Debug]: J ann > {j}")
                
                x = []
                y = []
                temp_x = j.split()[0::2]
                # print(f"[Debug]: temp_x > {temp_x}")
                temp_y = j.split()[1::2]
                # print(f"[Debug]: temp_y > {temp_y}")
                # print(f"Temp_X >>> {temp_x}")
                # print(f"Temp_Y >>> ", temp_y)

                for k in range(len(temp_x)):
                    x_value = float(temp_x[k])
                    y_value = int(float(temp_y[k]))
                    if 0 < x_value < original_size_x and 0 < y_value < original_size_y:
                        x.append( x_value )
                        y.append( y_value )

                temp_lanes.append( x )
                temp_h.append( y )
            target_lanes.append(np.array(temp_lanes))
            target_h.append(np.array(temp_h))

        return np.array(inputs), path, ratio_w, ratio_h, target_h, target_lanes

    def Resize_data(self, start, end, sampling_list):
        inputs = []
        target_lanes = []
        target_h = []
        data_list = []

        # choose data from each number of lanes
        for i in range(start, end):
            print(f"Start >> {start}")
            print(f"End >>> {end}")
            if sampling_list == None:
                train_img = random.sample(self.train_data, 1)[0]
                #data = self.train_data[0]
                data_list.append(train_img)
            elif len(sampling_list) < 10:
                train_img = random.sample(self.train_data, 1)[0]
                data_list.append(train_img)
            else:            
                choose = random.random()
                if choose > 0.2:
                    train_img = random.sample(self.train_data, 1)[0]
                    data_list.append(train_img)
                else:
                    train_img = random.sample(sampling_list, 1)[0]
                    data_list.append(train_img)

            # train set image
            # print(f"Data >>> {train_img[0:-1]}")
            # print(f"[Debug]: Dataset Root 2 >> {self.dataset_root}")
            # print(f"[Debug]: Temp Image Path : {self.dataset_root + train_img[0:-1]}")
            
            # temp_image = cv2.imread(os.path.join(self.dataset_root, train_img[0:-1]))
            # print(f"Train Img >> {train_img}")
            
            train_img_pth = train_img.strip()
            # print(f"Train Img Path >>> {train_img_pth}")
            ext_idx = train_img_pth.find('.png')
            train_img_fp = train_img_pth[:ext_idx + 4]
            # print(f"Train Img :: >> {self.dataset_root + '/' + train_img_fp}")
            train_img_path = self.dataset_root + '/' + train_img_fp
            temp_image = cv2.imread(train_img_path)
            
            if i==start:
                print(train_img[1:-1])
            original_size_x = temp_image.shape[1]
            original_size_y = temp_image.shape[0]
            ratio_w = self.dataset_cfg['img_width']*1.0/temp_image.shape[1]
            ratio_h = self.dataset_cfg['img_height']*1.0/temp_image.shape[0]
            temp_image = cv2.resize(temp_image, (self.dataset_cfg['img_width'],self.dataset_cfg['img_height']))
            inputs.append( np.rollaxis(temp_image, axis=2, start=0) )

            temp_lanes = []
            temp_h = []

            # annoatation = self.p.train_root_url+data[0:-4]+"lines.txt"
            # annotation = os.path.join(self.dataset_root, f"{train_img[0:-4]}lines.txt")
            annotation = self.dataset_root + '/' + f"{train_img_pth[:ext_idx]}.lines.txt"
            with open(annotation) as f:
                annoatation_data = f.readlines()         

            for j in annoatation_data:
                x = []
                y = []
                temp_x = j.split()[0::2]
                temp_y = j.split()[1::2]

                for k in range(len(temp_x)):
                    x_value = float(temp_x[k])
                    y_value = int(float(temp_y[k]))
                    if 0 < x_value < original_size_x and 0 < y_value < original_size_y:
                        x.append( x_value )
                        y.append( y_value )

                l, h = self.make_dense_x(np.array(x), np.array(y))
                temp_lanes.append( l*ratio_w )
                temp_h.append( h*ratio_h )
            target_lanes.append(np.array(temp_lanes))
            target_h.append(np.array(temp_h))
        

        #test set image
        test_index = random.randrange(0, self.size_test-1)
        # test_image = cv2.imread(self.p.test_root_url+self.test_data[test_index][1:-1])
        # train_image = cv2.imread(os.path.join(self.dataset_root, self.test_data[test_index][0:-1]))
        train_immage_path = self.dataset_root + '/' + self.test_data[test_index][0:-1]
        test_image = cv2.imread(train_immage_path)
        test_image = cv2.resize(test_image, (self.dataset_cfg['img_width'],self.dataset_cfg['img_height']))
        
        return np.array(inputs), target_lanes, target_h, np.rollaxis(test_image, axis=2, start=0), data_list

    def make_dense_x(self, l, h):
        out_x = []
        out_y = []

        p_x = -1
        p_y = -1
        for x, y in zip(l, h):
            if x > 0:
                if p_x < 0:
                    p_x = x
                    p_y = y
                else:
                    out_x.append(x)
                    out_y.append(y)
                    for dense_x in range(min(int(p_x), int(x)), max(int(p_x), int(x)), 10):
                        dense_y = p_y - abs(p_x - dense_x) * abs(p_y-y)/float(abs(p_x - x))
                        if dense_x>=0 and dense_y>=0:
                            out_x.append(dense_x)
                            out_y.append( p_y - abs(p_x - dense_x) * abs(p_y-y)/float(abs(p_x - x)) )
                    p_x = x
                    p_y = y

        return np.array(out_x), np.array(out_y)
        #return l, h

    #################################################################################################################
    ## Generate random unique indices according to ratio
    #################################################################################################################
    def Random_indices(self, ratio):
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)

    #################################################################################################################
    ## Add Gaussian noise
    #################################################################################################################
    def Gaussian(self):
        indices = self.Random_indices(self.dataset_cfg['noise_ratio'])
        img = np.zeros((self.dataset_cfg['img_height'],self.dataset_cfg['img_width'], 3), np.uint8)
        m = (0,0,0) 
        s = (20,20,20)
        
        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            cv2.randn(img,m,s)
            test_image = test_image + img
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            self.inputs[i] = test_image

    #################################################################################################################
    ## Change intensity
    #################################################################################################################
    def Change_intensity(self):
        indices = self.Random_indices(self.dataset_cfg['intensity_ratio'])
        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            test_image =  np.rollaxis(test_image, axis=2, start=0)

            hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            value = int(random.uniform(-60.0, 60.0))
            if value > 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = -1*value
                v[v < lim] = 0
                v[v >= lim] -= lim                
            final_hsv = cv2.merge((h, s, v))
            test_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            self.inputs[i] = test_image

    #################################################################################################################
    ## Generate random shadow in random region
    #################################################################################################################
    def Shadow(self, min_alpha=0.5, max_alpha = 0.75):
        indices = self.Random_indices(self.dataset_cfg['shadow_ratio'])
        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image =  np.rollaxis(test_image, axis=2, start=0)
            test_image =  np.rollaxis(test_image, axis=2, start=0)

            top_x, bottom_x = np.random.randint(0, self.dataset_cfg['img_width'], 2)
            coin = 0
            rows, cols, _ = test_image.shape
            shadow_img = test_image.copy()
            if coin == 0:
                rand = np.random.randint(2)
                vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
                if rand == 0:
                    vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
                elif rand == 1:
                    vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
                mask = test_image.copy()
                channel_count = test_image.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (0,) * channel_count
                cv2.fillPoly(mask, [vertices], ignore_mask_color)
                rand_alpha = np.random.uniform(min_alpha, max_alpha)
                cv2.addWeighted(mask, rand_alpha, test_image, 1 - rand_alpha, 0., shadow_img)
                shadow_img =  np.rollaxis(shadow_img, axis=2, start=0)
                self.inputs[i] = shadow_img

    #################################################################################################################
    ## Flip
    #################################################################################################################
    def Flip(self):
        indices = self.Random_indices(self.dataset_cfg['flip_ratio'])
        for i in indices:
            temp_image = deepcopy(self.inputs[i])
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)

            temp_image = cv2.flip(temp_image, 1)
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image

            x = self.target_lanes[i]
            for j in range(len(x)):
                x[j][x[j]>0]  = self.dataset_cfg['img_width'] - x[j][x[j]>0]
                x[j][x[j]<0] = -2
                x[j][x[j]>=self.dataset_cfg['img_width']] = -2

            self.target_lanes[i] = x

    #################################################################################################################
    ## Translation
    #################################################################################################################
    def Translation(self):
        indices = self.Random_indices(self.dataset_cfg['translation_ratio'])
        for i in indices:
            temp_image = deepcopy(self.inputs[i])
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)       

            tx = np.random.randint(-50, 50)
            ty = np.random.randint(-30, 30)

            temp_image = cv2.warpAffine(temp_image, np.float32([[1,0,tx],[0,1,ty]]), (self.dataset_cfg['img_width'], self.dataset_cfg['img_height']))
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image

            x = self.target_lanes[i]
            for j in range(len(x)):
                x[j][x[j]>0]  = x[j][x[j]>0] + tx
                x[j][x[j]<0] = -2
                x[j][x[j]>=self.dataset_cfg['img_width']] = -2

            y = self.target_h[i]
            for j in range(len(y)):
                y[j][y[j]>0]  = y[j][y[j]>0] + ty
                x[j][y[j]<0] = -2
                x[j][y[j]>=self.dataset_cfg['img_height']] = -2

            self.target_lanes[i] = x
            self.target_h[i] = y

    #################################################################################################################
    ## Rotate
    #################################################################################################################
    def Rotate(self):
        indices = self.Random_indices(self.dataset_cfg['rotate_ratio'])
        for i in indices:
            temp_image = deepcopy(self.inputs[i])
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)  

            angle = np.random.randint(-10, 10)

            M = cv2.getRotationMatrix2D((self.dataset_cfg['img_width']//2, self.dataset_cfg['img_height']//2),angle,1)

            temp_image = cv2.warpAffine(temp_image, M, (self.dataset_cfg['img_width'], self.dataset_cfg['img_height']))
            temp_image =  np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image

            x = self.target_lanes[i]
            y = self.target_h[i]

            for j in range(len(x)):
                index_mask = deepcopy(x[j]>0)
                x[j][index_mask], y[j][index_mask] = Rotate_Points((self.dataset_cfg['img_width']//2, self.dataset_cfg['img_height']//2),(x[j][index_mask], y[j][index_mask]),(-angle * 2 * np.pi)/360)
                x[j][x[j]<0] = -2
                x[j][x[j]>=self.dataset_cfg['img_width']] = -2
                x[j][y[j]<0] = -2
                x[j][y[j]>=self.dataset_cfg['img_height']] = -2

            self.target_lanes[i] = x
            self.target_h[i] = y
