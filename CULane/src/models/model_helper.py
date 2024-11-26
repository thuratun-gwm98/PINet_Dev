#########################################################################
##
## train agent that has some utility for training and saving.
##
#########################################################################

import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from copy import deepcopy
import numpy as np
import cv2
from torch.autograd import Variable
from torch.autograd import Function as F
from src.data.data_parameters import Parameters
from configs.parameters import OPTIMIZER_CFG, DATASET_CFG, TRAINER_CFG, LOSS_CFG
import math
from src.data import util
import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models.hourglass_network import lane_detection_network
from src.models.backbones.util_hourglass import *
from src.models.backbones import hard_sampling
from src.models.PI_DDRNet_slim import PI_DDRNet_slim
from src.models.PI_DDRNetSlim_modified import PIDDRNetSlim1
from src.models.PI_DDRNetSlim_modified import PIDDRNetSlim2

############################################################
##
## agent for lane detection
##
############################################################
class ModelAgent(nn.Module):

    #####################################################
    ## Initialize
    #####################################################
    def __init__(self):
        super(ModelAgent, self).__init__()

        self.p = Parameters()
        self.dataset_cfg = DATASET_CFG
        self.optimizer_cfg = OPTIMIZER_CFG
        self.trainer_cfg = TRAINER_CFG

        self.grid_x = self.dataset_cfg["img_width"]//self.dataset_cfg["width_ratio"]       # 64
        self.grid_y = self.dataset_cfg["img_height"]//self.dataset_cfg["height_ratio"]      # 32

        print(f"Grid x >>> {self.grid_x}")
        print(f"Grid y >>> {self.grid_y}")

        # self.lane_detection_network = lane_detection_network()
        # self.lane_detection_network = PI_DDRNet_slim(self.trainer_cfg["pretrained_weight"], self.trainer_cfg["pretrained"])
        self.lane_detection_network = PIDDRNetSlim1(self.trainer_cfg["pretrained_weight"], self.trainer_cfg["pretrained"])

        self.setup_optimizer()

        self.current_epoch = 0

        self.hard_sampling = hard_sampling.hard_sampling()

        print("model parameters: ")
        print(self.count_parameters(self.lane_detection_network))

    def count_parameters(self, model):
	    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_optimizer(self):
        self.l_rate = self.optimizer_cfg["lr"]
        self.lane_detection_optim = torch.optim.AdamW(self.lane_detection_network.parameters(),
                                                    lr=self.l_rate,
                                                    weight_decay=self.optimizer_cfg["weight_decay"])

    #####################################################
    ## Make ground truth for key point estimation
    #####################################################
    def make_ground_truth_point(self, inputs, target_lanes, target_h):

        image =  np.rollaxis(inputs[0], axis=2, start=0)
        image =  np.rollaxis(image, axis=2, start=0)*255.0
        # print(f"[Debug]: Image Shape >> {image.shape}")
        viz_img = image.astype(np.uint8).copy()
        # print(f"[Debug]: Target Lanes >>> {target_lanes[0].ndim}")
        # print(f"[Debug]: Target Heights >>> {target_h}")

        target_lanes, target_h = util.sort_batch_along_y(target_lanes, target_h)
        # util.visualize_points(inputs[0], target_lanes[0], target_h[0], "RawGT_Pts")
        
        ground = np.zeros((len(target_lanes), 3, self.grid_y, self.grid_x))
        ground_binary = np.zeros((len(target_lanes), 1, self.grid_y, self.grid_x))

        for batch_index, batch in enumerate(target_lanes):  # All Lanes - X pts
            for lane_index, lane in enumerate(batch):       # Single Batch
                
                for point_index, point in enumerate(lane):  # X pt
                    # print(f"[Debug]: Point >>> {point}")
                    if point > 0:
                        x_index = int(point/self.dataset_cfg["width_ratio"])
                        # print(f"[Debug]: X idx >> {x_index}")
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.dataset_cfg["height_ratio"])

                        x_pt = int(point)
                        y_pt = int(target_h[batch_index][lane_index][point_index])

                        # x_pt_n = int(target_lanes[batch_index][lane_index][point_index+1])
                        # y_pt_n = int(target_h[batch_index][lane_index][point_index+1])
                        # cv2.line(viz_img, (x_pt, y_pt), (x_pt_n, y_pt_n), (0, 255, 0), 2)
                        cv2.circle(viz_img, (int(x_pt), int(y_pt)), 3, (0, 255, 0), -1)

                        # x_offxet = 0.5 # ((point*1.0/self.dataset_cfg['width_ratio']) - x_index) * self.dataset_cfg["width_ratio"]
                        # # # print(f"[Debug]: X Pt >>> {x_pt}")
                        # # # print(f"[Debug]: X Offset >>> {x_offxet}")
                        # # # print(f"[Debug]: Offseted x >>> {x_pt+x_offxet}")
                        # y_offset = 0 # ((target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg["height_ratio"]) - y_index) * self.dataset_cfg["height_ratio"]
                        # cv2.circle(viz_img, (int(x_pt+x_offxet), int(y_pt+y_offset)), 2, (0, 0, 255), -1)
                        # cv2.putText(viz_img, str((round(x_offxet, 2))), (int(x_pt+x_offxet), int(y_pt+y_offset)), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
                        # viz_img = cv2.resize(viz_img, (1920, 768))
                        # cv2.imshow("Instance Viz", viz_img)
                        # key = cv2.waitKey(0)
                        # if key==27:
                        #     cv2.destroyAllWindows()
                        #     break
                        # print(f"[Debug]: Y idx >> {y_index}")
                        ground[batch_index][0][y_index][x_index] = 1.0      # 
                        # print(f"[Debug]: Idx 1_1 Val >> {(point*1.0/self.dataset_cfg['width_ratio']) - x_index}")
                        # print(f"[Debug]: Idx 1_2 Val >> {(point*1.0/self.dataset_cfg['width_ratio'])/64}")
                        ground[batch_index][1][y_index][x_index] = (point*1.0/self.dataset_cfg['width_ratio']) - x_index
                        # ground[batch_index][1][y_index][x_index]= (point*1.0/self.dataset_cfg['width_ratio'])/64
                        # print(f"[Debug]: Idx 2 Val >> {(target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg['height_ratio']) - y_index}")
                        ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg["height_ratio"]) - y_index
                        # ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg["height_ratio"])/32
                        ground_binary[batch_index][0][y_index][x_index] = 1

        return ground, ground_binary
    

    ###################################################
    ## Make Ground Truth Lane Width
    ###################################################
    def make_ground_truth_lane_width(self, inputs, target_lanes, target_h):

        

        target_lanes, target_h = util.sort_batch_along_y(target_lanes, target_h)
        # util.visualize_points(inputs[0], target_lanes[0], target_h[0], "RawGT_Pts")
        
        ground_width = np.zeros((len(target_lanes), 4, self.grid_y, self.grid_x))
        g_width_binary = np.zeros((len(target_lanes), 1, self.grid_y, self.grid_x))

        for batch_index, batch in enumerate(target_lanes):  # All Lanes - X pts
            image =  np.rollaxis(inputs[batch_index], axis=2, start=0)
            image =  np.rollaxis(image, axis=2, start=0)*255.0
            # print(f"[Debug]: Image Shape >> {image.shape}")
            viz_img = image.astype(np.uint8).copy()

            for i in range(64):
                cv2.line(viz_img, (i*30, 0), (i*30, 32*24), (255, 0, 0), 1)
            for j in range(32):
                cv2.line(viz_img, (0, j*24), (64*30, j*24), (255, 0, 0), 1)
                    

            for lane_index, lane in enumerate(batch):       # Single Lane's X pts
                idx = len(lane)-1
                for point_index, point in enumerate(lane):  # X pt
                    # print(f"[Debug]: Point >>> {point}")
                    if point > 0:
                        x_index = int(point/self.dataset_cfg["width_ratio"])
                        # print(f"[Debug]: X idx >> {x_index}")
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.dataset_cfg["height_ratio"])

                        x_pt = int(point)
                        y_pt = int(target_h[batch_index][lane_index][point_index])
                        
                        # cv2.line(viz_img, (x_pt, y_pt), (x_pt_n, y_pt_n), (0, 255, 0), 2)
                        # cv2.circle(viz_img, (int(x_pt), int(y_pt)), 3, (0, 255, 0), -1)

                        # x_offxet = 1.5 # ((point*1.0/self.dataset_cfg['width_ratio']) - x_index) * self.dataset_cfg["width_ratio"]
                        # print(f"[Debug]: X Pt >>> {x_pt}")
                        # print(f"[Debug]: X Offset >>> {x_offxet}")
                        # print(f"[Debug]: Offseted x >>> {x_pt+x_offxet}")
                        # y_offset = 1.5 # ((target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg["height_ratio"]) - y_index) * self.dataset_cfg["height_ratio"]
                        x_offset = 0.01 #(point*1.0/self.dataset_cfg['width_ratio']) - x_index
                        y_offset = (target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg["height_ratio"]) - y_index
                        # cv2.circle(viz_img, (int(x_pt-x_offset), int(y_pt)), 2, (255, 0, 0), -1)
                        
                        # if point_index == len(lane)-1:
                        #     p_index = point_index-1
                        # else:
                        #     p_index = point_index

                        # x_pt_d = point*1.0
                        # y_pt = target_h[batch_index][lane_index][point_index]

                        # x_pt_n = target_lanes[batch_index][lane_index][p_index+1]
                        # y_pt_n = target_h[batch_index][lane_index][p_index+1]
                        # cv2.line(viz_img, (int(x_pt_d-5), int(y_pt)), (int(x_pt_n-5), int(y_pt_n)), (255, 0, 0), 2)
                        # cv2.line(viz_img, (int(x_pt), int(y_pt)), (int(x_pt_n), int(y_pt_n)), (0, 255, 0), 2)
                        # cv2.line(viz_img, (int(x_pt_d+5), int(y_pt)), (int(x_pt_n+5), int(y_pt_n)), (0, 0, 255), 2)
                        # # cv2.putText(viz_img, str((round(x_offxet, 2))), (int(x_pt+x_offxet), int(y_pt+y_offset)), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
                        # viz_img = cv2.resize(viz_img, (1920, 768))
                        # cv2.imshow("IOU LaneWidth Viz", viz_img)
                        # key = cv2.waitKey(0)
                        # if key==27:
                        #     cv2.destroyAllWindows()
                        #     break
                        
                        # print(f"[Debug]: Y idx >> {y_index}")
                        ground_width[batch_index][0][y_index][x_index] = 1.0      # 
                        # print(f"[Debug]: Idx 1_1 Val >> {(point*1.0/self.dataset_cfg['width_ratio']) - x_index}")
                        # print(f"[Debug]: Idx 1_2 Val >> {(point*1.0/self.dataset_cfg['width_ratio'])/64}")
                        # ground[batch_index][1][y_index][x_index] = (point*1.0/self.dataset_cfg['width_ratio']) - x_index
                        # ground[batch_index][1][y_index][x_index]= (point*1.0/self.dataset_cfg['width_ratio'])/64
                        # print(f"[Debug]: Idx 2 Val >> {(target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg['height_ratio']) - y_index}")
                        # ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg["height_ratio"]) - y_index
                        # ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][point_index]*1.0/self.dataset_cfg["height_ratio"])/32

                        # Target X-1
                        # lane_width = 0.01
                        # ground_width[batch_index][1][y_index][x_index] = ((point*1.0/self.dataset_cfg['width_ratio']) - x_index) - LOSS_CFG["lane_width"]
                        # # Target X-2
                        # ground_width[batch_index][2][y_index][x_index] = ((point*1.0/self.dataset_cfg['width_ratio']) - x_index) + LOSS_CFG["lane_width"]
                        # # Target X-mid
                        # ground_width[batch_index][3][y_index][x_index] = (point*1.0/self.dataset_cfg['width_ratio']) - x_index


                        ground_width[batch_index][1][y_index][x_index] = (point*1.0/self.dataset_cfg['width_ratio']) - LOSS_CFG["lane_width"]
                        # Target X-2
                        ground_width[batch_index][2][y_index][x_index] = (point*1.0/self.dataset_cfg['width_ratio']) + LOSS_CFG["lane_width"]
                        # Target X-mid
                        ground_width[batch_index][3][y_index][x_index] = int(point*1.0/self.dataset_cfg['width_ratio'])

                        g_width_binary[batch_index][0][y_index][x_index] = 1

        return ground_width, g_width_binary


    #####################################################
    ## Make ground truth for instance feature
    #####################################################
    def make_ground_truth_instance(self, inputs, target_lanes, target_h):

        ground = np.zeros((len(target_lanes), 1, self.grid_y*self.grid_x, self.grid_y*self.grid_x))

        ground_type = np.zeros((len(target_lanes), 1, self.grid_y*self.grid_x, self.grid_y*self.grid_x))

        # viz_image = cv2.resize(image, (self.grid_x, self.grid_y))
        # print(f"[Debug]: Target Lane Length >> {len(target_lanes[0])} ")
        for batch_index, batch in enumerate(target_lanes):  # All Target Lanes

            image =  np.rollaxis(inputs[batch_index], axis=2, start=0)
            image =  np.rollaxis(image, axis=2, start=0)*255.0
            # print(f"[Debug]: Image Shape >> {image.shape}")
            viz_img = image.astype(np.uint8).copy()

            # for i in range(64):
            #     cv2.line(viz_img, (i*30, 0), (i*30, 32*24), (255, 0, 0), 1)
            # for j in range(32):
            #     cv2.line(viz_img, (0, j*24), (64*30, j*24), (255, 0, 0), 1)

            temp = np.zeros((1, self.grid_y, self.grid_x))
            type_temp = np.zeros((1, self.grid_y, self.grid_x))

            lane_cluster = 1
            for lane_index, lane in enumerate(batch):       # Single Lane's X pts

                x_lanes = [lane for lane in target_lanes[batch_index][lane_index] if lane > 0]
                # print(f"[Debug]: Single Lane X >>> {x_lanes}")
                y_lanes = [lane for lane in target_h[batch_index][lane_index] if lane > 0]
                # print(f"[Debug]: Single Lane Y >>> {y_lanes}")
                min_x, max_x = min(x_lanes), max(x_lanes)
                min_y, max_y = min(y_lanes), max(y_lanes)

                cv2.circle(viz_img, (int(min_x), int(min_y)), 5, (255, 0, 0), 3)
                cv2.putText(viz_img, f"{(min_x, min_y)}", (int(min_x), int(min_y-30)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                cv2.circle(viz_img, (int(max_x), int(max_y)), 5, (0, 0, 255), 3)
                cv2.putText(viz_img, f"{(max_x, max_y)}", (int(max_x), int(max_y-30)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)

                # print(f"[Debug]: Min_X & Y >> {(min_x, min_y)}")
                # print(f"[Debug]: Max_X & Y >> {max_x, max_y}")

                if (max_y - min_y)>=187:
                    lane_type = 0       # Straight Lane  
                elif (max_y - min_y)<187 and (600 < min_x < 1200):
                    lane_type = 1       # Branch Lane
                else:
                    lane_type = 2     # Curve Lane

                previous_x_index = 0
                previous_y_index = 0
                for point_index, point in enumerate(lane):   # X pt
                    if point > 0:
                        x_index = int(point/self.dataset_cfg["width_ratio"])
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.dataset_cfg["height_ratio"])
                        
                        # x_pt = int(point)
                        # y_pt = int(target_h[batch_index][lane_index][point_index])
                        # cv2.circle(viz_img, (x_index*30, y_index*24), 2, (0, 0, 255), -1)
                        # cv2.putText(viz_img, str(lane_cluster), (x_index*30, y_index*24), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
                        # viz_img = cv2.resize(viz_img, (1920, 768))
                        # cv2.imshow("Instance Viz", viz_img)
                        # key = cv2.waitKey(0)
                        # if key==27:
                        #     cv2.destroyAllWindows()
                        #     break

                        temp[0][y_index][x_index] = lane_cluster
                        type_temp[0][y_index][x_index] = lane_type

                    if previous_x_index != 0 or previous_y_index != 0: #interpolation make more dense data
                        temp_x = previous_x_index
                        temp_y = previous_y_index
                        while False:      ###############################################false
                            delta_x = 0
                            delta_y = 0
                            temp[0][temp_y][temp_x] = lane_cluster
                            if temp_x < x_index:
                                temp[0][temp_y][temp_x+1] = lane_cluster
                                # cv2.circle(viz_img, ((temp_x+1)*30, temp_y*24), 2, (0, 255, 0), -1)
                                # cv2.putText(viz_img, str(lane_cluster), ((temp_x+1)*30, temp_y*24), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
                                delta_x = 1
                            elif temp_x > x_index:
                                temp[0][temp_y][temp_x-1] = lane_cluster
                                # cv2.circle(viz_img, ((temp_x-1)*30, temp_y*24), 2, (0, 255, 0), -1)
                                # cv2.putText(viz_img, str(lane_cluster), ((temp_x-1)*30, temp_y*24), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
                                delta_x = -1
                            if temp_y < y_index:
                                temp[0][temp_y+1][temp_x] = lane_cluster
                                # cv2.circle(viz_img, (temp_x*30, (temp_y+1)*24), 2, (0, 255, 0), -1)
                                # cv2.putText(viz_img, str(lane_cluster), (temp_x*30, (temp_y+1)*24), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
                                delta_y = 1
                            elif temp_y > y_index:
                                temp[0][temp_y-1][temp_x] = lane_cluster
                                # cv2.circle(viz_img, (temp_x*30, (temp_y-1)*24), 2, (0, 255, 0), -1)
                                # cv2.putText(viz_img, str(lane_cluster), (temp_x*30, (temp_y-1)*24), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
                                delta_y = -1
                            temp_x += delta_x
                            temp_y += delta_y
                            if temp_x == x_index and temp_y == y_index:
                                break
                    if point > 0:
                        previous_x_index = x_index
                        previous_y_index = y_index
                lane_cluster += 1
            

            for i in range(self.grid_y*self.grid_x): #make gt
                # print(f"[Debug]: Temp >>> {len(temp)}")
                temp = temp[temp>-1]   # Filter Out -1 from Arr
                # print(f"[Debug]: Temp Unique Filter >>> {np.unique(temp)}")
                # print(f"[Debub]: Temp 2 >>> {temp}")
                gt_one = deepcopy(temp)
                gt_type = deepcopy(type_temp)
                if temp[i]>0:
                    gt_one[temp==temp[i]] = 1   #same instance  ## Convert the i-th / same instance to 1
                    # print(f"[Debug]: GT One >>> {gt_one}")
                    if temp[i] == 0:    
                        # print(f"[Debug]: Happened 1..... ")         # There is no class & no instance
                        gt_one[temp!=temp[i]] = 3 #different instance, different class / Convert to val 3 if the value isn't 0
                    else:
                        # print(f"[Debug]: Happened 2..... ") 
                        gt_one[temp!=temp[i]] = 2 #different instance, same class / convert >0 vals to 2
                        gt_one[temp==0] = 3 #different instance, different class  / convert ==0 vals to 3

                    # print(f"[Debug]: GT One >>> {gt_one}")
                    # print(f"[Debug]: GT One Shape >>> {gt_one.shape}")

                    ground[batch_index][0][i] += gt_one
                    # print(f"[Debug]: Gt One >>> {gt_one}")

                    # print(f"[Debug]: Gt Type >>> {np.unique(gt_type.flatten())}")
                    ground_type[batch_index][0][i] += gt_type.flatten()
                    

                # if type_temp[i] > 0:
                #     gt_type[temp]

                # print(f"[Debug] Final GT_One Unique >>> {np.unique(gt_one)}")

            # print(f"Ground Type >>> {gt_type}")


        return ground, ground_type

    #####################################################
    ## train 
    #####################################################
    def train(self, inputs, target_lanes, target_h, epoch, agent, data_list):
        point_loss, offset_loss, sisc_loss, disc_loss, exist_condidence_loss, nonexist_confidence_loss, attention_loss, iou_loss = self.train_point(inputs, target_lanes, target_h, epoch, data_list)
        return point_loss, offset_loss, sisc_loss, disc_loss, exist_condidence_loss, nonexist_confidence_loss, attention_loss, iou_loss

    #####################################################
    ## compute loss function and optimize
    #####################################################
    def train_point(self, inputs, target_lanes, target_h, epoch, data_list):
        real_batch_size = len(target_lanes)

        #generate ground truth
        ground_truth_point, ground_binary = self.make_ground_truth_point(inputs, target_lanes, target_h)

        ground_truth_width, g_width_binary = self.make_ground_truth_lane_width(inputs, target_lanes, target_h)
        # print(f"[Debug]: Ground Truth Point >>> {ground_truth_point}")
        # print(f"[Debug]: Ground Binary >>> {ground_binary}")
        ground_truth_instance, g_instance_type = self.make_ground_truth_instance(inputs, target_lanes, target_h)
        #util.visualize_gt(ground_truth_point[0], ground_truth_instance[0], 0, inputs[0])

        # convert GT Points numpy array to torch tensor
        ground_truth_point = torch.from_numpy(ground_truth_point).float()
        ground_truth_point = Variable(ground_truth_point).cuda()
        ground_truth_point.requires_grad=False

        ground_binary = torch.LongTensor(ground_binary.tolist()).cuda()
        ground_binary.requires_grad=False

        # convert GT Width to torch tensor
        ground_truth_width = torch.from_numpy(ground_truth_width).float()
        ground_truth_width = Variable(ground_truth_width).cuda()
        ground_truth_width.requires_grad=False

        g_width_binary = torch.LongTensor(g_width_binary.tolist()).cuda()
        g_width_binary.requires_grad=False

        # convert GT instance to torch tensor
        ground_truth_instance = torch.from_numpy(ground_truth_instance).float()
        ground_truth_instance = Variable(ground_truth_instance).cuda()
        ground_truth_instance.requires_grad=False

        g_instance_type = torch.from_numpy(g_instance_type).float()
        g_instance_type = Variable(g_instance_type).cuda()
        g_instance_type.requires_grad=False
        # print(f"[Debug]: Ground Truth Instance >> {ground_truth_instance.shape}")

        #util.visualize_gt(ground_truth_point[0], ground_truth_instance[0], inputs[0])

        # update lane_detection_network
        # print(f"Input Type >>>> {type(inputs[0])}")
        result, attentions = self.predict_lanes(inputs)
        lane_detection_loss = 0
        exist_condidence_loss = 0
        nonexist_confidence_loss = 0
        offset_loss = 0
        x_offset_loss = 0
        y_offset_loss = 0
        iou_loss = 0
        sisc_loss = 0
        straight_disc_loss = 0
        branch_disc_loss = 0
        curve_disc_loss = 0
        disc_loss = 0
        
        # hard sampling ##################################################################
        print(f"Result Length >>>> {len(result)}")

        confidance, offset, feature = result[-1]
        hard_loss = 0
        # print(f"Confidence Shape >>> {confidance.shape}")
        for i in range(real_batch_size):
            confidance_gt = ground_truth_point[i, 0, :, :]
            confidance_gt = confidance_gt.view(1, self.grid_y, self.grid_x)
            # print(f"Confidece Gt SHape >> {confidance_gt.shape}")
            hard_loss =  hard_loss +\
                torch.sum( (1-confidance[i][confidance_gt==1])**2 )/\
                (torch.sum(confidance_gt==1)+1)

            target = confidance[i][confidance_gt==0]
            hard_loss =  hard_loss +\
				torch.sum( ( target[target>0.01] )**2 )/\
				(torch.sum(target>0.01)+1)

            node = hard_sampling.sampling_node(loss = hard_loss.cpu().data, data = data_list[i], previous_node = None, next_node = None)
            self.hard_sampling.insert(node)

        # compute loss for point prediction #############################################
        for (confidance, offset, feature) in result:
            #compute loss for point prediction

            #exist confidance loss##########################
            confidance_gt = ground_truth_point[:, 0, :, :]      # gt-val is 1
            print(f"[Debug]: Confidence GT Shape >>> {confidance_gt.shape}")
            confidance_gt = confidance_gt.view(real_batch_size, 1, self.grid_y, self.grid_x)
            a = confidance_gt[0][confidance_gt[0]==1] - confidance[0][confidance_gt[0]==1]
            exist_condidence_loss =  exist_condidence_loss +\
				torch.sum( (1-confidance[confidance_gt==1])**2 )/\
				(torch.sum(confidance_gt==1)+1)

            #non exist confidance loss##########################
            target = confidance[confidance_gt==0]
            nonexist_confidence_loss =  nonexist_confidence_loss +\
				torch.sum( ( target[target>0.01] )**2 )/\
				(torch.sum(target>0.01)+1)

            #offset loss ##################################
            offset_x_gt = ground_truth_point[:, 1:2, :, :]
            offset_y_gt = ground_truth_point[:, 2:3, :, :]

            print(f"[Debug]: Pred Offset Shape >>> {offset.shape}")
            predict_x = offset[:, 0:1, :, :]
            predict_y = offset[:, 1:2, :, :]
            # print(f"Confidence GT >> {confidance_gt}")
            # print(f"Offset X GT >> {len(offset_x_gt)}")
            # print(f"Offset X GT Ch >> {offset_x_gt[confidance_gt==1]}")
            # print(f"Pred X >> {len(predict_x)}")
            # print(f"Pred X GT >> {predict_x[confidance_gt==1]}")
            # print(f"Pred Y GT >> {predict_y[confidance_gt==1]}")

            offset_loss = offset_loss + \
			            torch.sum( (offset_x_gt[confidance_gt==1] - predict_x[confidance_gt==1])**2 )/\
				        (torch.sum(confidance_gt==1)+1) + \
			            torch.sum( (offset_y_gt[confidance_gt==1] - predict_y[confidance_gt==1])**2 )/\
				        (torch.sum(confidance_gt==1)+1)                                                     # Regression/Offset Loss for (x, y)

            ### LaneIOU Loss ###
            ### Option 1 ###
            # confidence_lw_gt = ground_truth_width[:, 0, :, :]
            # confidence_lw_gt = confidence_lw_gt.view(real_batch_size, 1, self.grid_y, self.grid_x)
            # # print(f"[Debug]: Confidence LW GT >>> {ground_truth_width}")
            # x1_gt = ground_truth_width[:, 1:2, :, :][confidence_lw_gt==1]
            # # print(f"[Debug]: X1 GT >>> {x1_gt[confidence_lw_gt==1]}")
            # x2_gt = ground_truth_width[:, 2:3, :, :][confidence_lw_gt==1]
            # mid_gt = ground_truth_width[:, 3:4, :, :][confidence_lw_gt==1]
            
            # pred_x1 = predict_x[confidence_lw_gt==1] - LOSS_CFG["lane_width"]
            # pred_x2 = predict_x[confidence_lw_gt==1] + LOSS_CFG["lane_width"]

            ### Option 2 ###
            confidence_lw_gt = ground_truth_width[:, 0, :, :]
            confidence_lw_gt = confidence_lw_gt.view(real_batch_size, 1, self.grid_y, self.grid_x)

            x1_gt = ground_truth_width[:, 1:2, :, :][confidence_lw_gt==1]
            x2_gt = ground_truth_width[:, 2:3, :, :][confidence_lw_gt==1]

            pred_x1 = (ground_truth_width[:, 3:4, :, :][confidence_lw_gt==1] + predict_x[confidence_lw_gt==1]) - LOSS_CFG["lane_width"]
            pred_x2 = (ground_truth_width[:, 3:4, :, :][confidence_lw_gt==1] + predict_x[confidence_lw_gt==1]) + LOSS_CFG["lane_width"]

            mid_gt = (ground_truth_width[:, 2:3, :, :][confidence_lw_gt==1]) - LOSS_CFG["lane_width"]

            invalid_mask = mid_gt
            ovr = torch.min(pred_x2, x2_gt) - torch.max(pred_x1, x1_gt)
            union = torch.max(pred_x2, x2_gt) - torch.min(pred_x1, x1_gt)

            invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
            # print(f"[Debug]: Invalid Masks >> {invalid_mask}")
            ovr[invalid_masks] = 0.0
            union[invalid_masks] = 0.0
            iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
            iou_loss = iou_loss + (1-iou).mean()

            #compute loss for similarity ###############
            # with autocast():
            print(f"[Debug]: Feature >>> {feature.shape}")
            # break
            feature_map = feature.view(real_batch_size, self.trainer_cfg["feature_size"], 1, self.grid_y*self.grid_x)
            feature_map = feature_map.expand(real_batch_size, self.trainer_cfg["feature_size"], self.grid_y*self.grid_x, self.grid_y*self.grid_x)#.detach()

            point_feature = feature.view(real_batch_size, self.trainer_cfg["feature_size"], self.grid_y*self.grid_x, 1)
            point_feature = point_feature.expand(real_batch_size, self.trainer_cfg["feature_size"], self.grid_y*self.grid_x, self.grid_y*self.grid_x)#.detach()
            # print(f"Feature Map Shape >>> {feature_map.shape}")
            # print(f"Point Feature Shape >>> {point_feature.shape}")
            distance_map = (feature_map-point_feature)**2
            # print(f"[Debug]: Distance Map >>> {distance_map.shape}")
            distance_map = torch.sum( distance_map, dim=1 ).view(real_batch_size, 1, self.grid_y*self.grid_x, self.grid_y*self.grid_x)
            # print(f"[Debug]: Distance Map 2 >>> {distance_map[ground_truth_instance==1]}")
            
            # same instance
            sisc_loss = sisc_loss+\
				torch.sum(distance_map[ground_truth_instance==1])/\
				torch.sum(ground_truth_instance==1)

            # different instance, same class
            # straight_disc
            # print(f"DistanceMap 1 >>> {distance_map[ground_truth_instance==2].shape}")
            # print(f"[Debug]: GT instance Shape >>> {ground_truth_instance.shape}")
            # print(f"[Debug]: GT instance type shape >>> {g_instance_type.shape}")
            # straight_instance_mask = g_instance_type == 0  # mask_arr with True value
            # straight_distance_map = distance_map
            # straight_distance_map[~straight_instance_mask] = 3  # leave mask's True index value & convert False value to 3
            # print(f"[Debug]: Straight Dist Map >>> {straight_distance_map[ground_truth_instance==2]}")
            # straight_disc_loss = straight_disc_loss + \
			# 	torch.sum((LOSS_CFG['K_S']-straight_distance_map[ground_truth_instance==2])[(LOSS_CFG['K_S']-straight_distance_map[ground_truth_instance==2]) > 0])/\
			# 	torch.sum(ground_truth_instance==2)

            # # branch_discbranch_distance_map
            
            # branch_instance_mask = g_instance_type == 1  # mask_arr with True value
            # branch_distance_map = distance_map
            # branch_distance_map[~branch_instance_mask] = 5e-11  # leave mask's True index value & convert False value to 3
            # print(f"[Debug]: Branch Dist Map >>> {branch_distance_map[ground_truth_instance==2]}")
            # branch_disc_loss = branch_disc_loss + \
			# 	torch.sum((LOSS_CFG['K_B']-branch_distance_map[ground_truth_instance==2])[(LOSS_CFG['K_B']-branch_distance_map[ground_truth_instance==2]) > 0])/\
			# 	torch.sum(ground_truth_instance==2) * LOSS_CFG["constant_branch_disc"]

            # # curve_disc
            # curve_instance_mask = g_instance_type == 1  # mask_arr with True value
            # curve_distance_map = distance_map
            # curve_distance_map[~curve_instance_mask] = 0.09  # leave mask's True index value & convert False value to 3
            # curve_disc_loss = curve_disc_loss + \
			# 	torch.sum((LOSS_CFG['K_C']-curve_distance_map[ground_truth_instance==2])[(LOSS_CFG['K_C']-curve_distance_map[ground_truth_instance==2]) > 0])/\
			# 	torch.sum(ground_truth_instance==2)
            

            # disc_loss = disc_loss + straight_disc_loss + branch_disc_loss + curve_disc_loss
            disc_loss = disc_loss + \
				torch.sum((self.p.K1-distance_map[ground_truth_instance==2])[(self.p.K1-distance_map[ground_truth_instance==2]) > 0])/\
				torch.sum(ground_truth_instance==2)

        #attention loss
        attention_loss = 0
        source = attentions[:-1]
        m = nn.Softmax(dim=0)
        
        for i in range(real_batch_size):
            target = torch.sum((attentions[-1][i].data)**2, dim=0).view(-1) 
            #target = target/torch.max(target)
            # print(len(target))
            target = m(target)
            for j in source:
                s = torch.sum(j[i]**2, dim=0).view(-1)
                attention_loss = attention_loss + torch.sum( (m(s) - target)**2 )/(len(target)*real_batch_size)

        lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_exist"]*exist_condidence_loss
        lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_nonexist"]*nonexist_confidence_loss
        lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_offset"]*offset_loss
        lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_alpha"]*sisc_loss
        # lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_beta"]*straight_disc_loss + 0.0001*torch.sum(feature**2)
        # lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_beta"]*branch_disc_loss + 0.000001*torch.sum(feature**2)
        # lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_beta"]*curve_disc_loss + 0.00001*torch.sum(feature**2)
        lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_beta"]*disc_loss + 0.00001*torch.sum(feature**2)

        lane_detection_loss = lane_detection_loss + LOSS_CFG["constant_attention"]*attention_loss
        lane_detection_loss = lane_detection_loss + LOSS_CFG["iou_loss_weight"]*iou_loss

        ### For return
        offset_loss_r = offset_loss
        sisc_loss_r = sisc_loss
        disc_loss_r = disc_loss
        # s_disc_loss_r = straight_disc_loss
        # b_disc_loss_r = branch_disc_loss
        # c_disc_loss_r = curve_disc_loss
        exist_condidence_loss_r = exist_condidence_loss
        nonexist_confidence_loss_r = nonexist_confidence_loss
        attention_loss_r = attention_loss
        iou_loss_r = iou_loss

        print("######################################################################")
        print(f"Epoch >>> {str(epoch+1)}")
        print("seg loss")
        print("same instance loss: ", sisc_loss.data)
        # print("different straight instance loss: ", straight_disc_loss.data)
        # print("different branch instance loss: ", branch_disc_loss.data)
        # print("different curve instance loss: ", curve_disc_loss.data)
        print("total different instance loss: ", disc_loss.data)

        print("point loss")
        print("exist loss: ", exist_condidence_loss.data)
        print("non-exit loss: ", nonexist_confidence_loss.data)
        print("offset loss: ", offset_loss.data)

        print("attention loss")
        print("attention loss: ", attention_loss)

        print("IOU Loss")
        print("IOU Loss: ", iou_loss)
        
        print("--------------------------------------------------------------------")
        print("total loss: ", lane_detection_loss.data)

        self.lane_detection_optim.zero_grad()
        lane_detection_loss.backward()   #divide by batch size
        self.lane_detection_optim.step()

        del confidance, offset, feature
        del ground_truth_point, ground_binary, ground_truth_instance
        del feature_map, point_feature, distance_map
        del exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, iou_loss

        trim = 180
        if epoch>0 and self.current_epoch != epoch:
            self.current_epoch = epoch
            if epoch == 1-trim:
                self.l_rate = 0.0005
                self.setup_optimizer()
            elif epoch == 2-trim:
                self.l_rate = 0.0002
                self.setup_optimizer()
            elif epoch == 3-trim:
                self.l_rate = 0.0001
                self.setup_optimizer()
            elif epoch == 5-trim:
                self.l_rate = 0.00005
                self.setup_optimizer()
            elif epoch == 7-trim:
                self.l_rate = 0.00002
                self.setup_optimizer()
            elif epoch == 9-trim:
                self.l_rate = 0.00001
                self.setup_optimizer()
            elif epoch == 11-trim:
                self.l_rate = 0.000005
                self.setup_optimizer()
            elif epoch == 13-trim:
                self.l_rate = 0.000002
                self.setup_optimizer()
            elif epoch == 15-trim:
                self.l_rate = 0.000001
                self.setup_optimizer()
            elif epoch == 21-trim:  
                self.l_rate = 0.0000001
                self.setup_optimizer()

        return lane_detection_loss, offset_loss_r, sisc_loss_r, disc_loss_r, exist_condidence_loss_r, nonexist_confidence_loss_r, attention_loss_r, iou_loss_r

    #####################################################
    ## predict lanes
    #####################################################
    def predict_lanes(self, inputs):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()

        return self.lane_detection_network(inputs)

    #####################################################
    ## predict lanes in test
    #####################################################
    def predict_lanes_test(self, inputs):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()
        outputs, features = self.lane_detection_network(inputs)

        return outputs

    #####################################################
    ## Training mode
    #####################################################                                                
    def training_mode(self):
        self.lane_detection_network.train()

    #####################################################
    ## evaluate(test mode)
    #####################################################                                                
    def evaluate_mode(self):
        self.lane_detection_network.eval()

    #####################################################
    ## Setup GPU computation
    #####################################################                                                
    def cuda(self):
        GPU_NUM = 0
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        self.lane_detection_network.cuda()

    #####################################################
    ## Load save file
    #####################################################
    def load_weights(self, weight_file):
        self.lane_detection_network.load_state_dict(
            torch.load(weight_file, map_location='cuda:0'),False
        )

    #####################################################
    ## Save model
    #####################################################
    def save_model(self, epoch, loss):
        # torch.save(
        #     self.lane_detection_network.state_dict(),
        #     self.p.save_path+str(epoch)+'_'+str(loss)+'_'+'lane_detection_network.pkl'
        # )
        model_dest = os.path.join(self.trainer_cfg['save_path'], f'{str(epoch)}_{str(loss)}_lane_detection_network.pth')
        torch.save(self.lane_detection_network.state_dict(), model_dest)

    def get_data_list(self):
        return self.hard_sampling.get_list()

    def sample_reset(self):
        self.hard_sampling = hard_sampling.hard_sampling()
