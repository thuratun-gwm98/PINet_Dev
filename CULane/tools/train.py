#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import cv2
import torch
import visdom
import os, sys
#sys.path.append('/home/kym/research/autonomous_car_vision/lanedection/code/')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from src.models import model_helper
import numpy as np
from src.data.data_loader import DataGenerator
from src.data.data_parameters import Parameters
from configs.parameters import TRAINER_CFG
from collections import defaultdict
import test
import evaluation
from src.data import util
import os
import json
import copy
torch.cuda.empty_cache()
p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training():
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')
    train_cfg = TRAINER_CFG
    # vis = visdom.Visdom()
    # loss_window = vis.line(X=torch.zeros((1,)).cpu(),
    #                        Y=torch.zeros((1)).cpu(),
    #                        opts=dict(xlabel='epoch',
    #                                  ylabel='Loss',
    #                                  title='Training Loss',
    #                                  legend=['Loss']))
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = DataGenerator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')

    # if train_cfg['model_path'] == "":
    lane_model = model_helper.ModelAgent()
    # else:
    #     lane_model = model_helper.ModelAgent()
    #     lane_model.load_weights(0, "tensor(1.3984)")

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_model.cuda()
        #torch.backends.cudnn.benchmark=True

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    sampling_list = None
    loss_dict = defaultdict(list)
    for epoch in range(train_cfg['n_epoch']):
        lane_model.training_mode()
        count=0
        # print(f"[Debug]: Initial Sampleing List >>> {sampling_list}")
        for inputs, target_lanes, target_h, test_image, data_list in loader.Generate(sampling_list):
            # print(f"[Debug]: Target Lane 2 Length >> {len(target_lanes[0])} ")
            # util.visualize_points(inputs[0], target_lanes[0], target_h[0], "ToTrain")
            #training
            loss_p, offset_loss, sisc_loss, disc_loss, exist_condidence_loss, nonexist_confidence_loss, attention_loss, iou_loss = lane_model.train(inputs, target_lanes, target_h, epoch, lane_model, data_list)
            torch.cuda.synchronize()
            loss_p = loss_p.cpu().data
            
            # if step%50 == 0:
            #     vis.line(
            #         X=torch.ones((1, 1)).cpu() * int(step/50),
            #         Y=torch.Tensor([loss_p]).unsqueeze(0).cpu(),
            #         win=loss_window,
            #         update='append')
                
            if epoch==0 or (epoch+1)%5==0:
                # lane_model.save_model(epoch, loss_p)
                step_ = f"{epoch}_{step}"
                # print(f"Test Image Shape >>> {test_image.shape}")
                testing(lane_model, test_image, epoch, count, offset_loss)

            # testing(lane_model, test_image, epoch, count, offset_loss)

            count+=1
            step += 1
        
        sampling_list = copy.deepcopy(lane_model.get_data_list())
        # print(f"[Debug]: Return Sampling List >>> {sampling_list}")
        # print(f"[Debug]: Return Sampling List Length >>> {len(sampling_list)}")
        lane_model.sample_reset()

        loss_dict["total_loss"].append(loss_p.item())
        loss_dict["offset_loss"].append(offset_loss.item())
        loss_dict["sisc_loss"].append(sisc_loss.item())
        # loss_dict["s_disc_loss"].append(s_disc_loss.item())
        # loss_dict["b_disc_loss"].append(b_disc_loss.item())
        # loss_dict["c_disc_loss"].append(c_disc_loss.item())
        loss_dict["disc_loss"].append(disc_loss.item())
        loss_dict["exist_condidence_loss"].append(exist_condidence_loss.item())
        loss_dict["nonexist_confidence_loss"].append(nonexist_confidence_loss.item())
        loss_dict["attention_loss"].append(attention_loss)
        loss_dict["iou_loss"].append(iou_loss.item())

        #evaluation
        if epoch==0 or (epoch+1)%10==0:
            print("evaluation")
            lane_model.evaluate_mode()
            th_list = [0.9]
            index = [0]
            lane_model.save_model(epoch=epoch+1, loss=loss_p)

            # for idx in index:
            print("generate result")
            test.evaluation(loader, lane_model, epoch=str(epoch))
            name = "epoch_idx_"+str(epoch) + str(step/100)
            os.system("sh /home/kym/research/autonomous_car_vision/lane_detection/code/ITS/CuLane/evaluation_code/SCNN_Pytorch/utils/lane_evaluation/CULane/Run.sh " + name)

        if int(step)>700000:
            break
    
    dump_loss_data(loss_dict)
    print(f"[INFO] Training Completed")

def testing(lane_model, test_image, epoch, count, loss):
    lane_model.evaluate_mode()

    _, _, ti = test.test(lane_model, np.array([test_image]))

    save_path = f"test_result/images/epoch/"
    os.makedirs(save_path, exist_ok=True)

    cv2.putText(ti[0], str(loss), (50, 70), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 255), 1)

    cv2.imwrite(f'{save_path}/result_{str(epoch)}_{str(count)}.jpg', ti[0])

    lane_model.training_mode()

def dump_loss_data(loss_dict):
    json_result = os.path.join(f"test_result/images", "loss_data.json")
    with open(json_result, "w") as json_file:
        json.dump(loss_dict, json_file, indent=4)
    
if __name__ == '__main__':
    Training()