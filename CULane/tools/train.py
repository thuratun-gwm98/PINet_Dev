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
from src.data.parameters import Parameters
from configs.parameters import TRAINER_CFG
import test
import evaluation
from src.data import util
import os
import copy

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
    for epoch in range(train_cfg['n_epoch']):
        lane_model.training_mode()
        for inputs, target_lanes, target_h, test_image, data_list in loader.Generate(sampling_list):
            print(f"Inputss >>> {len(inputs)}")
            print(f"DataList >>> {len(data_list)}")
            util.visualize_points(inputs[0], target_lanes[0], target_h[0])
            #training
            print("epoch : " + str(epoch))
            print("step : " + str(step))
            loss_p = lane_model.train(inputs, target_lanes, target_h, epoch, lane_model, data_list)
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
                testing(lane_model, test_image, step_, loss_p)
            step += 1

        sampling_list = copy.deepcopy(lane_model.get_data_list())
        lane_model.sample_reset()

        #evaluation
        if epoch==0 or (epoch+1)%10==0:
            print("evaluation")
            lane_model.evaluate_mode()
            th_list = [0.9]
            index = [0]
            lane_model.save_model(epoch=epoch+1, loss=loss_p)

            # for idx in index:
            print("generate result")
            test.evaluation(loader, lane_model, name="test_result_"+str(epoch)+"_"+".json")
            name = "epoch_idx_"+str(epoch) + str(step/100)
            os.system("sh /home/kym/research/autonomous_car_vision/lane_detection/code/ITS/CuLane/evaluation_code/SCNN_Pytorch/utils/lane_evaluation/CULane/Run.sh " + name)

        if int(step)>700000:
            break


def testing(lane_model, test_image, step, loss):
    lane_model.evaluate_mode()

    _, _, ti = test.test(lane_model, np.array([test_image]))

    cv2.imwrite('test_result/images/result_'+str(step)+'_'+str(loss)+'.png', ti[0])

    lane_model.training_mode()

    
if __name__ == '__main__':
    Training()