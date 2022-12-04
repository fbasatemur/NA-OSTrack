from ctypes import resize

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


import numpy as np
import pylab as plt

from occlusion_handler.occlusion_handler import OcclusionHandler
import importlib
from lib.test.utils.display_patch import display

class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        self.occlusion_handler = OcclusionHandler()


    def initialize(self, image, info: dict):
        # initialize dimp module
        self.occlusion_handler.initialize(image, info['init_bbox'])

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        input_state = self.state
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response_map_hn = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(response_map_hn, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        
        # get the final box result => output_state
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        output_state = self.state

        
        self.state = self.occlusion_handler.check(image, prev_state=input_state, curr_state=output_state, response_hn=response_map_hn)


        ####################### DISPLAY STAGE ########################
        # search_mask_cpu = x_amask_arr.astype(np.uint8) * 255

        # ### SHOW RESPONSE [16,16]
        # response_map_dis = response.cpu().detach().numpy()[0][0]
        # size_map_dis = out_dict['size_map'].cpu().detach().numpy()[0][0]
        # offset_map_dis = out_dict['offset_map'].cpu().detach().numpy()[0][0]
        # ###

        # ### SHOW ATTENTION LAYERS[0:12]
        # display(out_dict['attn'][0][11] * 255.0 / torch.max(out_dict['attn'][0][11])).show()
        # ###

        # save_func = lambda : plt.savefig("~/Desktop/OSTrack/debug/plot_figs_dimp/" + ("0" * 7)[:-len(str(self.frame_id))] + str(self.frame_id) + ".jpg")
        # save_func_3d = lambda : plt.savefig("~/Desktop/OSTrack/debug/plot_resp_3d/" + ("0" * 7)[:-len(str(self.frame_id))] + str(self.frame_id) + ".jpg")

        # pred_bbox_i = list(map(int, self.state))
        # crop_roi = image[pred_bbox_i[1]:pred_bbox_i[1] + pred_bbox_i[3], pred_bbox_i[0]:pred_bbox_i[0] + pred_bbox_i[2], :]
        # display(self, title="Frame: " + str(self.frame_id), 
        #             fig_title_dict={
        #                 "Mask":search_mask_cpu, 
        #                 "Response":response_map_dis, 
        #                 "input-patch:":x_patch_arr, 
        #                 "Result:":crop_roi,
        #                 "size_map":size_map_dis,
        #                 "offset_map":offset_map_dis}, 
        #             cmaps = [0, 1] + [0] * 4
        #             # , plt_func = save_func
        #             )
        
        # display_3d(self, response_map_dis, save_func_3d)
        #########################################################

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
