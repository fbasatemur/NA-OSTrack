import torch
import torch.nn.functional as F
import numpy as np
import math

import importlib
from occlusion_handler.features.preprocessing import numpy_to_torch, sample_patch_multiscale
from occlusion_handler.libs import dcf
from occlusion_handler.libs.kalmanfilter import KalmanFilter

class OcclusionHandler():

    def __init__(self):
        param_module = importlib.import_module('occlusion_handler.parameter.{}.{}'.format("dimp", "dimp18"))
        self.params = param_module.parameters()
        

    """ initialize()
        image:      input frame
        init_bbox:  initial target bounding box [x,y,w,h]
    """
    def initialize(self, image: np.ndarray, init_bbox: list) -> None:
        # Initialize some stuff
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = init_bbox
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set sizes
        self.image_sz_list = [image.shape[0], image.shape[1]]       # height, width
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])    
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        ksz = 4
        self.kernel_size = torch.Tensor([ksz, ksz])
        self.feature_sz = torch.Tensor([18, 18])
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        self.prev_flag = None
        self.interpolate_sz = [19,19]
        self.interpolate_mode = 'bicubic'
        self.target_lost_counter = self.target_found_counter = 0
        self.target_action_th = 5
        self.frame_id = 0
        self.search_region_k = 3
        self.use_search_region = True
        # self.KF = KalmanFilter(0.05, 0.5, 0.5, 1, 0.1,0.1)


    """ check() 
        image:          input frame                                     
        prev_state:     t-1. frame sample track state [x,y,w,h]         
        curr_state:     t. frame sample track state [x,y,w,h]           
        response_hn:    Hanning window applied response map (1,1,N,N)   
        ---------------
        output_state:   return state [x,y,w,h]                                     
    """
    def check(self, image: np.ndarray, prev_state: list = None, curr_state: list = None, response_hn: torch.Tensor = None) -> list:

        # Convert image
        im = numpy_to_torch(image)
        self.frame_id +=1

        # ------- LOCALIZATION ------- #

        # prev_state -> x,y,w,h
        if prev_state and self.prev_flag not in [None, 'not_found', 'uncertain']:
            self.pos = torch.Tensor([prev_state[1] + (prev_state[3] - 1)/2, prev_state[0] + (prev_state[2] - 1)/2])
            self.target_sz = torch.Tensor([prev_state[3], prev_state[2]])

        # get multiscale sample coords
        _, sample_coords = sample_patch_multiscale(im, self.get_centered_sample_pos(), self.target_scale * self.params.scale_factors, self.img_sample_sz,
                                                    mode=self.params.get('border_mode', 'replicate'),
                                                    max_scale_change=self.params.get('patch_max_scale_change', None))

        # Location of sample  
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Resize response map (1,1,16,16) -> (1,1,19,19)
        score_map = F.interpolate(response_hn, self.interpolate_sz, mode=self.interpolate_mode)                 

        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(score_map, sample_pos, sample_scales)
        new_pos = sample_pos[scale_ind,:] + translation_vec


        # Update position and scale
        if flag != 'not_found':
            self.update_state(new_pos, sample_scales[scale_ind])

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        output_state = new_state.tolist()
        self.prev_flag = flag


        if self.target_lost():
            output_state = self.state_target_lost(output_state)
        
        else:
            output_state = self.state_target_stable(prev_state, curr_state)

        if self.use_search_region:
            if not self.in_search_region(prev_state, curr_state):
                output_state = prev_state

        return output_state

    def in_search_region(self, prev_state: list, curr_state: list) -> bool:
        """"""
        prev_x, prev_y = prev_state[0], prev_state[1]
        prev_w, prev_h = prev_state[2], prev_state[3]

        ratio_x = prev_w // 2
        ratio_y = prev_h // 2
        margin_x0, margin_y0 = prev_x + ratio_x * (1 - self.search_region_k), prev_y + ratio_y * (1 - self.search_region_k)
        margin_x1, margin_y1 = prev_x + ratio_x * (1 + self.search_region_k), prev_y + ratio_y * (1 + self.search_region_k)

        margin_y0 = 0 if margin_y0 < 0 else margin_y0 
        margin_x0 = 0 if margin_x0 < 0 else margin_x0
        margin_y1 = self.image_sz_list[0] if margin_y1 > self.image_sz_list[0] else margin_y1 
        margin_x1 = self.image_sz_list[1] if margin_x1 > self.image_sz_list[1] else margin_x1

        curr_w, curr_h = curr_state[2], curr_state[3]
        curr_cx, curr_cy = curr_state[0] + curr_w // 2, curr_state[1] + curr_h // 2

        return curr_cx > margin_x0 and curr_cx < margin_x1 and curr_cy > margin_y0 and curr_cy < margin_y1
 
    def target_lost(self) -> bool:
        return self.prev_flag in ['not_found', 'uncertain', "hard_negative"]
    
    def state_target_lost(self, output_state: list):
        """ When the object lost, this section is run. 
            If you want use to Kalman Filter, You can remove comment of KF object line. (line 77)"""

        self.target_found_counter = 0
        self.target_lost_counter += 1
        if self.target_lost_counter > self.target_action_th:
            if hasattr(self, "KF"):
                self.kalman_predict(output_state)
        return output_state
    
    def state_target_stable(self, prev_state: list, curr_state: list):
        """ When the object tracked, this section is run. 
        If you want use to Kalman Filter, You can remove comment of KF object line. (line 77)"""

        self.target_lost_counter = 0
        self.target_found_counter += 1
        if self.target_found_counter > self.target_action_th:
            if hasattr(self, "KF"):
                self.kalman_update(pos=self.pos.cpu().detach().numpy())
            return curr_state
        else:
            return prev_state
    
    def kalman_predict(self, output_state: list):
        """Predict the object localization with Kalman Filter."""
        (predict_x, predict_y) = self.KF.predict()
        output_state[:2] = [float(predict_x), float(predict_y)]

    def kalman_update(self, pos: np.ndarray):
        """Update Kalman Filter to the object localization estimation."""
        (pred_x, pred_y) = self.KF.predict()
        (est_x, est_y) = self.KF.update([[pos[1]], [pos[0]]])

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales
    

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)


    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None


    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'hard_negative'

        return translation_vec1, scale_ind, scores, 'normal'

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)