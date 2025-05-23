import os 
import torch
import numpy as np
from scipy.io import savemat, loadmat
from yacs.config import CfgNode as CN
from scipy.signal import savgol_filter

import safetensors
import safetensors.torch 

from src.audio2pose_models.audio2pose import Audio2Pose
from src.audio2exp_models.networks import SimpleWrapperV2 
from src.audio2exp_models.audio2exp import Audio2Exp
from src.utils.safetensor_helper import load_x_from_safetensor  

def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']

class Audio2Coeff():

    def __init__(self, sadtalker_path, device):
        # Load configuration files
        fcfg_pose = open(sadtalker_path['audio2pose_yaml_path'])
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()
        fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
        cfg_exp = CN.load_cfg(fcfg_exp)
        cfg_exp.freeze()

        # Load audio2pose model
        self.audio2pose_model = Audio2Pose(cfg_pose, None, device=device)
        self.audio2pose_model = self.audio2pose_model.to(device)
        self.audio2pose_model.eval()
        for param in self.audio2pose_model.parameters():
            param.requires_grad = False 
        
        try:
            if sadtalker_path['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                self.audio2pose_model.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2pose'))
            else:
                load_cpk(sadtalker_path['audio2pose_checkpoint'], model=self.audio2pose_model, device=device)
        except:
            raise Exception("Failed in loading audio2pose_checkpoint")

        # Load audio2exp model
        netG = SimpleWrapperV2()
        netG = netG.to(device)
        for param in netG.parameters():
            netG.requires_grad = False
        netG.eval()
        try:
            if sadtalker_path['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            else:
                load_cpk(sadtalker_path['audio2exp_checkpoint'], model=netG, device=device)
        except:
            raise Exception("Failed in loading audio2exp_checkpoint")
        
        self.audio2exp_model = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
        self.audio2exp_model = self.audio2exp_model.to(device)
        for param in self.audio2exp_model.parameters():
            param.requires_grad = False
        self.audio2exp_model.eval()

        self.device = device

    def generate(self, batch, coeff_save_dir, pose_style, ref_pose_coeff_path=None):
        with torch.no_grad():
            # Expression prediction
            results_dict_exp = self.audio2exp_model.test(batch)
            exp_pred = results_dict_exp['exp_coeff_pred']  # bs T 64

            # Pose prediction
            batch['class'] = torch.LongTensor([pose_style]).to(self.device)
            results_dict_pose = self.audio2pose_model.test(batch)
            pose_pred = results_dict_pose['pose_pred']  # bs T 6

            # Smooth pose
            pose_len = pose_pred.shape[1]
            if pose_len < 13:
                pose_len = int((pose_len - 1) / 2) * 2 + 1
            pose_pred = torch.Tensor(
                savgol_filter(pose_pred.cpu().numpy(), pose_len, 2, axis=1)
            ).to(self.device)

            # Merge expression and pose coefficients
            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)  # bs T 70
            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy()

            # Apply reference pose if available
            if ref_pose_coeff_path is not None:
                coeffs_pred_numpy = self.using_refpose(coeffs_pred_numpy, ref_pose_coeff_path)

            # Construct file paths for both versions
            pic_name = batch['pic_name']
            audio_name = batch['audio_name']
            file_with_audio = f"{pic_name}##{audio_name}.mat"
            file_simple = f"{pic_name}.mat"

            path_with_audio = os.path.join(coeff_save_dir, file_with_audio)
            path_simple = os.path.join(coeff_save_dir, file_simple)

            # Save both versions for flexibility
            savemat(path_with_audio, {'coeff_3dmm': coeffs_pred_numpy})
            savemat(path_simple, {'coeff_3dmm': coeffs_pred_numpy})

            # Return the simple path for first_coeff_path compatibility
            return path_simple
    
    def using_refpose(self, coeffs_pred_numpy, ref_pose_coeff_path):
        num_frames = coeffs_pred_numpy.shape[0]
        refpose_coeff_dict = loadmat(ref_pose_coeff_path)
        refpose_coeff = refpose_coeff_dict['coeff_3dmm'][:,64:70]
        refpose_num_frames = refpose_coeff.shape[0]
        if refpose_num_frames < num_frames:
            div = num_frames // refpose_num_frames
            re = num_frames % refpose_num_frames
            refpose_coeff_list = [refpose_coeff for i in range(div)]
            refpose_coeff_list.append(refpose_coeff[:re, :])
            refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)

        # Apply relative head pose adjustment
        coeffs_pred_numpy[:, 64:70] = coeffs_pred_numpy[:, 64:70] + (refpose_coeff[:num_frames, :] - refpose_coeff[0:1, :])
        return coeffs_pred_numpy
