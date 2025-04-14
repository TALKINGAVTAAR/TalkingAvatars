import os
from PIL import Image
import numpy as np
import cv2
import torch
from src.models.reconstruction import load_recon_model
from torchvision import transforms

class CropAndExtract:
    def __init__(self, device='cuda'):
        self.device = device
        # Move the Preprocesser import here to avoid circular import
        from src.utils.preprocess import Preprocesser
        self.preprocessor = Preprocesser(device=device)
        self.net_recon, self.face_info = load_recon_model(device=device)

    def generate(self, source_image_list, save_dir=None, pic_path=None, preprocess='crop', verbose=False, size=512, use_mask=True):
        if isinstance(source_image_list[0], str):
            img_np = [cv2.imread(source_image_list[0])]
        else:
            img_np = source_image_list

        # Step 1: Crop and align the face
        cropped_img_list, crop_info, lm = self.preprocessor.crop(img_np, xsize=size)

        # Step 2: Extract 3DMM parameters
        image = Image.fromarray(cropped_img_list[0])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image_tensor = test_transform(image)[None].to(self.device)
        with torch.no_grad():
            codedict = self.net_recon.module.forward_test(image_tensor)

        # Save if needed
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(cropped_img_list[0]).save(os.path.join(save_dir, 'cropped.png'))
            np.save(os.path.join(save_dir, 'crop_info.npy'), crop_info)
            torch.save(codedict, os.path.join(save_dir, 'params.pt'))

        return cropped_img_list, crop_info, codedict
