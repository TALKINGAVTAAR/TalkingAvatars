import sys
import os
import shutil
from time import strftime

# Add parent dir to sys.path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports from src
try:
    from src.utils.preprocess import CropAndExtract
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data
    from src.utils.init_path import init_path
except ImportError as e:
    print(f"[Import Error] {e}")
    sys.exit(1)

# Configuration
RESULT_FOLDER = 'results'

# Core pipeline logic
def run_sadtalker(pic_path, audio_path, ref_eyeblink=None, ref_pose=None):
    save_dir = os.path.join(RESULT_FOLDER, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_config = {
        'path_of_net_recon_model': 'checkpoints/epoch_20.pth',
        'dir_of_BFM_fitting': 'checkpoints/BFM_Fitting',
        'checkpoint': 'checkpoints/sadtalker.safetensors',
        'use_safetensor': True
    }

    try:
        preprocess_model = CropAndExtract(checkpoint_config, device="cpu")
        audio_to_coeff = Audio2Coeff(checkpoint_config, device="cpu")
        animate_from_coeff = AnimateFromCoeff(checkpoint_config, device="cpu")
    except Exception as e:
        return None, f"Model loading failed: {str(e)}"

    try:
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)

        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            pic_path, first_frame_dir, 'crop', source_image_flag=True, pic_size=256
        )

        if not first_coeff_path:
            return None, "Failed to process the source image."

        if ref_eyeblink:
            preprocess_model.generate(ref_eyeblink, save_dir, 'crop', source_image_flag=False)
        if ref_pose:
            preprocess_model.generate(ref_pose, save_dir, 'crop', source_image_flag=False)

        batch = get_data(first_coeff_path, audio_path, device="cpu")
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style=0)

        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path, audio_path,
            batch_size=2, input_yaw_list=None, input_pitch_list=None, input_roll_list=None
        )

        result = animate_from_coeff.generate(
            data, save_dir, pic_path, crop_info,
            enhancer=None, background_enhancer=None, preprocess='crop', img_size=256
        )

        video_path = os.path.join(save_dir, 'output_video.mp4')
        shutil.move(result, video_path)

        return video_path.replace("\\", "/"), None

    except Exception as e:
        return None, f"Processing error: {str(e)}"

# Main function to run the script
def main(image_path, audio_path):
    video_path, error = run_sadtalker(image_path, audio_path)

    if error:
        print(f"Error: {error}")
    else:
        print(f"Video successfully generated: {video_path}")

if __name__ == '__main__':
    # Get the file paths from command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Generate video from image and audio using SadTalker model.")
    parser.add_argument('--image', required=True, help="Path to the source image")
    parser.add_argument('--audio', required=True, help="Path to the audio file")
    args = parser.parse_args()

    # Run inference
    main(args.image, args.audio)
