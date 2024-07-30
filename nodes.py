import torch
import os
import numpy as np
import json

from contextlib import nullcontext

from .load_model import load_model

import comfy.model_management as mm
from comfy.utils import ProgressBar, common_upscale
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

class DownloadAndLoadSAM2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ([ 
                    'sam2_hiera_base_plus.safetensors',
                    'sam2_hiera_large.safetensors',
                    'sam2_hiera_small.safetensors',
                    'sam2_hiera_tiny.safetensors',
                    ],),
            "segmentor": (
                    ['single_image','video',],
                    ),
            "device": (['cuda', 'cpu',], ),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'bf16'
                    }),

            },
        }

    RETURN_TYPES = ("SAM2MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "SAM2"

    def loadmodel(self, model, segmentor, device, precision):
        #device = mm.get_torch_device()
        #offload_device = mm.unet_offload_device()
        if device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        download_path = os.path.join(folder_paths.models_dir, "sam2")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            print(f"Downloading SAM2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/sam2-safetensors",
                            allow_patterns=[f"*{model}*"],
                            local_dir=download_path,
                            local_dir_use_symlinks=False)

        model_mapping = {
            "base": "sam2_hiera_b+.yaml",
            "large": "sam2_hiera_l.yaml",
            "small": "sam2_hiera_s.yaml",
            "tiny": "sam2_hiera_t.yaml"
        }

        model_cfg_path = next(
            (os.path.join(script_directory, "sam2_configs", cfg) for key, cfg in model_mapping.items() if key in model),
            None
            )

        model =load_model(model_path, model_cfg_path, segmentor, dtype, device)
        
        sam2_model = {
            'model': model, 
            'dtype': dtype,
            'segmentor' : segmentor
            }

        return (sam2_model,)
    
class Sam2Segmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam2_model": ("SAM2MODEL", ),
                "image": ("IMAGE", ),
                "coordinates": ("STRING", {"forceInput": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MASK", )
    RETURN_NAMES =("mask", )
    FUNCTION = "segment"
    CATEGORY = "SAM2"

    def segment(self, image, sam2_model, coordinates, keep_model_loaded):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = sam2_model["model"]
        dtype = sam2_model["dtype"]
        segmentor = sam2_model["segmentor"]
        B, H, W, C = image.shape

        if segmentor == 'video': # video model needs images resized first thing
            model_input_image_size = model.image_size
            print("Resizing to model input image size: ", model_input_image_size)
            image = common_upscale(image.movedim(-1,1), model_input_image_size, model_input_image_size, "bilinear", "disabled").movedim(1,-1)

        image_np = (image[0].contiguous() * 255).byte().numpy()

        coordinates = json.loads(coordinates.replace("'", '"'))
        coordinates = [(coord['x'], coord['y']) for coord in coordinates]
        point_coords = np.array(coordinates)
        print("coordinates: ", point_coords)
        point_labels = [1] * len(point_coords)  # 1 = foreground, 0 = background,all points are foreground for now
        point_labels = np.array(point_labels)
        #print("point_labels: ", point_labels)
        
        autocast_condition = not mm.is_device_mps(device)
        mask_list = []
        try:
            model.to(device)
        except:
            model.model.to(device)
        with torch.autocast(mm.get_autocast_device(model.device), dtype=dtype) if autocast_condition else nullcontext():
            if image.shape[0] == 1:
                model.set_image(image_np) 
                masks, scores, logits = model.predict(
                    point_coords=point_coords, 
                    point_labels=point_labels,
                    multimask_output=True,
                    )

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind][0] #choose only the best result for now
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]
                mask_list.append(np.expand_dims(masks, axis=0))

            else:
                mask_list = []
                if hasattr(self, 'inference_state'):
                    model.reset_state(self.inference_state)
                self.inference_state = model.init_state(image.permute(0, 3, 1, 2).contiguous(), H, W)

                _, out_obj_ids, out_mask_logits = model.add_new_points(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=point_coords,
                    labels=point_labels,
                )
                pbar = ProgressBar(B)
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(self.inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    pbar.update(1)
                for frame_idx, obj_masks in video_segments.items():
                    for out_obj_id, out_mask in obj_masks.items():
                        mask_list.append(out_mask)

        if not keep_model_loaded:
            try:
                model.to(offload_device)
            except:
                model.model.to(offload_device)
        
        out_list = []
        for mask in mask_list:
            mask_tensor = torch.from_numpy(mask)
            mask_tensor = mask_tensor.permute(1, 2, 0).cpu().float()
            mask_tensor = mask_tensor.mean(dim=-1, keepdim=True)
            mask_tensor = mask_tensor.repeat(1, 1, 3)
            mask_tensor = mask_tensor[:, :, 0]
            out_list.append(mask_tensor)
        mask_tensor = torch.stack(out_list, dim=0)
        return (mask_tensor,)
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadSAM2Model": DownloadAndLoadSAM2Model,
    "Sam2Segmentation": Sam2Segmentation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadSAM2Model": "(Down)Load SAM2Model",
    "Sam2Segmentation": "Sam2Segmentation",
}
