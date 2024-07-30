import torch

import os
import numpy as np
import yaml
import json
from .sam2.modeling.sam2_base import SAM2Base
from .sam2.modeling.backbones.image_encoder import ImageEncoder
from .sam2.modeling.backbones.hieradet import Hiera
from .sam2.modeling.backbones.image_encoder import FpnNeck
from .sam2.modeling.position_encoding import PositionEmbeddingSine
from .sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from .sam2.modeling.sam.transformer import RoPEAttention
from .sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from contextlib import nullcontext

from .sam2.sam2_image_predictor import SAM2ImagePredictor

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

class DownloadAndLoadSAM2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'sam2_hiera_base_plus.safetensors',
                    #'sam2_hiera_large.pt',
                    ],
                    {
                    "default": 'sam2_hiera_base_plus.safetensors'
                    }),
            "device": (
                    [ 
                    'cuda',
                    'cpu',
                    ],
                    {
                    "default": 'cpu'
                    }),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),

            },
        }

    RETURN_TYPES = ("SAM2MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "SAM2"

    def loadmodel(self, model, device, precision):
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

        if "base" in model:  
            model_cfg_path = os.path.join(script_directory, "sam2_configs", "sam2_hiera_b+.yaml")
        elif "large" in model:
            model_cfg_path = os.path.join(script_directory, "sam2_configs", "sam2_hiera_l.yaml")
        elif "small" in model:
            model_cfg_path = os.path.join(script_directory, "sam2_configs", "sam2_hiera_s.yaml")
        elif "tiny" in model:
            model_cfg_path = os.path.join(script_directory, "sam2_configs", "sam2_hiera_t.yaml")

       # Load the YAML configuration
        with open(model_cfg_path, 'r') as file:
            config = yaml.safe_load(file)

        # Extract the model configuration
        model_config = config['model']

        # Instantiate the image encoder components
        trunk_config = model_config['image_encoder']['trunk']
        neck_config = model_config['image_encoder']['neck']
        position_encoding_config = neck_config['position_encoding']

        position_encoding = PositionEmbeddingSine(
            num_pos_feats=position_encoding_config['num_pos_feats'],
            normalize=position_encoding_config['normalize'],
            scale=position_encoding_config['scale'],
            temperature=position_encoding_config['temperature']
        )

        neck = FpnNeck(
            position_encoding=position_encoding,
            d_model=neck_config['d_model'],
            backbone_channel_list=neck_config['backbone_channel_list'],
            fpn_top_down_levels=neck_config['fpn_top_down_levels'],
            fpn_interp_model=neck_config['fpn_interp_model']
        )

        trunk = Hiera(
            embed_dim=trunk_config['embed_dim'],
            num_heads=trunk_config['num_heads']
        )

        image_encoder = ImageEncoder(
            scalp=model_config['image_encoder']['scalp'],
            trunk=trunk,
            neck=neck
        )
        # Instantiate the memory attention components
        memory_attention_layer_config = config['model']['memory_attention']['layer']
        self_attention_config = memory_attention_layer_config['self_attention']
        cross_attention_config = memory_attention_layer_config['cross_attention']

        self_attention = RoPEAttention(
            rope_theta=self_attention_config['rope_theta'],
            feat_sizes=self_attention_config['feat_sizes'],
            embedding_dim=self_attention_config['embedding_dim'],
            num_heads=self_attention_config['num_heads'],
            downsample_rate=self_attention_config['downsample_rate'],
            dropout=self_attention_config['dropout']
        )

        cross_attention = RoPEAttention(
            rope_theta=cross_attention_config['rope_theta'],
            feat_sizes=cross_attention_config['feat_sizes'],
            rope_k_repeat=cross_attention_config['rope_k_repeat'],
            embedding_dim=cross_attention_config['embedding_dim'],
            num_heads=cross_attention_config['num_heads'],
            downsample_rate=cross_attention_config['downsample_rate'],
            dropout=cross_attention_config['dropout'],
            kv_in_dim=cross_attention_config['kv_in_dim']
        )

        memory_attention_layer = MemoryAttentionLayer(
            activation=memory_attention_layer_config['activation'],
            dim_feedforward=memory_attention_layer_config['dim_feedforward'],
            dropout=memory_attention_layer_config['dropout'],
            pos_enc_at_attn=memory_attention_layer_config['pos_enc_at_attn'],
            self_attention=self_attention,
            d_model=memory_attention_layer_config['d_model'],
            pos_enc_at_cross_attn_keys=memory_attention_layer_config['pos_enc_at_cross_attn_keys'],
            pos_enc_at_cross_attn_queries=memory_attention_layer_config['pos_enc_at_cross_attn_queries'],
            cross_attention=cross_attention
        )

        memory_attention = MemoryAttention(
            d_model=config['model']['memory_attention']['d_model'],
            pos_enc_at_input=config['model']['memory_attention']['pos_enc_at_input'],
            layer=memory_attention_layer,
            num_layers=config['model']['memory_attention']['num_layers']
        )

        # Instantiate the memory encoder components
        memory_encoder_config = config['model']['memory_encoder']
        position_encoding_mem_enc_config = memory_encoder_config['position_encoding']
        mask_downsampler_config = memory_encoder_config['mask_downsampler']
        fuser_layer_config = memory_encoder_config['fuser']['layer']

        position_encoding_mem_enc = PositionEmbeddingSine(
            num_pos_feats=position_encoding_mem_enc_config['num_pos_feats'],
            normalize=position_encoding_mem_enc_config['normalize'],
            scale=position_encoding_mem_enc_config['scale'],
            temperature=position_encoding_mem_enc_config['temperature']
        )

        mask_downsampler = MaskDownSampler(
            kernel_size=mask_downsampler_config['kernel_size'],
            stride=mask_downsampler_config['stride'],
            padding=mask_downsampler_config['padding']
        )

        fuser_layer = CXBlock(
            dim=fuser_layer_config['dim'],
            kernel_size=fuser_layer_config['kernel_size'],
            padding=fuser_layer_config['padding'],
            layer_scale_init_value=float(fuser_layer_config['layer_scale_init_value'])
        )
        fuser = Fuser(
            num_layers=memory_encoder_config['fuser']['num_layers'],
            layer=fuser_layer
        )

        memory_encoder = MemoryEncoder(
            position_encoding=position_encoding_mem_enc,
            mask_downsampler=mask_downsampler,
            fuser=fuser,
            out_dim=memory_encoder_config['out_dim']
        )

        sam_mask_decoder_extra_args = {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        }

        base_model = SAM2Base(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
            num_maskmem=model_config['num_maskmem'],
            image_size=model_config['image_size'],
            sigmoid_scale_for_mem_enc=model_config['sigmoid_scale_for_mem_enc'],
            sigmoid_bias_for_mem_enc=model_config['sigmoid_bias_for_mem_enc'],
            use_mask_input_as_output_without_sam=model_config['use_mask_input_as_output_without_sam'],
            directly_add_no_mem_embed=model_config['directly_add_no_mem_embed'],
            use_high_res_features_in_sam=model_config['use_high_res_features_in_sam'],
            multimask_output_in_sam=model_config['multimask_output_in_sam'],
            iou_prediction_use_sigmoid=model_config['iou_prediction_use_sigmoid'],
            use_obj_ptrs_in_encoder=model_config['use_obj_ptrs_in_encoder'],
            add_tpos_enc_to_obj_ptrs=model_config['add_tpos_enc_to_obj_ptrs'],
            only_obj_ptrs_in_the_past_for_eval=model_config['only_obj_ptrs_in_the_past_for_eval'],
            pred_obj_scores=model_config['pred_obj_scores'],
            pred_obj_scores_mlp=model_config['pred_obj_scores_mlp'],
            fixed_no_obj_ptr=model_config['fixed_no_obj_ptr'],
            multimask_output_for_tracking=model_config['multimask_output_for_tracking'],
            use_multimask_token_for_obj_ptr=model_config['use_multimask_token_for_obj_ptr'],
            compile_image_encoder = model_config['compile_image_encoder'],
            multimask_min_pt_num = model_config['multimask_min_pt_num'],
            multimask_max_pt_num = model_config['multimask_max_pt_num'],
            use_mlp_for_obj_ptr_proj = model_config['use_mlp_for_obj_ptr_proj'],

        ).to(dtype).to(device)
        
        #print(base_model)
        sd = load_torch_file(model_path)
        #for key in sd['model']:
        #    print(key)
        base_model.load_state_dict(sd)
            
        model = SAM2ImagePredictor(base_model)
        
        sam2_model = {
            'model': model, 
            'dtype': dtype
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
                "point_labels": ("INT", {"default": 0}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("MASK", )
    RETURN_NAMES =("mask", )
    FUNCTION = "segment"
    CATEGORY = "SAM2"

    def segment(self, image, sam2_model, coordinates, keep_model_loaded, point_labels):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = sam2_model["model"]
        dtype = sam2_model["dtype"]

        image_np = (image[0].contiguous() * 255).byte().numpy()

        coordinates = json.loads(coordinates.replace("'", '"'))
        coordinates = [(coord['x'], coord['y']) for coord in coordinates]
        point_coords = np.array([coordinates[0]])
        print(point_coords)
        point_labels = np.array([point_labels])
        
        autocast_condition = not mm.is_device_mps(device)
        
        with torch.autocast(mm.get_autocast_device(model.device), dtype=dtype) if autocast_condition else nullcontext():
            model.set_image(image_np)
            print(model._features["image_embed"].shape, model._features["image_embed"][-1].shape)
            masks, scores, logits = model.predict(
                point_coords=point_coords, 
                point_labels=point_labels,
                multimask_output=True,
                )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        print(type(masks))
        print(masks.shape)
        print(masks.min(), masks.max())
        mask_tensor = torch.from_numpy(masks)
        mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
        mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
        mask_tensor = mask_tensor[:, :, :, 0]
        print(mask_tensor.shape)
        print(mask_tensor.min(), mask_tensor.max())
        return (mask_tensor,)
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadSAM2Model": DownloadAndLoadSAM2Model,
    "Sam2Segmentation": Sam2Segmentation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadSAM2Model": "(Down)Load SAM2Model",
    "Sam2Segmentation": "Sam2Segmentation",
}
