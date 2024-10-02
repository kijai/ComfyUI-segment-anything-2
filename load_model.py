import yaml
from .sam2.modeling.sam2_base import SAM2Base
from .sam2.modeling.backbones.image_encoder import ImageEncoder
from .sam2.modeling.backbones.hieradet import Hiera
from .sam2.modeling.backbones.image_encoder import FpnNeck
from .sam2.modeling.position_encoding import PositionEmbeddingSine
from .sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from .sam2.modeling.sam.transformer import RoPEAttention
from .sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock

from .sam2.sam2_image_predictor import SAM2ImagePredictor
from .sam2.sam2_video_predictor import SAM2VideoPredictor
from .sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from comfy.utils import load_torch_file

def load_model(model_path, model_cfg_path, segmentor, dtype, device):
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

    keys_to_include = ['embed_dim', 'num_heads', 'global_att_blocks', 'window_pos_embed_bkg_spatial_size', 'stages']
    trunk_kwargs = {key: trunk_config[key] for key in keys_to_include if key in trunk_config}
    trunk = Hiera(**trunk_kwargs)

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

    def initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device):
        return model_class(
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
            compile_image_encoder=model_config['compile_image_encoder'],
            multimask_min_pt_num=model_config['multimask_min_pt_num'],
            multimask_max_pt_num=model_config['multimask_max_pt_num'],
            use_mlp_for_obj_ptr_proj=model_config['use_mlp_for_obj_ptr_proj'],
            proj_tpos_enc_in_obj_ptrs=model_config['proj_tpos_enc_in_obj_ptrs'],
            no_obj_embed_spatial=model_config['no_obj_embed_spatial'],
            use_signed_tpos_enc_to_obj_ptrs=model_config['use_signed_tpos_enc_to_obj_ptrs'],
            binarize_mask_from_pts_for_mem_enc=True if segmentor == 'video' else False,
        ).to(dtype).to(device).eval()

    # Load the state dictionary
    sd = load_torch_file(model_path)

    # Initialize model based on segmentor type
    if segmentor == 'single_image':
        model_class = SAM2Base
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
        model = SAM2ImagePredictor(model)
    elif segmentor == 'video':
        model_class = SAM2VideoPredictor
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
    elif segmentor == 'automaskgenerator':
        model_class = SAM2Base
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
        model = SAM2AutomaticMaskGenerator(model)
    else:
        raise ValueError(f"Segmentor {segmentor} not supported")

    return model