"""
项目配置文件
"""

# 项目基本信息
PROJECT_NAME = "TrimodalCLIP"
VERSION = "1.0.0"
DESCRIPTION = "三模态CLIP训练项目：图像+文本+空间编码器"

# 路径配置
PATHS = {
    "base_dir": "/data/qinzhuyuan/0926国庆",
    "clip_model": "/data/qinzhuyuan/0926国庆/clip-vit-large-patch14",
    "fetalclip_weights": "/data/qinzhuyuan/0926国庆/FetalCLIP_weights_hf_vitl14.bin",
    "train_data": "/data/qinzhuyuan/0926国庆/regenerated_train_data.json",
    "val_data": "/data/qinzhuyuan/0926国庆/regenerated_valid_data.json",
    "test_data": "/data/qinzhuyuan/0926国庆/regenerated_test_data.json",
    "output_dir": "/data/qinzhuyuan/0926国庆/outputs"
}

# 模型配置
MODEL_CONFIG = {
    "embed_dim": 768,
    "projection_dim": 768,
    "max_objects": 32,  # 提升到32个目标
    "max_text_length": 117,  # 保持原有长度
    "image_size": (224, 224),  # 保持原有分辨率
    "fusion_type": "adaptive_gated",  # 改为自适应门控式融合
    
    # Adapter/LoRA配置 - 主干网络增强
    "adapter_config": {
        "enabled": True,
        "adapter_type": "lora",  # "adapter" or "lora"
        "target_modules": ["vision_transformer", "text_transformer"],
        "target_layers": ["last_2_blocks"],  # 最后2个block
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "adapter_bottleneck": 64,
        "insert_positions": ["attention_output", "mlp_output"]
    },
    
    # 自适应门控式晚融合配置
    "adaptive_fusion": {
        "enabled": True,
        "gate_network": {
            "hidden_dim": 256,
            "dropout": 0.1,
            "l1_regularization": 1e-4
        },
        "modality_weights_init": [0.4, 0.4, 0.2],  # img, txt, spa初始权重
        "temperature": 1.0
    },
    
    # 空间编码器配置 - DETR风格
    "spatial_encoder": {
        "encoding_type": "detr_style",  # "legacy" or "detr_style"
        "num_anatomy_labels": 14,
        "box_encoding": {
            "normalized_coords": True,  # 使用标准化的(cx,cy,w,h)
            "positional_encoding": "2d_sincos",  # 二维正余弦位置编码
            "encoding_dim": 256
        },
        "cross_attention": {
            "enabled": True,
            "num_layers": 1,
            "num_heads": 1,
            "dropout": 0.1,
            "only_spatial_image": True  # 只让空间token和图像CLS交互
        },
        "num_transformer_layers": 2,
        "num_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.1,
        "use_cls_token": True,
        "token_drop_prob": 0.1,
        "confidence_scaling": True
    },
    
    # 解剖标签映射
    "anatomy_labels": {
        'CB': 0, 'CP1': 1, 'CP2': 2, 'CF': 3, 'NT': 4, 'NB': 5,
        'F': 6, 'NA': 7, 'CRL': 8, 'S': 9, 'AS': 10, 'UC': 11,
        'UI': 12, 'UNKNOWN': 13
    }
}

# 训练配置
TRAINING_CONFIG = {
    "batch_size": 8,  # 恢复原batch size
    "num_workers": 4,
    "pin_memory": True,
    
    # 三阶段训练 - 根据用户建议更新
    "stages": {
        "stage0": {
            "epochs": 8,  # 增加轮次
            "description": "冻结CLIP主干，训练空间分支+门控+分类头",
            "freeze_clip": True,
            "use_contrastive": False,
            "use_modality_dropout": False,
            "use_amp": True,
            "warmup_ratio": 0.2,  # 更长warmup
            "learning_rates": {
                "classifier": 2e-3,  # 大幅提升
                "spatial": 8e-4,     # 提升
                "fusion": 8e-4,      # 提升
                "adaptive_gate": 8e-4
            }
        },
        "stage1": {
            "epochs": 15,  # 增加轮次
            "description": "解冻投影层+LoRA，开启监督对比学习",
            "unfreeze_projection": True,
            "unfreeze_lora": True,
            "use_contrastive": True,
            "contrastive_type": "supervised",
            "use_modality_dropout": False,
            "use_amp": True,
            "warmup_ratio": 0.2,
            "contrast_warmup_ratio": 0.4,  # 更快warmup
            "learning_rates": {
                "projection": 2e-4,      # 大幅提升
                "classifier": 1.5e-3,    # 提升
                "spatial": 6e-4,         # 提升
                "fusion": 6e-4,          # 提升
                "lora_adapters": 2e-4,   # 提升
                "adaptive_gate": 6e-4
            }
        },
        "stage2": {
            "epochs": 35,  # 大幅增加轮次冲击95%
            "description": "微调主干(通过LoRA)+长时间优化",
            "unfreeze_layers": {
                "vision_layers": 4,  # 解冻更多层
                "text_layers": 4
            },
            "use_preservation_loss": True,
            "preservation_weight": 2e-4,  # 增加权重
            "use_amp": False,
            "warmup_ratio": 0.3,  # 更长warmup给充分时间
            "contrast_warmup_ratio": 0.3,
            "learning_rates": {
                "backbone": 5e-6,        # 提升主干学习率
                "projection": 1.5e-4,    # 大幅提升
                "classifier": 2.5e-4,    # 大幅提升
                "spatial": 2.5e-4,       # 大幅提升
                "fusion": 2.5e-4,        # 大幅提升
                "lora_adapters": 2.5e-4, # 大幅提升
                "adaptive_gate": 2.5e-4
            }
        }
    },
    
    # 监督式对比学习配置
    "supervised_contrastive": {
        "enabled": True,
        "temperature": 0.1,
        "use_hard_negatives": True,
        "within_batch_positives": True,  # 同类样本作为正样本
        "fp32_computation": True,
        "temperature_clamp": [0.01, 100],
        "freeze_temperature_stage2": True
    },
    
    # 优化器配置
    "optimizer": {
        "type": "AdamW",
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "max_grad_norm": 0.3,  # 用户建议的平衡值
        "warmup_scheduler": "cosine",
        "min_lr_ratio": 0.1
    },
    
    # 损失权重配置 - 稳定的对比学习权重避免NaN
    "loss_weights": {
        "classification": 1.0,
        "contrastive": {
            "image_text": 0.06,      # 温和的IT权重
            "image_spatial": 0.04,    # 温和的IS权重  
            "text_spatial": 0.04      # 温和的TS权重
        },
        "preservation": 1e-4,  # 适中保持项
        "adaptive_gate_l1": 1e-4
    },
    
    # Class-Balanced Focal Loss配置 - 优化损失函数
    "focal_loss": {
        "enabled": True,
        "alpha": "auto",  # 自动计算类别平衡权重
        "gamma": 2.5,     # 增加gamma提升困难样本关注
        "label_smoothing": 0.1,  # 增加标签平滑
        "reduction": "mean"
    },
    
    # 数值稳定性配置 - 强化EMA和稳定性
    "numerical_stability": {
        "nan_detection_threshold": 3,  # 容忍更多NaN
        "lr_decay_on_nan": 0.8,        # 更温和衰减
        "infonce_fp32": True,
        "gradient_clipping": True,
        "ema_decay": 0.9999,           # 更强EMA
        "use_ema": True
    },
    
    # 其他训练配置
    "eval_frequency": 0.2,  # 更频繁验证
    "save_best_metric": "val_acc",
    "save_top_k": 5,        # 保存更多模型
    "seed": 42  # 改为42确保一致性
}

# 数据处理配置 - 保持原有设置
DATA_CONFIG = {
    "letterbox_size": (224, 224),  # 保持原分辨率
    "letterbox_fill": (128, 128, 128),
    "text_max_length": 117,        # 保持原文本长度
    "text_truncation": True,
    "text_padding": "max_length",
    
    # 数据增强 - 适度增强
    "augmentation": {
        "training": {
            "color_jitter": {
                "brightness": 0.15,  # 适度增强
                "contrast": 0.15,    
                "saturation": 0.12,  
                "hue": 0.06         
            },
            "horizontal_flip": 0.12,  # 适度增加翻转
            "rotation": 3,            # 添加轻微旋转
            "gaussian_blur": 0.05     # 轻微高斯模糊
        },
        "validation": None,
        "test": None
    }
}

# 评估配置
EVAL_CONFIG = {
    "metrics": [
        "accuracy",
        "f1_score", 
        "auc_roc",
        "auc_pr",
        "confusion_matrix"
    ],
    "alignment_metrics": [
        "image_text_similarity",
        "image_spatial_similarity", 
        "text_spatial_similarity"
    ],
    
    # 自适应阈值配置
    "adaptive_threshold": {
        "enabled": True,
        "search_range": [0.1, 0.9],
        "search_steps": 100,
        "optimize_metric": "accuracy",
        "use_validation_set": True
    },
    
    # TTA (Test Time Augmentation) 配置
    "tta": {
        "enabled": True,
        "augmentations": [
            "horizontal_flip",
            "center_crop",
            "multi_scale"
        ],
        "scales": [0.9, 1.0, 1.1],  # 多尺度
        "max_samples": 4,  # 不超过4次采样
        "ensemble_method": "logits_mean"  # 对logits求均值
    },
    
    "save_plots": True,
    "plot_formats": ["png", "pdf"]
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "save_to_file": True,
    "use_tensorboard": True,
    "progress_bar": True
}

# 硬件配置
HARDWARE_CONFIG = {
    "device": "cuda",
    "gpu_ids": [0],
    "mixed_precision": True,
    "dataloader_workers": 4,
    "pin_memory": True
}

# 模型保存配置
SAVE_CONFIG = {
    "save_best": True,
    "save_last": True,
    "save_intermediate": True,
    "checkpoint_format": "pytorch",
    "max_checkpoints": 3
}