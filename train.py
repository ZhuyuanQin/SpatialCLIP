"""
三阶段训练脚本
Stage0: 冻结CLIP，训练空间分支+门控+分类头
Stage1: 解冻投影层，加入InfoNCE和模态dropout
Stage2: 解冻部分CLIP层，联合微调
"""
import os
import json
import argparse
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve

from trimodal_clip import TrimodalCLIP, TrimodalClassifier
from dataset import create_dataloaders, ModalityDropout


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据路径
    train_json: str = "/data/qinzhuyuan/0926国庆/regenerated_train_data.json"
    val_json: str = "/data/qinzhuyuan/0926国庆/regenerated_valid_data.json"
    test_json: str = "/data/qinzhuyuan/0926国庆/regenerated_test_data.json"
    clip_model_path: str = "/data/qinzhuyuan/0926国庆/clip-vit-large-patch14"
    fetalclip_weights: str = "/data/qinzhuyuan/0926国庆/FetalCLIP_weights_hf_vitl14.bin"
    
    # 模型参数
    max_objects: int = 32
    max_length: int = 117
    image_size: Tuple[int, int] = (224, 224)
    fusion_type: str = "gated"  # "gated" or "concat"
    
    # 训练参数
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    
    # 阶段配置 - 大幅增加训练轮次冲击95%
    stage0_epochs: int = 8   # 增加到8轮
    stage1_epochs: int = 15  # 增加到15轮  
    stage2_epochs: int = 35  # 增加到35轮
    
    # 学习率 - 稳定且有效的设置 (避免NaN)
    clip_backbone_lr: float = 8e-6   # 保守安全的学习率
    clip_projection_lr: float = 2e-5  # 适中学习率
    spatial_lr: float = 6e-5         # 适中学习率
    
    # 优化器参数
    weight_decay: float = 0.02  # 增加正则化
    warmup_ratio: float = 0.2   # 更长warmup
    max_grad_norm: float = 1.0
    
    # 损失权重 - 稳定的对比学习权重 (避免NaN)
    classification_weight: float = 1.0
    contrastive_weights: Tuple[float, float, float] = (0.06, 0.04, 0.04)  # 温和的对比损失
    preservation_weight: float = 1e-4  # 适中保持项权重
    
    # 模态dropout - 保持关闭避免过度正则化
    modality_dropout_prob: float = 0.0
    
    # 解冻层数 - 解冻更多层增强学习能力
    unfreeze_vision_layers: int = 8  # 增加ViT顶部层数
    unfreeze_text_layers: int = 6    # 增加文本顶部层数
    
    # 其他
    use_amp: bool = True
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./outputs"
    experiment_name: str = "trimodal_training"
    
    # 评估和保存 - 更频繁验证和保存更多模型
    eval_steps: int = 200   # 更频繁评估
    save_steps: int = 500
    log_steps: int = 50     # 更频繁日志


class TrimodalTrainer:
    """三模态训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 创建输出目录
        self.output_dir = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 保存配置
        self._save_config()
        
        # 创建数据加载器
        self._create_dataloaders()
        
        # 创建模型
        self._create_model()
        
        # 创建优化器和调度器
        self._create_optimizers()
        
        # 创建损失函数
        self._create_loss_functions()
        
        # 其他组件
        self.scaler = torch.amp.GradScaler(device='cuda') if config.use_amp else None
        self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        self.global_step = 0
        self.best_auc = 0.0
        
        # 模态dropout
        self.modality_dropout = ModalityDropout(
            image_dropout_prob=config.modality_dropout_prob,
            text_dropout_prob=config.modality_dropout_prob,
            spatial_dropout_prob=config.modality_dropout_prob
        ).to(self.device)
        
        self.logger.info("Trainer initialized successfully")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _save_config(self):
        """保存配置"""
        config_dict = self.config.__dict__.copy()
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        self.logger.info("Creating data loaders...")
        
        # 加载processor
        self.processor = CLIPProcessor.from_pretrained(
            self.config.clip_model_path,
            local_files_only=True
        )
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_json=self.config.train_json,
            val_json=self.config.val_json,
            test_json=self.config.test_json,
            processor=self.processor,
            batch_size=self.config.batch_size,
            max_objects=self.config.max_objects,
            max_length=self.config.max_length,
            image_size=self.config.image_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
        self.logger.info(f"Test batches: {len(self.test_loader)}")
    
    def _create_model(self):
        """创建模型"""
        self.logger.info("Creating model...")
        
        # 创建配置
        spatial_config = {
            'embed_dim': 768,
            'num_anatomy_labels': 14,
            'num_transformer_layers': 2,
            'num_heads': 8,
            'ffn_dim': 2048,
            'dropout': 0.1,
            'max_objects': self.config.max_objects,
            'use_cls_token': True,
            'token_drop_prob': 0.1,
            'confidence_scaling': True
        }
        
        # 直接创建三模态CLIP
        trimodal_clip = TrimodalCLIP(
            clip_model_path=self.config.clip_model_path,
            fetalclip_weights_path=self.config.fetalclip_weights,
            spatial_config=spatial_config,
            projection_dim=768,
            freeze_backbone=True,  # 初始冻结
            contrastive_weights=self.config.contrastive_weights
        )
        
        # 创建分类模型
        self.model = TrimodalClassifier(
            trimodal_clip=trimodal_clip,
            num_classes=1,  # 二分类
            fusion_type=self.config.fusion_type,
            embed_dim=768,
            freeze_backbone=True  # 初始冻结
        ).to(self.device)
        
        # 存储初始权重用于保持项
        self.initial_image_features = None
        self.initial_text_features = None
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _create_optimizers(self):
        """创建优化器和调度器"""
        # 参数分组
        self.param_groups = {
            'clip_backbone': [],
            'clip_projection': [],
            'spatial_and_fusion': []
        }
        
        # 分组参数
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'trimodal_clip.clip_model' in name:
                if 'projection' in name:
                    self.param_groups['clip_projection'].append(param)
                else:
                    self.param_groups['clip_backbone'].append(param)
            else:
                self.param_groups['spatial_and_fusion'].append(param)
        
        # 创建优化器（初始只有空间分支参数）
        self._update_optimizer()
        
        # 学习率调度器将在每个阶段重新创建
        self.scheduler = None
    
    def _update_optimizer(self):
        """更新优化器"""
        optimizer_params = []
        
        if self.param_groups['clip_backbone']:
            optimizer_params.append({
                'params': self.param_groups['clip_backbone'],
                'lr': self.config.clip_backbone_lr,
                'weight_decay': self.config.weight_decay
            })
        
        if self.param_groups['clip_projection']:
            optimizer_params.append({
                'params': self.param_groups['clip_projection'],
                'lr': self.config.clip_projection_lr,
                'weight_decay': self.config.weight_decay
            })
        
        if self.param_groups['spatial_and_fusion']:
            optimizer_params.append({
                'params': self.param_groups['spatial_and_fusion'],
                'lr': self.config.spatial_lr,
                'weight_decay': self.config.weight_decay
            })
        
        self.optimizer = torch.optim.AdamW(optimizer_params)
    
    def _create_scheduler(self, num_training_steps: int):
        """创建学习率调度器"""
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def _create_loss_functions(self):
        """创建损失函数"""
        self.classification_criterion = nn.BCEWithLogitsLoss()
    
    def _freeze_clip_backbone(self):
        """冻结CLIP主干"""
        for param in self.model.trimodal_clip.clip_model.parameters():
            param.requires_grad = False
        self.logger.info("CLIP backbone frozen")
    
    def _unfreeze_clip_projection(self):
        """解冻CLIP投影层"""
        # 解冻图像投影
        if hasattr(self.model.trimodal_clip.clip_model, 'visual_projection'):
            for param in self.model.trimodal_clip.clip_model.visual_projection.parameters():
                param.requires_grad = True
        
        # 解冻文本投影  
        if hasattr(self.model.trimodal_clip.clip_model, 'text_projection'):
            for param in self.model.trimodal_clip.clip_model.text_projection.parameters():
                param.requires_grad = True
        
        # 解冻logit_scale
        self.model.trimodal_clip.logit_scale.requires_grad = True
        
        self.logger.info("CLIP projection layers unfrozen")
    
    def _unfreeze_clip_layers(self):
        """解冻CLIP部分层"""
        # 解冻视觉编码器顶部层
        vision_layers = self.model.trimodal_clip.clip_model.vision_model.encoder.layers
        for layer in vision_layers[-self.config.unfreeze_vision_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # 解冻文本编码器顶部层
        text_layers = self.model.trimodal_clip.clip_model.text_model.encoder.layers
        for layer in text_layers[-self.config.unfreeze_text_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.logger.info(f"Unfrozen top {self.config.unfreeze_vision_layers} vision layers and {self.config.unfreeze_text_layers} text layers")
    
    def _store_initial_features(self, batch: Dict[str, torch.Tensor]):
        """存储初始特征用于保持项"""
        if self.initial_image_features is not None:
            return
        
        self.model.eval()
        with torch.no_grad():
            features = self.model.extract_features(batch)
            if 'image_embeds' in features:
                self.initial_image_features = features['image_embeds'].clone()
            if 'text_embeds' in features:
                self.initial_text_features = features['text_embeds'].clone()
        
        self.logger.info("Initial features stored for preservation loss")
    
    def train_epoch(self, epoch: int, stage: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_metrics = {'loss': [], 'classification_loss': []}
        
        # 添加准确率跟踪
        all_preds = []
        all_labels = []
        
        # 添加InfoNCE损失（Stage1开始）
        if stage >= 1:
            epoch_metrics.update({
                'image_text_loss': [],
                'image_spatial_loss': [],
                'text_spatial_loss': []
            })
        
        # 添加保持项损失（Stage2）
        if stage >= 2:
            epoch_metrics['preservation_loss'] = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} (Stage {stage})")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # 模态dropout（Stage1开始）
            if stage >= 1:
                batch = self.modality_dropout(batch)
            
            # 存储初始特征（Stage2）
            if stage >= 2 and self.initial_image_features is None:
                self._store_initial_features(batch)
            
            # 前向传播
            if self.config.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self._forward_pass(batch, stage)
            else:
                outputs = self._forward_pass(batch, stage)
            
            loss = outputs['total_loss']
            
            # 计算预测准确率
            with torch.no_grad():
                logits = outputs['logits'].squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
            
            # 检查loss是否为NaN
            if torch.isnan(loss):
                self.logger.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
                continue
            
            # 反向传播
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            # 记录指标
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['classification_loss'].append(outputs['classification_loss'].item())
            
            if stage >= 1:
                for key in ['image_text_loss', 'image_spatial_loss', 'text_spatial_loss']:
                    if key in outputs:
                        epoch_metrics[key].append(outputs[key].item())
            
            if stage >= 2 and 'preservation_loss' in outputs:
                epoch_metrics['preservation_loss'].append(outputs['preservation_loss'].item())
            
            # 计算当前累积准确率
            current_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 日志记录
            if self.global_step % self.config.log_steps == 0:
                self._log_metrics(outputs, stage, 'train')
            
            # 验证
            if self.global_step % self.config.eval_steps == 0:
                val_metrics = self.validate(stage)
                self._log_metrics(val_metrics, stage, 'val')
                
                # 保存最佳模型
                if val_metrics.get('auc', 0) > self.best_auc:
                    self.best_auc = val_metrics['auc']
                    self._save_checkpoint('best')
            
            # 保存检查点
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint('latest')
            
            self.global_step += 1
        
        # 计算epoch平均指标和最终准确率
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        final_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        avg_metrics['accuracy'] = final_acc
        
        return avg_metrics
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor], stage: int) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 显式参数调用，在Stage>=1时传递return_loss=True启用InfoNCE
        outputs = self.model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            bboxes=batch['bboxes'],
            anatomy_labels=batch['anatomy_labels'],
            confidences=batch['confidences'],
            spatial_padding_mask=batch['spatial_padding_mask'],
            image_size=batch['image_size'],
            return_loss=(stage >= 1),  # Stage≥1时启用InfoNCE
            return_dict=True
        )
        
        total_loss = 0.0
        
        # 分类损失
        classification_loss = self.classification_criterion(
            outputs['logits'].squeeze(-1),
            batch['labels'].float()
        )
        total_loss += self.config.classification_weight * classification_loss
        outputs['classification_loss'] = classification_loss
        
        # InfoNCE损失（Stage1开始）
        if stage >= 1:
            contrastive_losses = outputs
            
            if 'image_text_loss' in contrastive_losses:
                total_loss += self.config.contrastive_weights[0] * contrastive_losses['image_text_loss']
            if 'image_spatial_loss' in contrastive_losses:
                total_loss += self.config.contrastive_weights[1] * contrastive_losses['image_spatial_loss']
            if 'text_spatial_loss' in contrastive_losses:
                total_loss += self.config.contrastive_weights[2] * contrastive_losses['text_spatial_loss']
        
        # 保持项损失（Stage2）
        if stage >= 2 and self.initial_image_features is not None:
            preservation_loss = 0.0
            current_features = self.model.extract_features(batch)
            
            # 获取当前batch size
            current_batch_size = batch['pixel_values'].shape[0]
            
            if 'image_embeds' in current_features and self.initial_image_features is not None:
                # 只对相同数量的样本计算preservation loss
                min_size = min(current_batch_size, self.initial_image_features.shape[0])
                preservation_loss += F.mse_loss(
                    current_features['image_embeds'][:min_size], 
                    self.initial_image_features[:min_size]
                )
            
            if 'text_embeds' in current_features and self.initial_text_features is not None:
                # 只对相同数量的样本计算preservation loss  
                min_size = min(current_batch_size, self.initial_text_features.shape[0])
                preservation_loss += F.mse_loss(
                    current_features['text_embeds'][:min_size], 
                    self.initial_text_features[:min_size]
                )
            
            total_loss += self.config.preservation_weight * preservation_loss
            outputs['preservation_loss'] = preservation_loss
        
        outputs['total_loss'] = total_loss
        return outputs
    
    def validate(self, stage: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False, ncols=0, disable=True):
                # 移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 前向传播
                outputs = self._forward_pass(batch, stage)
                val_losses.append(outputs['total_loss'].item())
                
                # 预测
                logits = outputs['logits'].squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算指标 - 检查NaN值
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        # 检查概率值中是否有NaN
        all_probs_clean = np.array(all_probs)
        if np.any(np.isnan(all_probs_clean)):
            self.logger.warning("NaN detected in validation probabilities, setting AUC to 0.5")
            auc = 0.5
        else:
            auc = roc_auc_score(all_labels, all_probs_clean)
        
        avg_loss = np.mean(val_losses)
        
        return {
            'loss': avg_loss,
            'accuracy': acc,
            'f1': f1,
            'auc': auc
        }
    
    def _log_metrics(self, metrics: Dict[str, Any], stage: int, prefix: str):
        """记录指标"""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # 标量tensor
                    value = value.item()
                else:  # 多元素tensor，跳过记录
                    continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/stage{stage}/{key}", value, self.global_step)
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_auc': self.best_auc,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{name}.pt")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """主训练流程"""
        self.logger.info("Starting training...")
        
        # Stage 0: 冻结CLIP，训练空间分支
        self.logger.info("="*60)
        self.logger.info("STAGE 0: Training spatial encoder + fusion + classifier")
        self.logger.info("="*60)
        self.logger.info("STAGE 0: Training spatial encoder + fusion + classifier")
        self.logger.info("="*60)
        
        self._freeze_clip_backbone()
        self._update_optimizer()
        
        num_training_steps = len(self.train_loader) * self.config.stage0_epochs
        self._create_scheduler(num_training_steps)
        
        for epoch in range(self.config.stage0_epochs):
            epoch_metrics = self.train_epoch(epoch, stage=0)
            self.logger.info(f"Stage 0 Epoch {epoch+1}: " + 
                           " | ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]))
        
        # Stage 1: 解冻投影层，加InfoNCE和dropout
        self.logger.info("="*60)
        self.logger.info("STAGE 1: Unfreeze projections + InfoNCE + modality dropout")
        self.logger.info("="*60)
        self.logger.info("STAGE 1: Unfreeze projections + InfoNCE + modality dropout")
        self.logger.info("="*60)
        
        self._unfreeze_clip_projection()
        self._update_optimizer()
        
        num_training_steps = len(self.train_loader) * self.config.stage1_epochs
        self._create_scheduler(num_training_steps)
        
        for epoch in range(self.config.stage1_epochs):
            epoch_metrics = self.train_epoch(epoch, stage=1)
            self.logger.info(f"Stage 1 Epoch {epoch+1}: " + 
                           " | ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]))
        
        # Stage 2: 解冻部分层，联合微调
        self.logger.info("="*60)
        self.logger.info("STAGE 2: Unfreeze top layers + joint fine-tuning")
        self.logger.info("="*60)
        self.logger.info("STAGE 2: Unfreeze top layers + joint fine-tuning")
        self.logger.info("="*60)
        
        self._unfreeze_clip_layers()
        self._update_optimizer()
        
        num_training_steps = len(self.train_loader) * self.config.stage2_epochs
        self._create_scheduler(num_training_steps)
        
        for epoch in range(self.config.stage2_epochs):
            epoch_metrics = self.train_epoch(epoch, stage=2)
            self.logger.info(f"Stage 2 Epoch {epoch+1}: " + 
                           " | ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]))
        
        # 最终验证
        self.logger.info("="*60)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("="*60)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("="*60)
        
        final_metrics = self.validate(stage=2)
        self.logger.info("Final validation metrics:")
        for key, value in final_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        # 保存最终模型
        self._save_checkpoint('final')
        self.writer.close()
        
        self.logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="三模态CLIP训练")
    
    # 数据路径
    parser.add_argument("--train_json", default="/data/qinzhuyuan/0926国庆/regenerated_train_data.json")
    parser.add_argument("--val_json", default="/data/qinzhuyuan/0926国庆/regenerated_valid_data.json")
    parser.add_argument("--test_json", default="/data/qinzhuyuan/0926国庆/regenerated_test_data.json")
    parser.add_argument("--clip_model_path", default="/data/qinzhuyuan/0926国庆/clip-vit-large-patch14")
    parser.add_argument("--fetalclip_weights", default="/data/qinzhuyuan/0926国庆/FetalCLIP_weights_hf_vitl14.bin")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--fusion_type", choices=["gated", "concat"], default="gated")
    
    # 阶段配置
    parser.add_argument("--stage0_epochs", type=int, default=5)
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=15)
    
    # 学习率
    parser.add_argument("--clip_backbone_lr", type=float, default=1e-5)
    parser.add_argument("--clip_projection_lr", type=float, default=2e-5)
    parser.add_argument("--spatial_lr", type=float, default=1e-4)
    
    # 其他
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--experiment_name", default="trimodal_training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(**vars(args))
    
    # 创建训练器
    trainer = TrimodalTrainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()