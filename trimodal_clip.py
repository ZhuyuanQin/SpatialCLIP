"""
三模态CLIP模型 (Trimodal CLIP)
集成图像、文本和空间编码器的多模态模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPConfig
from typing import Dict, List, Tuple, Optional, Union
import os
import json

from spatial_encoder import SpatialEncoder, create_spatial_encoder


class TrimodalCLIP(nn.Module):
    """
    三模态CLIP模型
    - 图像分支：保持原CLIP视觉编码器
    - 文本分支：保持原CLIP文本编码器  
    - 空间分支：新增的增强型空间编码器
    """
    
    def __init__(
        self,
        clip_model_path: str,
        spatial_config: Optional[Dict] = None,
        projection_dim: int = 768,
        freeze_backbone: bool = False,
        fetalclip_weights_path: Optional[str] = None,
        contrastive_weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone
        
        # 加载CLIP基座模型
        self.clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
        
        # 扩展文本位置嵌入到117（FetalCLIP的长度）
        self._expand_text_position_embeddings(117)
        
        # 冻结CLIP参数（可选）
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # 创建空间编码器
        if spatial_config is None:
            spatial_config = {
                'embed_dim': projection_dim,
                'num_anatomy_labels': 14,
                'num_transformer_layers': 2,
                'num_heads': 8,
                'ffn_dim': 2048,
                'dropout': 0.1,
                'max_objects': 100,
                'use_cls_token': True,
                'token_drop_prob': 0.1,
                'confidence_scaling': True
            }
        
        self.spatial_encoder = create_spatial_encoder(spatial_config)
        
        # 温度参数（可学习，强制限制初始值）
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)).clamp(max=4.0))  # 限制初始值
        
        # 初始化权重
        self._init_weights()
        
        # 加载FetalCLIP权重（如果提供）
        if fetalclip_weights_path is not None:
            print("Loading FetalCLIP weights...")
            missing_keys, unexpected_keys = load_fetalclip_weights(self, fetalclip_weights_path, strict=False)
            if not missing_keys and not unexpected_keys:
                print("✅ FetalCLIP weights loaded successfully!")
            else:
                print(f"⚠️  FetalCLIP weights loaded with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys")
        
        # 存储对比学习权重
        self.contrastive_weights = contrastive_weights or [1.0, 0.5, 0.5]
    
    def _expand_text_position_embeddings(self, target_length: int = 117):
        """扩展文本位置嵌入长度以支持FetalCLIP（参考fetalclip_upload.py）"""
        emb = self.clip_model.text_model.embeddings.position_embedding
        
        if emb.num_embeddings != target_length:
            print(f"Expanding text position embeddings: {emb.num_embeddings} -> {target_length}")
            
            # 创建新的embedding层
            dim = emb.embedding_dim
            new_emb = torch.nn.Embedding(target_length, dim)
            
            with torch.no_grad():
                # 复制原有的权重
                n = min(emb.num_embeddings, target_length)
                new_emb.weight[:n] = emb.weight[:n]
                
                # 如果需要更多位置，使用最后几个位置的插值或复制
                if target_length > emb.num_embeddings and emb.num_embeddings > 1:
                    # 对剩余位置使用最后一个位置嵌入的小幅变化
                    last_emb = emb.weight[-1]
                    for i in range(emb.num_embeddings, target_length):
                        # 添加小的随机扰动避免完全相同
                        noise = torch.randn_like(last_emb) * 0.02
                        new_emb.weight[i] = last_emb + noise
            
            # 替换模型中的embedding层
            self.clip_model.text_model.embeddings.position_embedding = new_emb
            
            # 更新配置
            self.clip_model.config.text_config.max_position_embeddings = target_length
            
            # 重新注册position_ids buffer
            pos_ids = torch.arange(target_length, dtype=torch.long).unsqueeze(0)
            self.clip_model.text_model.embeddings.register_buffer("position_ids", pos_ids, persistent=False)
            
            print(f"Text position embeddings expanded to {target_length}")
    
    def _init_weights(self):
        """初始化新增参数的权重"""
        # logit_scale 使用保守的初始化，强制限制范围
        with torch.no_grad():
            init_val = torch.log(torch.tensor(1.0 / 0.07)).clamp(max=4.0)  # 约等于2.66
            self.logit_scale.data.fill_(init_val)
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            pixel_values: [batch_size, 3, height, width] 图像像素值
        
        Returns:
            image_embeds: [batch_size, projection_dim] L2归一化的图像嵌入
        """
        outputs = self.clip_model.get_image_features(pixel_values=pixel_values)
        return F.normalize(outputs, p=2, dim=-1)
    
    def encode_text(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码文本（使用FP32保护数值稳定性）
        
        Args:
            input_ids: [batch_size, sequence_length] token IDs
            attention_mask: [batch_size, sequence_length] 注意力掩码
        
        Returns:
            text_embeds: [batch_size, projection_dim] L2归一化的文本嵌入
        """
        # 文本分支使用FP32计算确保数值稳定性
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # 确保输入为正确类型
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            if attention_mask is not None and attention_mask.dtype != torch.long:
                attention_mask = attention_mask.long()
            
            # 额外的attention_mask安全检查
            if attention_mask is not None:
                # 确保每个样本至少有一个有效token
                for i in range(attention_mask.shape[0]):
                    if attention_mask[i].sum().item() == 0:
                        print(f"Warning: Empty attention mask detected for sample {i}, forcing first token to be valid")
                        attention_mask[i][0] = 1
                        if input_ids[i][0].item() == 0:
                            input_ids[i][0] = 49406  # CLIP起始token
            
            outputs = self.clip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 确保输出为FP32再进行归一化
            outputs = outputs.float()
            
            # 检查输出中的NaN
            if torch.isnan(outputs).any():
                print("Warning: NaN detected in text features, replacing with zeros")
                outputs = torch.zeros_like(outputs)
            
            return F.normalize(outputs, p=2, dim=-1)
    
    def encode_spatial(
        self,
        bboxes: torch.Tensor,
        anatomy_labels: torch.Tensor,
        confidences: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        编码空间信息
        
        Args:
            bboxes: [batch_size, num_objects, 4] 归一化边界框
            anatomy_labels: [batch_size, num_objects] 解剖标签索引
            confidences: [batch_size, num_objects] 检测置信度
            padding_mask: [batch_size, num_objects] True表示padding位置
            image_size: 图像尺寸
        
        Returns:
            spatial_embeds: [batch_size, projection_dim] L2归一化的空间嵌入
        """
        return self.spatial_encoder(
            bboxes=bboxes,
            anatomy_labels=anatomy_labels,
            confidences=confidences,
            padding_mask=padding_mask,
            image_size=image_size
        )
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bboxes: Optional[torch.Tensor] = None,
        anatomy_labels: Optional[torch.Tensor] = None,
        confidences: Optional[torch.Tensor] = None,
        spatial_padding_mask: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (224, 224),
        return_loss: bool = False,
        return_dict: bool = True
    ) -> Union[Dict, Tuple]:
        """
        前向传播
        
        Args:
            pixel_values: 图像像素值
            input_ids: 文本token IDs
            attention_mask: 文本注意力掩码
            bboxes: 边界框
            anatomy_labels: 解剖标签
            confidences: 检测置信度
            spatial_padding_mask: 空间数据的padding掩码
            image_size: 图像尺寸
            return_loss: 是否计算对比学习损失
            return_dict: 是否返回字典格式
        
        Returns:
            字典包含各种嵌入和损失（如果计算）
        """
        outputs = {}
        
        # 编码各个模态
        if pixel_values is not None:
            image_embeds = self.encode_image(pixel_values)
            outputs['image_embeds'] = image_embeds
        
        if input_ids is not None:
            text_embeds = self.encode_text(input_ids, attention_mask)
            outputs['text_embeds'] = text_embeds
        
        if bboxes is not None and anatomy_labels is not None and confidences is not None:
            spatial_embeds = self.encode_spatial(
                bboxes, anatomy_labels, confidences, 
                spatial_padding_mask, image_size
            )
            outputs['spatial_embeds'] = spatial_embeds
        
        # 计算相似度和损失
        if return_loss:
            loss_dict = self.compute_contrastive_loss(outputs)
            outputs.update(loss_dict)
        
        # 计算温度缩放的相似度矩阵（强制限制logit_scale范围防止数值溢出）
        logit_scale = self.logit_scale.exp().clamp(min=1.0, max=50.0)  # 更保守的范围
        outputs['logit_scale'] = logit_scale
        
        # 计算所有可能的相似度矩阵
        if 'image_embeds' in outputs and 'text_embeds' in outputs:
            outputs['logits_per_image'] = logit_scale * outputs['image_embeds'] @ outputs['text_embeds'].t()
            outputs['logits_per_text'] = outputs['logits_per_image'].t()
        
        if 'image_embeds' in outputs and 'spatial_embeds' in outputs:
            outputs['logits_per_image_spatial'] = logit_scale * outputs['image_embeds'] @ outputs['spatial_embeds'].t()
            outputs['logits_per_spatial_image'] = outputs['logits_per_image_spatial'].t()
        
        if 'text_embeds' in outputs and 'spatial_embeds' in outputs:
            outputs['logits_per_text_spatial'] = logit_scale * outputs['text_embeds'] @ outputs['spatial_embeds'].t()
            outputs['logits_per_spatial_text'] = outputs['logits_per_text_spatial'].t()
        
        if return_dict:
            return outputs
        else:
            # 返回元组格式（兼容性）
            return tuple(outputs.values())
    
    def compute_contrastive_loss(self, outputs: Dict) -> Dict:
        """
        计算对比学习损失
        
        Args:
            outputs: 包含各模态嵌入的字典
        
        Returns:
            包含各种损失的字典
        """
        # InfoNCE损失使用FP32计算确保数值稳定性
        with torch.amp.autocast(device_type='cuda', enabled=False):
            losses = {}
            total_loss = 0.0
            
            # 获取batch size
            batch_size = None
            for key in ['image_embeds', 'text_embeds', 'spatial_embeds']:
                if key in outputs:
                    batch_size = outputs[key].shape[0]
                    break
            
            if batch_size is None:
                return losses
            
            # 创建标签（对角线为正样本）
            labels = torch.arange(batch_size, device=next(iter(outputs.values())).device, dtype=torch.long)
            
            logit_scale = self.logit_scale.exp().clamp(min=1.0, max=50.0)  # 更严格限制防止数值溢出
            
            # Image-Text对比损失
            if 'image_embeds' in outputs and 'text_embeds' in outputs:
                # 检查嵌入中是否有NaN
                if torch.isnan(outputs['image_embeds']).any() or torch.isnan(outputs['text_embeds']).any():
                    print("Warning: NaN detected in image/text embeddings, skipping image-text loss")
                else:
                    logits_per_image = logit_scale * outputs['image_embeds'] @ outputs['text_embeds'].t()
                    logits_per_text = logits_per_image.t()
                    
                    # 检查logits中是否有NaN或Inf
                    if torch.isnan(logits_per_image).any() or torch.isinf(logits_per_image).any():
                        print("Warning: NaN/Inf detected in image-text logits, skipping this loss")
                    else:
                        loss_i2t = F.cross_entropy(logits_per_image, labels)
                        loss_t2i = F.cross_entropy(logits_per_text, labels)
                        
                        # 最后检查损失值
                        if torch.isnan(loss_i2t) or torch.isnan(loss_t2i):
                            print("Warning: NaN detected in image-text cross entropy loss")
                        else:
                            it_loss = (loss_i2t + loss_t2i) / 2
                            losses['image_text_loss'] = it_loss
                            total_loss += it_loss
            
            # Image-Spatial对比损失
            if 'image_embeds' in outputs and 'spatial_embeds' in outputs:
                # 检查嵌入中是否有NaN
                if torch.isnan(outputs['image_embeds']).any() or torch.isnan(outputs['spatial_embeds']).any():
                    print("Warning: NaN detected in image/spatial embeddings, skipping image-spatial loss")
                else:
                    logits_per_image = logit_scale * outputs['image_embeds'] @ outputs['spatial_embeds'].t()
                    logits_per_spatial = logits_per_image.t()
                    
                    # 检查logits中是否有NaN或Inf
                    if torch.isnan(logits_per_image).any() or torch.isinf(logits_per_image).any():
                        print("Warning: NaN/Inf detected in image-spatial logits, skipping this loss")
                    else:
                        loss_i2s = F.cross_entropy(logits_per_image, labels)
                        loss_s2i = F.cross_entropy(logits_per_spatial, labels)
                        
                        # 最后检查损失值
                        if torch.isnan(loss_i2s) or torch.isnan(loss_s2i):
                            print("Warning: NaN detected in image-spatial cross entropy loss")
                        else:
                            is_loss = (loss_i2s + loss_s2i) / 2
                            losses['image_spatial_loss'] = is_loss
                            total_loss += is_loss
            
            # Text-Spatial对比损失
            if 'text_embeds' in outputs and 'spatial_embeds' in outputs:
                # 检查嵌入中是否有NaN
                if torch.isnan(outputs['text_embeds']).any() or torch.isnan(outputs['spatial_embeds']).any():
                    print("Warning: NaN detected in text/spatial embeddings, skipping text-spatial loss")
                else:
                    logits_per_text = logit_scale * outputs['text_embeds'] @ outputs['spatial_embeds'].t()
                    logits_per_spatial = logits_per_text.t()
                    
                    # 检查logits中是否有NaN或Inf
                    if torch.isnan(logits_per_text).any() or torch.isinf(logits_per_text).any():
                        print("Warning: NaN/Inf detected in text-spatial logits, skipping this loss")
                    else:
                        loss_t2s = F.cross_entropy(logits_per_text, labels)
                        loss_s2t = F.cross_entropy(logits_per_spatial, labels)
                        
                        # 最后检查损失值
                        if torch.isnan(loss_t2s) or torch.isnan(loss_s2t):
                            print("Warning: NaN detected in text-spatial cross entropy loss")
                        else:
                            ts_loss = (loss_t2s + loss_s2t) / 2
                            losses['text_spatial_loss'] = ts_loss
                            total_loss += ts_loss
            
            losses['total_loss'] = total_loss
            
            # 检查损失值中的NaN
            for loss_name, loss_value in losses.items():
                if torch.isnan(loss_value):
                    print(f"Warning: NaN detected in {loss_name}")
                    losses[loss_name] = torch.tensor(0.0, device=loss_value.device, requires_grad=True)
            
            return losses
    
    def extract_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        提取各模态特征，用于preservation loss计算
        
        Args:
            batch: 输入batch数据
            
        Returns:
            各模态的特征嵌入
        """
        features = {}
        
        # 提取图像特征
        if 'pixel_values' in batch:
            features['image_embeds'] = self.encode_image(batch['pixel_values'])
        
        # 提取文本特征
        if 'input_ids' in batch and 'attention_mask' in batch:
            features['text_embeds'] = self.encode_text(
                batch['input_ids'], 
                batch['attention_mask']
            )
        
        # 提取空间特征
        if all(k in batch for k in ['bboxes', 'anatomy_labels', 'confidences']):
            features['spatial_embeds'] = self.encode_spatial(
                batch['bboxes'],
                batch['anatomy_labels'], 
                batch['confidences'],
                batch.get('spatial_padding_mask'),
                batch.get('image_size', (224, 224))
            )
        
        return features
    
    def get_similarity_scores(
        self,
        image_embeds: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        spatial_embeds: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算各模态之间的相似度分数
        
        Args:
            image_embeds: 图像嵌入
            text_embeds: 文本嵌入  
            spatial_embeds: 空间嵌入
        
        Returns:
            相似度分数字典
        """
        scores = {}
        logit_scale = self.logit_scale.exp()
        
        if image_embeds is not None and text_embeds is not None:
            scores['image_text_similarity'] = logit_scale * image_embeds @ text_embeds.t()
        
        if image_embeds is not None and spatial_embeds is not None:
            scores['image_spatial_similarity'] = logit_scale * image_embeds @ spatial_embeds.t()
        
        if text_embeds is not None and spatial_embeds is not None:
            scores['text_spatial_similarity'] = logit_scale * text_embeds @ spatial_embeds.t()
        
        return scores
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型权重
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # 保存配置
        config = {
            "model_type": "trimodal_clip",
            "projection_dim": self.projection_dim,
            "freeze_clip": self.freeze_clip,
            "spatial_config": self.spatial_encoder.__dict__ if hasattr(self.spatial_encoder, '__dict__') else {}
        }
        
        config_path = os.path.join(save_directory, "config.json") 
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        clip_model_path: str,
        **kwargs
    ):
        """从预训练权重加载模型"""
        
        # 读取配置
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            spatial_config = config.get('spatial_config', {})
            projection_dim = config.get('projection_dim', 768)
            freeze_clip = config.get('freeze_clip', False)
        else:
            spatial_config = kwargs.get('spatial_config', {})
            projection_dim = kwargs.get('projection_dim', 768) 
            freeze_clip = kwargs.get('freeze_clip', False)
        
        # 创建模型
        model = cls(
            clip_model_path=clip_model_path,
            spatial_config=spatial_config,
            projection_dim=projection_dim,
            freeze_clip=freeze_clip
        )
        
        # 加载权重
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {weights_path}")
        
        return model


def load_fetalclip_weights(
    model: TrimodalCLIP,
    fetal_weights_path: str,
    strict: bool = False
) -> Tuple[List[str], List[str]]:
    """
    加载FetalCLIP权重到三模态模型（参考fetalclip_upload.py）
    
    Args:
        model: 三模态CLIP模型
        fetal_weights_path: FetalCLIP权重路径
        strict: 是否严格匹配权重
    
    Returns:
        (missing_keys, unexpected_keys)
    """
    print(f"Loading FetalCLIP weights from {fetal_weights_path}")
    
    # 加载权重
    try:
        if fetal_weights_path.endswith('.bin'):
            # HF格式已转换权重
            state_dict = torch.load(fetal_weights_path, map_location='cpu')
        elif fetal_weights_path.endswith('.pt'):
            # 原始OpenCLIP格式，需要转换
            ckpt = torch.load(fetal_weights_path, map_location='cpu')
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt
            print("Warning: .pt format detected, recommend using converted .bin format")
        else:
            raise ValueError(f"Unsupported weight format: {fetal_weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return [], []
    
    # 过滤权重：只保留CLIP相关的权重，排除spatial_encoder
    clip_state_dict = {}
    model_keys = set(model.state_dict().keys())
    
    for key, value in state_dict.items():
        # 排除我们自己的spatial_encoder权重
        if key.startswith('spatial_encoder.'):
            continue
            
        new_key = key
        
        # 处理键名映射
        if key.startswith('clip_model.'):
            new_key = key
        elif key in ['logit_scale', 'text_projection.weight', 'visual_projection.weight']:
            # 顶层的权重需要加clip_model前缀
            new_key = f'clip_model.{key}'
        elif key.startswith(('text_model.', 'vision_model.')):
            # 文本和视觉模型权重加clip_model前缀
            new_key = f'clip_model.{key}'
        else:
            # 其他权重也尝试加前缀
            new_key = f'clip_model.{key}'
        
        # 特殊处理position_embedding
        if 'position_embedding' in key and not key.endswith('.weight'):
            if f'{new_key}.weight' in model_keys:
                new_key = f'{new_key}.weight'
        
        # 处理位置嵌入的维度问题（参考fetalclip_upload.py）
        if 'position_embedding' in new_key:
            if 'vision_model' in new_key:
                # 视觉位置嵌入：FetalCLIP可能是 [1, 257, 1024] 或其他形状
                if value.dim() == 3:
                    # 移除batch维度：[1, N, C] -> [N, C]
                    value = value.squeeze(0)
                    print(f"Removed batch dimension from vision position embedding: {value.shape}")
                elif value.dim() == 2:
                    # 已经是正确的形状 [N, C]
                    pass
                else:
                    print(f"Warning: unexpected vision position embedding shape: {value.shape}")
            elif 'text_model' in new_key:
                # 文本位置嵌入：FetalCLIP可能是 [117, 768] 形状
                if value.dim() == 2:
                    # 正确的形状 [seq_len, embed_dim]
                    pass
                elif value.dim() == 3 and value.shape[0] == 1:
                    # 移除batch维度：[1, seq_len, embed_dim] -> [seq_len, embed_dim]
                    value = value.squeeze(0)
                    print(f"Removed batch dimension from text position embedding: {value.shape}")
                else:
                    print(f"Warning: unexpected text position embedding shape: {value.shape}")
            
            # 验证形状是否匹配
            if new_key in model_keys:
                model_param = model.state_dict()[new_key]
                if value.shape != model_param.shape:
                    print(f"Position embedding shape mismatch for {new_key}: got {value.shape}, expected {model_param.shape}")
                    # 对于文本位置嵌入，我们已经在初始化时扩展了长度，所以应该匹配
                    # 对于视觉位置嵌入，如果形状不匹配，可能需要插值（这里暂时跳过）
                    if 'text_model' in new_key and value.shape[0] != model_param.shape[0]:
                        print(f"Text position embedding length mismatch, skipping this weight")
                        continue
        
        # 只保留模型实际需要的权重
        if new_key in model_keys:
            clip_state_dict[new_key] = value
    
    # 加载权重到模型
    missing_keys, unexpected_keys = model.load_state_dict(clip_state_dict, strict=strict)
    
    print(f"FetalCLIP weights loaded: {len(clip_state_dict)} parameters")
    print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print(f"  Sample missing keys: {missing_keys[:5]}")
    if unexpected_keys:
        print(f"  Sample unexpected keys: {unexpected_keys[:5]}")
    
    return missing_keys, unexpected_keys


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型（测试模式，使用较小配置）
    clip_path = "/data/qinzhuyuan/0926国庆/clip-vit-large-patch14"
    spatial_config = {
        'embed_dim': 768,
        'num_anatomy_labels': 14,
        'num_transformer_layers': 1,  # 测试时使用更小的配置
        'num_heads': 8,
        'ffn_dim': 1024,
        'dropout': 0.1,
        'max_objects': 20,
        'use_cls_token': True,
        'token_drop_prob': 0.0,  # 测试时不drop
        'confidence_scaling': True
    }
    
    model = TrimodalCLIP(
        clip_model_path=clip_path,
        spatial_config=spatial_config,
        projection_dim=768,
        freeze_clip=False
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 创建测试数据
    batch_size = 2
    
    # 图像数据
    pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 文本数据
    input_ids = torch.randint(0, 1000, (batch_size, 77)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # 空间数据
    num_objects = 10
    bboxes = torch.rand(batch_size, num_objects, 4).to(device)
    anatomy_labels = torch.randint(0, 14, (batch_size, num_objects)).to(device)
    confidences = torch.rand(batch_size, num_objects).to(device)
    spatial_padding_mask = torch.zeros(batch_size, num_objects, dtype=torch.bool).to(device)
    spatial_padding_mask[:, 8:] = True  # 最后两个为padding
    
    # 前向传播测试
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bboxes=bboxes,
            anatomy_labels=anatomy_labels,
            confidences=confidences,
            spatial_padding_mask=spatial_padding_mask,
            return_loss=True
        )
        
        print("Forward pass successful!")
        print(f"Image embeds shape: {outputs['image_embeds'].shape}")
        print(f"Text embeds shape: {outputs['text_embeds'].shape}")
        print(f"Spatial embeds shape: {outputs['spatial_embeds'].shape}")
        print(f"Total loss: {outputs['total_loss'].item():.4f}")
        
        # 检查嵌入是否归一化
        print(f"Image embeds norm: {torch.norm(outputs['image_embeds'], dim=-1).mean():.4f}")
        print(f"Text embeds norm: {torch.norm(outputs['text_embeds'], dim=-1).mean():.4f}")
        print(f"Spatial embeds norm: {torch.norm(outputs['spatial_embeds'], dim=-1).mean():.4f}")


class TrimodalClassifier(nn.Module):
    """
    三模态分类器
    包装TrimodalCLIP并添加分类头
    """
    
    def __init__(
        self,
        trimodal_clip: TrimodalCLIP,
        num_classes: int = 1,
        fusion_type: str = "gated",
        embed_dim: int = 768,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.trimodal_clip = trimodal_clip
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        
        # 分类头
        if fusion_type == "gated":
            # 门控融合：三个模态的加权组合
            self.gate_proj = nn.Linear(embed_dim * 3, 3)
            self.classifier = nn.Linear(embed_dim, num_classes)
        elif fusion_type == "concat":
            # 简单拼接
            self.classifier = nn.Linear(embed_dim * 3, num_classes)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # 冻结backbone
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """冻结trimodal_clip的参数"""
        for param in self.trimodal_clip.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻trimodal_clip的参数"""
        for param in self.trimodal_clip.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        anatomy_labels: Optional[torch.Tensor] = None,
        confidences: Optional[torch.Tensor] = None,
        spatial_padding_mask: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (224, 224),
        return_loss: bool = False,
        return_dict: bool = True
    ) -> Union[Dict, torch.Tensor]:
        """
        前向传播
        
        Returns:
            如果return_dict=True: 包含logits和其他信息的字典
            否则: logits张量
        """
        # 获取三模态嵌入
        outputs = self.trimodal_clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bboxes=bboxes,
            anatomy_labels=anatomy_labels,
            confidences=confidences,
            spatial_padding_mask=spatial_padding_mask,
            image_size=image_size,
            return_loss=return_loss,
            return_dict=True
        )
        
        # 提取嵌入
        image_embeds = outputs.get('image_embeds')
        text_embeds = outputs.get('text_embeds')
        spatial_embeds = outputs.get('spatial_embeds')
        
        # 确保所有嵌入都存在
        if image_embeds is None or text_embeds is None or spatial_embeds is None:
            raise ValueError("Missing embeddings from trimodal_clip")
        
        # 融合嵌入
        if self.fusion_type == "gated":
            # 门控融合
            concat_embeds = torch.cat([image_embeds, text_embeds, spatial_embeds], dim=-1)
            gates = torch.softmax(self.gate_proj(concat_embeds), dim=-1)
            
            # 加权组合
            fused_embeds = (
                gates[:, 0:1] * image_embeds +
                gates[:, 1:2] * text_embeds +
                gates[:, 2:3] * spatial_embeds
            )
        elif self.fusion_type == "concat":
            # 简单拼接
            fused_embeds = torch.cat([image_embeds, text_embeds, spatial_embeds], dim=-1)
        
        # 分类
        logits = self.classifier(fused_embeds)
        
        # 构建输出
        result = outputs.copy() if isinstance(outputs, dict) else {}
        result['logits'] = logits
        result['fused_embeds'] = fused_embeds
        
        if self.fusion_type == "gated":
            result['fusion_gates'] = gates
        
        if return_dict:
            return result
        else:
            return logits
    
    def extract_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """提取特征，用于preservation loss计算"""
        return self.trimodal_clip.extract_features(batch)
    
    def compute_contrastive_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算对比学习损失"""
        return self.trimodal_clip.compute_contrastive_loss(outputs)