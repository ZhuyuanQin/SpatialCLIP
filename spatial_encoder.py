"""
增强型空间编码器 (Enhanced Spatial Encoder)
用于处理目标检测框的几何和语义信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict


class SpatialEncoder(nn.Module):
    """
    增强型空间编码器，处理目标集合的空间和语义信息
    """
    
    # 解剖标签映射
    ANATOMY_LABELS = {
        'CB': 0, 'CP1': 1, 'CP2': 2, 'CF': 3, 'NT': 4, 'NB': 5,
        'F': 6, 'NA': 7, 'CRL': 8, 'S': 9, 'AS': 10, 'UC': 11,
        'UI': 12, 'UNKNOWN': 13
    }
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_anatomy_labels: int = 14,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_objects: int = 100,
        use_cls_token: bool = True,
        token_drop_prob: float = 0.1,
        confidence_scaling: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_objects = max_objects
        self.use_cls_token = use_cls_token
        self.token_drop_prob = token_drop_prob
        self.confidence_scaling = confidence_scaling
        
        # 解剖标签嵌入
        self.anatomy_embedding = nn.Embedding(num_anatomy_labels, embed_dim // 4)
        
        # 几何特征维度：4(bbox) + 1(log_area) + 1(log_aspect) + 3(global_pos) + 3(nearest_neighbor) + 1(conf) = 13
        self.geometry_dim = 13
        
        # 几何特征投影
        self.geometry_proj = nn.Linear(self.geometry_dim, embed_dim - embed_dim // 4)
        
        # 特征融合MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # CLS token (可学习的全局池化)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def compute_geometry_features(
        self, 
        bboxes: torch.Tensor, 
        confidences: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        计算几何特征
        
        Args:
            bboxes: [batch_size, num_objects, 4] 归一化的边界框 (x, y, w, h)
            confidences: [batch_size, num_objects] 检测置信度
            image_size: 图像尺寸 (height, width)
        
        Returns:
            geometry_features: [batch_size, num_objects, 13] 几何特征
        """
        batch_size, num_objects, _ = bboxes.shape
        
        # 1. 基础几何特征 (x, y, w, h)
        x, y, w, h = bboxes.split(1, dim=-1)  # 每个都是 [B, N, 1]
        
        # 2. 面积和宽高比
        area = w * h
        aspect_ratio = w / (h + 1e-8)
        log_area = torch.log(area + 1e-8)
        log_aspect = torch.log(aspect_ratio + 1e-8)
        
        # 3. 到图像中心的全局相对位置
        center_x = x + w / 2
        center_y = y + h / 2
        
        # 相对于图像中心的距离和角度
        dx = center_x - 0.5  # 图像中心在 (0.5, 0.5)
        dy = center_y - 0.5
        
        r = torch.sqrt(dx ** 2 + dy ** 2)
        theta = torch.atan2(dy, dx)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # 4. 最近邻关系特征
        # 计算所有目标中心点之间的距离
        centers = torch.stack([center_x.squeeze(-1), center_y.squeeze(-1)], dim=-1)  # [B, N, 2]
        
        # 计算距离矩阵
        centers_expand1 = centers.unsqueeze(2)  # [B, N, 1, 2]
        centers_expand2 = centers.unsqueeze(1)  # [B, 1, N, 2]
        dist_matrix = torch.norm(centers_expand1 - centers_expand2, dim=-1)  # [B, N, N]
        
        # 对角线设为无穷大，避免选择自己
        dist_matrix = dist_matrix + torch.eye(num_objects, device=dist_matrix.device).unsqueeze(0) * 1e6
        
        # 找到最近邻
        min_dist, nearest_idx = torch.min(dist_matrix, dim=-1)  # [B, N]
        
        # 计算到最近邻的方向
        nearest_centers = torch.gather(centers, 1, nearest_idx.unsqueeze(-1).expand(-1, -1, 2))  # [B, N, 2]
        nn_dx = nearest_centers[:, :, 0:1] - center_x
        nn_dy = nearest_centers[:, :, 1:2] - center_y
        nn_phi = torch.atan2(nn_dy, nn_dx)
        nn_sin_phi = torch.sin(nn_phi)
        nn_cos_phi = torch.cos(nn_phi)
        
        # 归一化最近邻距离
        min_dist = min_dist.unsqueeze(-1)  # [B, N, 1]
        
        # 拼接所有几何特征（包含置信度）
        geometry_features = torch.cat([
            x, y, w, h,                    # 4维：基础bbox
            log_area, log_aspect,          # 2维：面积和宽高比
            r, sin_theta, cos_theta,       # 3维：全局位置
            min_dist, nn_sin_phi, nn_cos_phi,  # 3维：最近邻关系
            confidences.unsqueeze(-1)      # 1维：检测置信度
        ], dim=-1)
        
        return geometry_features
    
    def forward(
        self,
        bboxes: torch.Tensor,
        anatomy_labels: torch.Tensor,
        confidences: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            bboxes: [batch_size, num_objects, 4] 归一化边界框
            anatomy_labels: [batch_size, num_objects] 解剖标签索引
            confidences: [batch_size, num_objects] 检测置信度
            padding_mask: [batch_size, num_objects] True表示padding位置
            image_size: 图像尺寸
        
        Returns:
            spatial_embeddings: [batch_size, embed_dim] L2归一化的空间嵌入
        """
        batch_size, num_objects = bboxes.shape[:2]
        
        # 1. 计算几何特征
        geometry_features = self.compute_geometry_features(bboxes, confidences, image_size)  # [B, N, 13]
        
        # 2. 计算解剖标签嵌入
        anatomy_embeds = self.anatomy_embedding(anatomy_labels)  # [B, N, embed_dim//4]
        
        # 3. 投影几何特征
        geometry_embeds = self.geometry_proj(geometry_features)  # [B, N, 3*embed_dim//4]
        
        # 4. 拼接几何和语义特征
        token_features = torch.cat([geometry_embeds, anatomy_embeds], dim=-1)  # [B, N, embed_dim]
        
        # 5. 通过MLP融合特征
        token_features = self.feature_mlp(token_features)  # [B, N, embed_dim]
        
        # 6. Token dropping（训练时）
        if self.training and self.token_drop_prob > 0:
            # 基于置信度的token dropping
            drop_mask = torch.rand_like(confidences) < self.token_drop_prob
            low_conf_mask = confidences < 0.5  # 低置信度更容易被drop
            drop_mask = drop_mask | low_conf_mask
            
            if padding_mask is not None:
                drop_mask = drop_mask & ~padding_mask  # 不drop padding tokens
            
            # 将被drop的token置零
            token_features = token_features * (~drop_mask).unsqueeze(-1).float()
        
        # 7. 置信度缩放
        if self.confidence_scaling:
            conf_scale = confidences.unsqueeze(-1)  # [B, N, 1]
            token_features = token_features * conf_scale
        
        # 8. 添加CLS token（如果使用）
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
            token_features = torch.cat([cls_tokens, token_features], dim=1)  # [B, N+1, embed_dim]
            
            # 更新padding mask
            if padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        # 9. 随机打乱顺序（训练时）
        if self.training:
            # 为每个batch生成随机排列索引
            perm_indices = torch.stack([
                torch.randperm(token_features.size(1), device=token_features.device) 
                for _ in range(batch_size)
            ])
            
            # 应用排列
            batch_indices = torch.arange(batch_size, device=token_features.device).unsqueeze(1)
            token_features = token_features[batch_indices, perm_indices]
            
            if padding_mask is not None:
                padding_mask = padding_mask[batch_indices, perm_indices]
        
        # 10. Transformer编码
        if padding_mask is not None:
            # Transformer期望的mask：True表示需要被忽略的位置
            src_key_padding_mask = padding_mask
        else:
            src_key_padding_mask = None
        
        encoded_features = self.transformer(
            token_features, 
            src_key_padding_mask=src_key_padding_mask
        )  # [B, N+1, embed_dim] 或 [B, N, embed_dim]
        
        # 11. 池化得到最终表示
        if self.use_cls_token:
            # 使用CLS token作为全局表示
            spatial_embedding = encoded_features[:, 0]  # [B, embed_dim]
        else:
            # Mask-aware平均池化
            if padding_mask is not None:
                # 只对非padding位置计算平均
                valid_mask = ~padding_mask  # [B, N]
                valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
                valid_counts = torch.clamp(valid_counts, min=1.0)  # 避免除零
                
                masked_features = encoded_features * valid_mask.unsqueeze(-1).float()
                spatial_embedding = masked_features.sum(dim=1) / valid_counts  # [B, embed_dim]
            else:
                spatial_embedding = encoded_features.mean(dim=1)  # [B, embed_dim]
        
        # 12. 输出归一化
        spatial_embedding = self.output_norm(spatial_embedding)
        
        # 13. L2归一化
        spatial_embedding = F.normalize(spatial_embedding, p=2, dim=-1)
        
        return spatial_embedding


def create_spatial_encoder(config: Dict) -> SpatialEncoder:
    """
    创建空间编码器的工厂函数
    
    Args:
        config: 配置字典
    
    Returns:
        SpatialEncoder实例
    """
    return SpatialEncoder(
        embed_dim=config.get('embed_dim', 768),
        num_anatomy_labels=config.get('num_anatomy_labels', 14),
        num_transformer_layers=config.get('num_transformer_layers', 2),
        num_heads=config.get('num_heads', 8),
        ffn_dim=config.get('ffn_dim', 2048),
        dropout=config.get('dropout', 0.1),
        max_objects=config.get('max_objects', 100),
        use_cls_token=config.get('use_cls_token', True),
        token_drop_prob=config.get('token_drop_prob', 0.1),
        confidence_scaling=config.get('confidence_scaling', True)
    )


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = SpatialEncoder().to(device)
    
    # 创建测试数据
    batch_size, num_objects = 2, 10
    bboxes = torch.rand(batch_size, num_objects, 4).to(device)
    anatomy_labels = torch.randint(0, 14, (batch_size, num_objects)).to(device)
    confidences = torch.rand(batch_size, num_objects).to(device)
    padding_mask = torch.zeros(batch_size, num_objects, dtype=torch.bool).to(device)
    padding_mask[:, 8:] = True  # 最后两个为padding
    
    # 前向传播
    with torch.no_grad():
        output = model(bboxes, anatomy_labels, confidences, padding_mask)
        print(f"Output shape: {output.shape}")
        print(f"Output norm: {torch.norm(output, dim=-1)}")