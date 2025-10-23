"""
数据处理模块
包含Dataset、DataLoader和图像预处理
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPTokenizer
import torchvision.transforms as transforms

from spatial_encoder import SpatialEncoder


class LetterboxTransform:
    """Letterbox变换，保持纵横比并同步变换bbox"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def __call__(self, image: Image.Image, bboxes: Optional[List[List[float]]] = None) -> Tuple[Image.Image, Optional[np.ndarray], Dict]:
        """
        对图像进行letterbox变换，同时同步变换bbox
        
        Args:
            image: PIL图像
            bboxes: bbox列表，格式为[[x1,y1,x2,y2], ...]
            
        Returns:
            transformed_image: 变换后的图像
            transformed_bboxes: 变换后的bbox，归一化到[0,1]
            transform_info: 变换信息
        """
        orig_w, orig_h = image.size
        target_w, target_h = self.target_size
        
        # 计算缩放比例
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # 计算padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # resize图像
        image_resized = image.resize((new_w, new_h), Image.LANCZOS)
        
        # 创建目标图像并粘贴
        target_image = Image.new('RGB', self.target_size, (128, 128, 128))  # 灰色填充
        target_image.paste(image_resized, (pad_x, pad_y))
        
        # 变换bbox
        transformed_bboxes = None
        if bboxes is not None and len(bboxes) > 0:
            bboxes_array = np.array(bboxes)
            
            # 应用缩放和平移
            bboxes_array[:, [0, 2]] = bboxes_array[:, [0, 2]] * scale + pad_x  # x坐标
            bboxes_array[:, [1, 3]] = bboxes_array[:, [1, 3]] * scale + pad_y  # y坐标
            
            # 归一化到[0,1]
            bboxes_array[:, [0, 2]] /= target_w  # x坐标归一化
            bboxes_array[:, [1, 3]] /= target_h  # y坐标归一化
            
            # 限制在[0,1]范围内
            bboxes_array = np.clip(bboxes_array, 0, 1)
            
            # 转换为(x,y,w,h)格式
            transformed_bboxes = np.zeros_like(bboxes_array)
            transformed_bboxes[:, 0] = bboxes_array[:, 0]  # x
            transformed_bboxes[:, 1] = bboxes_array[:, 1]  # y
            transformed_bboxes[:, 2] = bboxes_array[:, 2] - bboxes_array[:, 0]  # w
            transformed_bboxes[:, 3] = bboxes_array[:, 3] - bboxes_array[:, 1]  # h
        
        transform_info = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'orig_size': (orig_w, orig_h),
            'target_size': self.target_size
        }
        
        return target_image, transformed_bboxes, transform_info


class TrimodalDataset(Dataset):
    """三模态数据集"""
    
    def __init__(
        self,
        json_path: str,
        processor: CLIPProcessor,
        max_objects: int = 32,
        max_length: int = 117,
        image_size: Tuple[int, int] = (224, 224),
        anatomy_labels: Optional[Dict[str, int]] = None,
        augment: bool = False
    ):
        self.json_path = json_path
        self.processor = processor
        self.max_objects = max_objects
        self.max_length = max_length
        self.image_size = image_size
        self.augment = augment
        
        # 解剖标签映射
        if anatomy_labels is None:
            self.anatomy_labels = {
                'CB': 0, 'CP1': 1, 'CP2': 2, 'CF': 3, 'NT': 4, 'NB': 5,
                'F': 6, 'NA': 7, 'CRL': 8, 'S': 9, 'AS': 10, 'UC': 11,
                'UI': 12, 'UNKNOWN': 13
            }
        else:
            self.anatomy_labels = anatomy_labels
        
        # 加载数据
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Letterbox变换
        self.letterbox = LetterboxTransform(image_size)
        
        # 数据增强
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomHorizontalFlip(p=0.1),  # 医学图像谨慎使用翻转
            ])
        else:
            self.augment_transform = None
        
        print(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def process_coordinates(self, coordinates: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        处理坐标信息
        
        Returns:
            bboxes: [N, 4] 边界框 (x,y,w,h)
            labels: [N] 解剖标签索引
            confidences: [N] 置信度
            valid_mask: [N] 有效性掩码
        """
        if not coordinates:
            # 空坐标，返回空数组
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=bool)
            )
        
        bboxes = []
        labels = []
        confidences = []
        
        for coord in coordinates:
            # 提取bbox (x1,y1,x2,y2)
            bbox = coord['bbox']
            if len(bbox) != 4:
                continue
                
            bboxes.append(bbox)
            
            # 提取标签
            label = coord.get('label', 'UNKNOWN')
            label_idx = self.anatomy_labels.get(label, self.anatomy_labels['UNKNOWN'])
            labels.append(label_idx)
            
            # 提取置信度
            conf = coord.get('confidence', 1.0)
            confidences.append(conf)
        
        if not bboxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=bool)
            )
        
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        confidences = np.array(confidences, dtype=np.float32)
        valid_mask = np.ones(len(bboxes), dtype=bool)
        
        return bboxes, labels, confidences, valid_mask
    
    def pad_sequences(
        self, 
        bboxes: np.ndarray, 
        labels: np.ndarray, 
        confidences: np.ndarray, 
        valid_mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        填充序列到max_objects长度
        
        Returns:
            padded_bboxes: [max_objects, 4]
            padded_labels: [max_objects]
            padded_confidences: [max_objects]
            padding_mask: [max_objects] True表示padding位置
        """
        num_objects = len(bboxes)
        
        # 创建填充后的数组
        padded_bboxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        padded_labels = np.zeros(self.max_objects, dtype=np.int64)
        padded_confidences = np.zeros(self.max_objects, dtype=np.float32)
        padding_mask = np.ones(self.max_objects, dtype=bool)  # True表示padding
        
        if num_objects > 0:
            # 如果目标数量超过max_objects，截断
            actual_num = min(num_objects, self.max_objects)
            
            padded_bboxes[:actual_num] = bboxes[:actual_num]
            padded_labels[:actual_num] = labels[:actual_num]
            padded_confidences[:actual_num] = confidences[:actual_num]
            padding_mask[:actual_num] = False  # 有效位置设为False
        
        return (
            torch.tensor(padded_bboxes, dtype=torch.float32),
            torch.tensor(padded_labels, dtype=torch.long),
            torch.tensor(padded_confidences, dtype=torch.float32),
            torch.tensor(padding_mask, dtype=torch.bool)
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        try:
            # 加载图像
            image_path = item['image_path']
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            # 处理坐标
            coordinates = item.get('coordinates', [])
            bboxes, labels, confidences, valid_mask = self.process_coordinates(coordinates)
            
            # Letterbox变换
            transformed_image, transformed_bboxes, transform_info = self.letterbox(
                image, bboxes.tolist() if len(bboxes) > 0 else None
            )
            
            # 数据增强
            if self.augment_transform and self.augment:
                transformed_image = self.augment_transform(transformed_image)
            
            # 处理文本
            description = item.get('description', '')
            text_inputs = self.processor.tokenizer(
                description,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 处理图像（使用processor）
            image_inputs = self.processor.image_processor(
                transformed_image,
                return_tensors="pt"
            )
            
            # 标签
            class_label = item.get('class_label', 'standard')
            binary_label = 1 if class_label == 'standard' else 0
            
            # 处理空间数据
            if transformed_bboxes is not None:
                padded_bboxes, padded_labels, padded_confidences, padding_mask = self.pad_sequences(
                    transformed_bboxes, labels, confidences, valid_mask
                )
            else:
                # 空的空间数据
                padded_bboxes, padded_labels, padded_confidences, padding_mask = self.pad_sequences(
                    np.zeros((0, 4)), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
                )
            
            # 计算质量提示
            num_objects = len(bboxes) if len(bboxes) > 0 else 0
            avg_confidence = confidences.mean() if len(confidences) > 0 else 0.0
            
            return {
                # 图像数据
                'pixel_values': image_inputs['pixel_values'].squeeze(0),  # [3, H, W]
                
                # 文本数据
                'input_ids': text_inputs['input_ids'].squeeze(0),  # [seq_len]
                'attention_mask': text_inputs['attention_mask'].squeeze(0),  # [seq_len]
                
                # 空间数据
                'bboxes': padded_bboxes,  # [max_objects, 4]
                'anatomy_labels': padded_labels,  # [max_objects]
                'confidences': padded_confidences,  # [max_objects]
                'spatial_padding_mask': padding_mask,  # [max_objects]
                
                # 质量提示
                'num_objects': torch.tensor(num_objects, dtype=torch.long),
                'avg_confidence': torch.tensor(avg_confidence, dtype=torch.float32),
                
                # 标签
                'labels': torch.tensor(binary_label, dtype=torch.long),
                'class_label': class_label,
                
                # 元信息
                'image_path': image_path,
                'description': description,
                'image_size': self.image_size,
                'transform_info': transform_info
            }
            
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # 返回空数据
            return self._get_empty_item()
    
    def _get_empty_item(self) -> Dict[str, Any]:
        """返回空的数据项"""
        return {
            'pixel_values': torch.zeros(3, *self.image_size, dtype=torch.float32),
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'bboxes': torch.zeros(self.max_objects, 4, dtype=torch.float32),
            'anatomy_labels': torch.zeros(self.max_objects, dtype=torch.long),
            'confidences': torch.zeros(self.max_objects, dtype=torch.float32),
            'spatial_padding_mask': torch.ones(self.max_objects, dtype=torch.bool),
            'num_objects': torch.tensor(0, dtype=torch.long),
            'avg_confidence': torch.tensor(0.0, dtype=torch.float32),
            'labels': torch.tensor(0, dtype=torch.long),
            'class_label': 'unknown',
            'image_path': '',
            'description': '',
            'image_size': self.image_size,
            'transform_info': {}
        }


class ModalityDropout(torch.nn.Module):
    """模态dropout，用于提升鲁棒性"""
    
    def __init__(
        self,
        image_dropout_prob: float = 0.1,
        text_dropout_prob: float = 0.1,
        spatial_dropout_prob: float = 0.1
    ):
        super().__init__()
        self.image_dropout_prob = image_dropout_prob
        self.text_dropout_prob = text_dropout_prob
        self.spatial_dropout_prob = spatial_dropout_prob
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用模态dropout"""
        if not self.training:
            return batch
        
        batch_size = batch['pixel_values'].shape[0]
        device = batch['pixel_values'].device
        
        # 图像dropout
        if self.image_dropout_prob > 0:
            image_mask = torch.rand(batch_size, device=device) > self.image_dropout_prob
            batch['pixel_values'] = batch['pixel_values'] * image_mask.view(-1, 1, 1, 1)
        
        # 文本dropout - 保留至少一个有效token
        if self.text_dropout_prob > 0:
            text_mask = torch.rand(batch_size, device=device) > self.text_dropout_prob
            
            # 对于需要dropout的样本，随机保留部分tokens而不是全部清零
            for i in range(batch_size):
                if not text_mask[i]:  # 这个样本需要dropout
                    # 获取原始有效token数量
                    valid_tokens = batch['attention_mask'][i].sum().item()
                    if valid_tokens > 1:
                        # 随机保留25-75%的tokens，但至少保留1个
                        keep_ratio = torch.rand(1).item() * 0.5 + 0.25  # 0.25-0.75
                        keep_count = max(1, int(valid_tokens * keep_ratio))
                        
                        # 找到有效token的位置
                        valid_positions = torch.where(batch['attention_mask'][i] == 1)[0]
                        if len(valid_positions) > keep_count:
                            # 随机选择要保留的位置
                            keep_positions = valid_positions[torch.randperm(len(valid_positions))[:keep_count]]
                            
                            # 创建新的mask，只保留选中的位置
                            new_attention_mask = torch.zeros_like(batch['attention_mask'][i])
                            new_attention_mask[keep_positions] = 1
                            batch['attention_mask'][i] = new_attention_mask
                            
                            # 同时清零未保留位置的input_ids
                            new_input_ids = batch['input_ids'][i].clone()
                            new_input_ids[~new_attention_mask.bool()] = 0
                            batch['input_ids'][i] = new_input_ids
        
        # 双重保险：检查并修复全0 attention_mask
        for i in range(batch_size):
            if batch['attention_mask'][i].sum().item() == 0:
                # 强制保留第一个token（通常是[CLS]或类似token）
                batch['attention_mask'][i][0] = 1
                if batch['input_ids'][i][0].item() == 0:
                    batch['input_ids'][i][0] = 49406  # CLIP的默认起始token
        
        # 空间dropout
        if self.spatial_dropout_prob > 0:
            spatial_mask = torch.rand(batch_size, device=device) > self.spatial_dropout_prob
            batch['bboxes'] = batch['bboxes'] * spatial_mask.view(-1, 1, 1)
            batch['confidences'] = batch['confidences'] * spatial_mask.view(-1, 1)
            # 更新padding mask
            batch['spatial_padding_mask'] = batch['spatial_padding_mask'] | ~spatial_mask.view(-1, 1)
        
        return batch


def create_dataloaders(
    train_json: str,
    val_json: str,
    test_json: str,
    processor: CLIPProcessor,
    batch_size: int = 16,
    max_objects: int = 32,
    max_length: int = 117,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 创建数据集
    train_dataset = TrimodalDataset(
        train_json, processor, max_objects, max_length, image_size, augment=True
    )
    val_dataset = TrimodalDataset(
        val_json, processor, max_objects, max_length, image_size, augment=False
    )
    test_dataset = TrimodalDataset(
        test_json, processor, max_objects, max_length, image_size, augment=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载
    from transformers import CLIPProcessor
    
    # 加载processor
    processor = CLIPProcessor.from_pretrained(
        "/data/qinzhuyuan/0926国庆/clip-vit-large-patch14",
        local_files_only=True
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_json="/data/qinzhuyuan/0926国庆/regenerated_train_data.json",
        val_json="/data/qinzhuyuan/0926国庆/regenerated_valid_data.json",
        test_json="/data/qinzhuyuan/0926国庆/regenerated_test_data.json",
        processor=processor,
        batch_size=4,
        max_objects=32,
        max_length=117
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 测试一个batch
    for batch in train_loader:
        print("Batch keys:", list(batch.keys()))
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape} {value.dtype}")
            else:
                print(f"{key}: {type(value)}")
        break