"""
评估脚本
在测试集上评估模型性能，输出分类指标和对齐健康度
"""
import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from trimodal_clip import TrimodalCLIP, TrimodalClassifier
from dataset import create_dataloaders
from transformers import CLIPProcessor


class TrimodalEvaluator:
    """三模态模型评估器"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        clip_model_path: str,
        test_json: str,
        batch_size: int = 16,
        device: str = "cuda"
    ):
        self.device = torch.device(device)
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载processor
        self.processor = CLIPProcessor.from_pretrained(
            clip_model_path,
            local_files_only=True
        )
        
        # 创建测试数据加载器
        _, _, self.test_loader = create_dataloaders(
            train_json=None,
            val_json=None,
            test_json=test_json,
            processor=self.processor,
            batch_size=batch_size,
            max_objects=self.config.get('max_objects', 32),
            max_length=self.config.get('max_length', 117),
            image_size=tuple(self.config.get('image_size', [224, 224])),
            num_workers=4,
            pin_memory=True
        )
        
        # 创建模型
        self._create_model(clip_model_path)
        
        # 加载权重
        self._load_checkpoint(model_path)
        
        print(f"Evaluator initialized with {len(self.test_loader)} test batches")
    
    def _create_model(self, clip_model_path: str):
        """创建模型"""
        # 空间编码器配置
        spatial_config = {
            'embed_dim': 768,
            'num_anatomy_labels': 14,
            'num_transformer_layers': 2,
            'num_heads': 8,
            'ffn_dim': 2048,
            'dropout': 0.1,
            'max_objects': self.config.get('max_objects', 32),
            'use_cls_token': True,
            'token_drop_prob': 0.0,  # 评估时不使用dropout
            'confidence_scaling': True
        }
        
        # 直接创建三模态CLIP
        trimodal_clip = TrimodalCLIP(
            clip_model_path=clip_model_path,
            spatial_config=spatial_config,
            projection_dim=768,
            freeze_backbone=False  # 评估时不冻结
        )
        
        # 创建分类模型
        self.model = TrimodalClassifier(
            trimodal_clip=trimodal_clip,
            num_classes=1,
            fusion_type=self.config.get('fusion_type', 'gated'),
            embed_dim=768,
            freeze_backbone=False
        ).to(self.device)
    
    def _load_checkpoint(self, model_path: str):
        """加载模型权重"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
        
        print(f"Model loaded from {model_path}")
    
    def evaluate_classification(self, save_plots: bool = True, output_dir: str = "./eval_results") -> Dict[str, Any]:
        """评估分类性能"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_logits = []
        all_gate_weights = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating classification"):
                # 移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 预测
                pred_outputs = self.model.predict(batch)
                
                all_preds.extend(pred_outputs['predictions'].cpu().numpy())
                all_probs.extend(pred_outputs['probabilities'].cpu().numpy())
                all_logits.extend(pred_outputs['logits'].squeeze(-1).cpu().numpy())
                all_gate_weights.append(pred_outputs['gate_weights'].cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        # 精确率-召回率曲线
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        pr_auc = np.trapz(precision, recall)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        # 分类报告
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        
        # 门控权重统计
        gate_weights = np.concatenate(all_gate_weights, axis=0)  # [N, 3]
        gate_stats = {
            'mean_weights': gate_weights.mean(axis=0).tolist(),
            'std_weights': gate_weights.std(axis=0).tolist(),
            'weight_names': ['image', 'text', 'spatial']
        }
        
        results = {
            'accuracy': acc,
            'f1_score': f1,
            'auc_roc': auc,
            'auc_pr': pr_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'gate_weights_stats': gate_stats,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'gate_weights': gate_weights.tolist()
        }
        
        # 保存图表
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            self._plot_confusion_matrix(cm, os.path.join(output_dir, "confusion_matrix.png"))
            self._plot_pr_curve(all_labels, all_probs, os.path.join(output_dir, "pr_curve.png"))
            self._plot_gate_weights(gate_weights, os.path.join(output_dir, "gate_weights.png"))
        
        return results
    
    def evaluate_alignment(self, save_plots: bool = True, output_dir: str = "./eval_results") -> Dict[str, Any]:
        """评估模态对齐健康度"""
        self.model.eval()
        
        all_similarities = {
            'image_text': [],
            'image_spatial': [],
            'text_spatial': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating alignment"):
                # 移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 提取特征
                features = self.model.extract_features(batch)
                
                # 计算相似度
                if 'image_embeds' in features and 'text_embeds' in features:
                    sim_matrix = features['image_embeds'] @ features['text_embeds'].t()
                    # 取对角线（同一样本的相似度）
                    diagonal_sim = torch.diag(sim_matrix)
                    all_similarities['image_text'].extend(diagonal_sim.cpu().numpy())
                
                if 'image_embeds' in features and 'spatial_embeds' in features:
                    sim_matrix = features['image_embeds'] @ features['spatial_embeds'].t()
                    diagonal_sim = torch.diag(sim_matrix)
                    all_similarities['image_spatial'].extend(diagonal_sim.cpu().numpy())
                
                if 'text_embeds' in features and 'spatial_embeds' in features:
                    sim_matrix = features['text_embeds'] @ features['spatial_embeds'].t()
                    diagonal_sim = torch.diag(sim_matrix)
                    all_similarities['text_spatial'].extend(diagonal_sim.cpu().numpy())
        
        # 计算统计指标
        alignment_stats = {}
        for modality_pair, similarities in all_similarities.items():
            if similarities:
                similarities = np.array(similarities)
                alignment_stats[modality_pair] = {
                    'mean': float(similarities.mean()),
                    'std': float(similarities.std()),
                    'median': float(np.median(similarities)),
                    'min': float(similarities.min()),
                    'max': float(similarities.max())
                }
        
        # 保存图表
        if save_plots and alignment_stats:
            os.makedirs(output_dir, exist_ok=True)
            self._plot_similarity_distributions(all_similarities, os.path.join(output_dir, "similarity_distributions.png"))
        
        return {
            'alignment_stats': alignment_stats,
            'similarities': all_similarities
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-standard', 'Standard'],
                    yticklabels=['Non-standard', 'Standard'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, labels: List[int], probs: List[float], save_path: str):
        """绘制精确率-召回率曲线"""
        precision, recall, _ = precision_recall_curve(labels, probs)
        auc_pr = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {auc_pr:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gate_weights(self, gate_weights: np.ndarray, save_path: str):
        """绘制门控权重分布"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 权重分布直方图
        modality_names = ['Image', 'Text', 'Spatial']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (name, color) in enumerate(zip(modality_names, colors)):
            axes[0].hist(gate_weights[:, i], bins=50, alpha=0.7, label=name, color=color)
        
        axes[0].set_xlabel('Gate Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Gate Weight Distributions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 平均权重柱状图
        mean_weights = gate_weights.mean(axis=0)
        std_weights = gate_weights.std(axis=0)
        
        x_pos = np.arange(len(modality_names))
        axes[1].bar(x_pos, mean_weights, yerr=std_weights, capsize=5, 
                    color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Modality')
        axes[1].set_ylabel('Average Gate Weight')
        axes[1].set_title('Average Gate Weights')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(modality_names)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_similarity_distributions(self, similarities: Dict[str, List[float]], save_path: str):
        """绘制相似度分布"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        pairs = ['image_text', 'image_spatial', 'text_spatial']
        titles = ['Image-Text', 'Image-Spatial', 'Text-Spatial']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (pair, title, color) in enumerate(zip(pairs, titles, colors)):
            if similarities[pair]:
                sims = np.array(similarities[pair])
                axes[i].hist(sims, bins=50, alpha=0.7, color=color, edgecolor='black')
                axes[i].axvline(sims.mean(), color='red', linestyle='--', 
                               label=f'Mean: {sims.mean():.3f}')
                axes[i].set_xlabel('Cosine Similarity')
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{title} Similarity Distribution')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_evaluation(self, output_dir: str = "./eval_results") -> Dict[str, Any]:
        """运行完整评估"""
        print("Running full evaluation...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 分类评估
        print("\n1. Evaluating classification performance...")
        classification_results = self.evaluate_classification(save_plots=True, output_dir=output_dir)
        
        # 对齐评估
        print("\n2. Evaluating modality alignment...")
        alignment_results = self.evaluate_alignment(save_plots=True, output_dir=output_dir)
        
        # 合并结果
        full_results = {
            'classification': classification_results,
            'alignment': alignment_results
        }
        
        # 保存结果
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            # 移除不能序列化的部分
            serializable_results = {
                'classification': {
                    'accuracy': classification_results['accuracy'],
                    'f1_score': classification_results['f1_score'],
                    'auc_roc': classification_results['auc_roc'],
                    'auc_pr': classification_results['auc_pr'],
                    'confusion_matrix': classification_results['confusion_matrix'],
                    'gate_weights_stats': classification_results['gate_weights_stats']
                },
                'alignment': alignment_results['alignment_stats']
            }
            json.dump(serializable_results, f, indent=2)
        
        # 打印摘要
        self._print_summary(classification_results, alignment_results)
        
        print(f"\nEvaluation completed! Results saved to {output_dir}")
        return full_results
    
    def _print_summary(self, classification_results: Dict, alignment_results: Dict):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # 分类指标
        print("\nClassification Metrics:")
        print(f"  Accuracy: {classification_results['accuracy']:.4f}")
        print(f"  F1 Score: {classification_results['f1_score']:.4f}")
        print(f"  AUC-ROC: {classification_results['auc_roc']:.4f}")
        print(f"  AUC-PR:  {classification_results['auc_pr']:.4f}")
        
        # 门控权重
        gate_stats = classification_results['gate_weights_stats']
        print("\nGate Weight Statistics:")
        for i, name in enumerate(gate_stats['weight_names']):
            mean_w = gate_stats['mean_weights'][i]
            std_w = gate_stats['std_weights'][i]
            print(f"  {name:8s}: {mean_w:.3f} ± {std_w:.3f}")
        
        # 对齐健康度
        if 'alignment_stats' in alignment_results:
            print("\nModality Alignment Health:")
            for pair, stats in alignment_results['alignment_stats'].items():
                print(f"  {pair:15s}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="三模态CLIP模型评估")
    
    parser.add_argument("--model_path", required=True, help="模型权重路径")
    parser.add_argument("--config_path", required=True, help="配置文件路径")
    parser.add_argument("--clip_model_path", default="/data/qinzhuyuan/0926国庆/clip-vit-large-patch14")
    parser.add_argument("--test_json", default="/data/qinzhuyuan/0926国庆/regenerated_test_data.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = TrimodalEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        clip_model_path=args.clip_model_path,
        test_json=args.test_json,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 运行评估
    results = evaluator.run_full_evaluation(output_dir=args.output_dir)
    
    return results


if __name__ == "__main__":
    main()