# -*- coding: utf-8 -*-
"""
DLC Fine-tuning Script (Transfer Learning) - v5 Linear Probing
==========================================
核心策略:
1. 【死锁】骨干网络 + 超图网络 (Hypergraph) -> 防止过拟合/特征扭曲
2. 【仅训】预测头 (Outcome Heads) -> 快速适配 LUAD 标签
3. 【提速】学习率提升至 1e-3 -> 线性层需要更大的步长
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.dlc.run_final_sota import load_data
from src.dlc.dlc_net import DLCNet

def main():
    print("\n" + "="*60)
    print("DLC EXP4: Transfer Learning (Linear Probing Mode)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    # 1. 配置路径
    pretrained_path = 'results/dlc_final_sota.pth'
    output_dir = Path('results/LUAD_Finetuned')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 加载 LUAD 数据
    print("\n[Step 1] 加载 LUAD 数据集...")
    X, y, true_prob, feature_names = load_data('LUAD', 'interaction', verbose=True)
    
    # 划分训练/测试
    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X, y, true_prob, test_size=0.2, random_state=42, stratify=y
    )
    
    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(device), 
        torch.FloatTensor(y_train).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 3. 加载预训练模型
    print("\n[Step 2] 加载 PANCAN 预训练权重...")
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
    best_params = checkpoint['best_params'].copy()
    model_params = {k: v for k, v in best_params.items() if k not in ['lr', 'batch_size', 'epochs']}
    
    model = DLCNet(input_dim=X.shape[1], **model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("✅ 权重加载成功！")
    
    # 4. 严格冻结 (Strict Freeze)
    print("\n[Step 3] 执行严格冻结策略...")
    model.requires_grad_(False) # 先全部冻结
    
    # 只解锁预测头 (Head)
    unfrozen_layers = []
    for name, param in model.named_parameters():
        # 排除 hypergraph, encoder, decoder，只留 outcome_head
        if 'outcome_head' in name:
            param.requires_grad = True
            unfrozen_layers.append(name.split('.')[0])
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ❄️ 已冻结: Encoder, Hypergraph, Decoder")
    print(f"   🔥 正在微调: {list(set(unfrozen_layers))}")
    print(f"   📊 可训练参数: {trainable_params} (极少，防止过拟合)")
    
    # 5. 初始体检 (Pre-check)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        init_out = model(X_test_tensor)
        if isinstance(init_out, dict):
            init_logits = init_out.get('pred', init_out.get('Y_1'))
        else:
            init_logits = init_out
        init_prob = torch.sigmoid(init_logits).squeeze().cpu().numpy()
        init_auc = roc_auc_score(y_test, init_prob)
    print(f"\n[Pre-check] 微调前初始 AUC: {init_auc:.4f}")
    
    # 6. 微调循环
    print("\n[Step 4] 开始微调 (Linear Probing)...")
    
    # 学习率调大到 1e-3，因为只训练一层线性层
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 50 # 线性层收敛很快，50轮足够
    best_auc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if isinstance(outputs, dict):
                logits = outputs.get('pred', outputs.get('Y_1'))
            else:
                logits = outputs
                
            loss = criterion(logits.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 验证
        model.eval()
        with torch.no_grad():
            val_out = model(X_test_tensor)
            if isinstance(val_out, dict):
                val_logits = val_out.get('pred', val_out.get('Y_1'))
            else:
                val_logits = val_out
            val_prob = torch.sigmoid(val_logits).squeeze().cpu().numpy()
            val_auc = roc_auc_score(y_test, val_prob)
            
            # 保存最佳状态
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

    print(f"\n🏆 最佳 Val AUC: {best_auc:.4f}")

    # 7. 保存最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        
    save_path = output_dir / 'dlc_finetuned.pth'
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'best_params': best_params, 
        'metrics': {'auc': best_auc}
    }
    torch.save(final_checkpoint, save_path)
    print(f"✅ 模型已保存至: {save_path}")

if __name__ == '__main__':
    main()