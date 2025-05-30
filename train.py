import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F

stimulus_window = 5
feature_dim = 520

X = torch.tensor(features_train, dtype=torch.float32)  # (N_samples, stimulus_window, feature_dim)
y = torch.tensor(fmri_train, dtype=torch.float32)      # (N_samples, fmri_dim)
X_val = torch.tensor(features_val, dtype=torch.float32)
y_val = torch.tensor(fmri_val, dtype=torch.float32)

# Conv用に次元変換 (PyTorch形式)
#X = X.permute(0, 2, 1)  # (N, C=feature_dim, T=stimulus_window)

# モデル定義
class ConvCompressor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)  # (N, out_channels, T=1)
        x = self.relu(x)
        return x.squeeze(-1)  # (N, out_channels)

class AttentionCompressor(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, 1)  # 各時間ステップごとの重要度スコアの算出
        self.output_layer = nn.Linear(feature_dim, output_dim)  # feature_dim→fMRIの次元

    def forward(self, x):
        attn_scores = self.query(x)  # (batch_size, time_steps, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # 時間方向でSoftmax (batch_size, time_steps, 1)を作用させる。
        compressed = (x * attn_weights).sum(dim=1)  # (batch_size, feature_dim)
        return self.output_layer(compressed)

# モデルと学習準備
#model = ConvCompressor(in_channels=feature_dim, out_channels=feature_dim, kernel_size=stimulus_window)
hrf_model = AttentionCompressor(feature_dim=520, output_dim = 1000)
optimizer = optim.Adam(hrf_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
best_val_loss = float('inf')
best_model_path = "best_hrf_model.pth"

# 学習ループ
epochs = 500
train_loss_list = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = hrf_model(X)
    loss = loss_fn(y_pred, y)
    train_loss_list.append(loss)
    loss.backward()
    optimizer.step()

    # バリデーション
    with torch.no_grad():
        y_val_pred = hrf_model(X_val)
        val_loss = loss_fn(y_val_pred, y_val).item()

    print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    # モデル保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(hrf_model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
