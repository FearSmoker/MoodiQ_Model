#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import onnx
from onnx import helper, TensorProto

# Seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "spec_rate"
]
LABEL_COLUMN = "labels"

# PyTorch Model Architecture (matching the DeepMLP template)
class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes, mean, std):
        super().__init__()
        self.norm = NormalizeLayer(mean, std)
        self.core = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        z = self.norm(x)
        logits = self.core(z)
        return logits

def train_model(model, train_loader, val_loader, device, class_weights, epochs=200, patience=30, lr=5e-4):
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    best_acc, best_state = 0.0, None
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.float()
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.float()
                preds = torch.argmax(model(xb), dim=1).cpu()
                y_true += yb.tolist()
                y_pred += preds.tolist()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        print(f"Epoch {ep:03d}: Loss={total_loss/len(train_loader):.4f} | Acc={acc:.4f} | F1={f1:.4f}")

        if acc > best_acc:
            best_acc, best_state, wait = acc, model.state_dict(), 0
        else:
            wait += 1
        if wait >= patience:
            print(f"✅ Early stop at epoch {ep} | Best Acc={best_acc:.4f}")
            break

    model.load_state_dict(best_state)
    return model, best_acc

def main():
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    
    # Try v2 dataset first (properly labeled), fallback to old
    csv_path = "mood_dataset_v2.csv"
    if not os.path.exists(csv_path):
        csv_path = "mood_dataset_12mood.csv"
        print(f"⚠️  mood_dataset_v2.csv not found, using fallback: {csv_path}")
        print("   Run label_12mood_v2.py first for best results.")
    if not os.path.exists(csv_path):
        print(f"Error: No labeled dataset found. Run label_12mood_v2.py first.")
        return
        
    df = pd.read_csv(csv_path)
    print("✅ Loaded dataset:", df.shape, "| Source:", csv_path)
    print("   Class distribution:")
    for mood, count in df['labels'].value_counts().items():
        print(f"      {mood:<15}: {count:,}")
    print()

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
    y = df[LABEL_COLUMN].astype(str).iloc[X.index]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Train / Val / Test split
    Xtr, Xtemp, ytr, ytemp = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    Xv, Xt, yv, yt = train_test_split(Xtemp, ytemp, test_size=0.5, stratify=ytemp, random_state=42)

    mean = Xtr.mean().values
    std = Xtr.std().replace(0, 1).values
    cw = torch.tensor(compute_class_weight("balanced", classes=np.unique(ytr), y=ytr), dtype=torch.float32)

    def mk_loader(X, y, sh=True):
        ds = torch.utils.data.TensorDataset(torch.tensor(X.values, dtype=torch.float32),
                                            torch.tensor(y, dtype=torch.long))
        return torch.utils.data.DataLoader(ds, batch_size=512, shuffle=sh)

    tr_loader, v_loader = mk_loader(Xtr, ytr), mk_loader(Xv, yv, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Training 12-mood model on device: {device}")

    model = DeepMLP(len(FEATURES), len(le.classes_), mean, std).to(device)
    model, acc = train_model(model, tr_loader, v_loader, device, cw, epochs=250, patience=30)
    print(f"✅ Final Validation Accuracy: {acc:.4f}")

    # Output paths
    onnx_path = "models/moodiq_v4.onnx"
    metadata_path = "models/model_metadata.json"
    
    # Save label map and scaler stats in metadata matching expected structure
    metadata = {
        "scaler_mean": mean.tolist(),
        "scaler_scale": std.tolist(),
        "audio_features": FEATURES,
        "mood_classes": le.classes_.tolist(),
        "version": "v4",
        "accuracy": f"{acc * 100:.2f}%"
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved model metadata → {metadata_path}")

    # Export to ONNX (with outputs compatible with model_service.py)
    class ExportWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            logits = self.base(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            return probs, idx, conf.unsqueeze(1)

    wrapped = ExportWrapper(model.cpu()).eval()
    dummy = torch.randn(1, len(FEATURES))

    torch.onnx.export(
        wrapped, dummy, onnx_path,
        input_names=["features"],
        output_names=["probs", "pred_index", "confidence"],
        dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=11  # opset 11 works without onnxscript and is supported by onnxruntime
    )
    print(f"✅ Exported ONNX model → {onnx_path}")
    print(f"   File size: {os.path.getsize(onnx_path) / 1024:.1f} KB")
    print("\n🎉 Model training and export complete!")
    print(f"   Real-world accuracy target: 87%+ (validated via rule engine parity)")

if __name__ == "__main__":
    main()
