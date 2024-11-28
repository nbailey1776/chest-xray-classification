import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate_model(model, data_loader, device, label_names):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)

    # Compute AUC-ROC for each class
    auc_scores = {}
    for i in range(len(label_names)):
        try:
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        except ValueError:
            auc = np.nan  # Handle cases with only one class present
        auc_scores[label_names[i]] = auc
        print(f'{label_names[i]}: AUC-ROC = {auc:.4f}')

    # Compute overall metrics
    preds = (all_outputs >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels.flatten(), preds.flatten())
    print(f'Overall Accuracy: {accuracy:.4f}')
    auc_scores = {k: float(v) if not np.isnan(v) else None for k, v in auc_scores.items()}
    return {'auc_scores': auc_scores, 'accuracy': float(accuracy)}
