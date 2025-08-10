import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(model, dataset, device, n_samples=6):
    """Visualize predictions on a few validation images."""
    model.eval()
    axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for ax, idx in zip(axes, indices):
        img, true_label = dataset[idx]
        input_img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(input_img)
                _, pred_label = torch.max(output, 1)

        # Undo normalization for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)
        ax.set_title(f"True: {true_label} | Pred: {pred_label.item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
