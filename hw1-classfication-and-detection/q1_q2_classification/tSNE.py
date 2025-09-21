import matplotlib.cm as cm
import random
import numpy as np
import torch
from sklearn.manifold import TSNE
from train_q2 import ResNet
import matplotlib.pyplot as plt
from voc_dataset import VOCDataset

path = 'checkpoint-model-epoch10.pth'  # Change this correspondingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(path, weights_only=False, map_location=device)
model.eval()

# Get class names from the dataset
voc_classes = VOCDataset.CLASS_NAMES
num_classes = len(voc_classes)  # VOC has 20 classes

# Feature extraction hook
features = []
def hook(module, inputs, outputs):
    features.append(inputs[0].detach().cpu().numpy().squeeze())

model.resnet.fc.register_forward_hook(hook)

test_dataset = VOCDataset(split='test', size=224, data_dir='data/VOCdevkit/VOC2007/')

# Sample random indices to visualize
indices = random.sample(range(len(test_dataset)), 1000)
X, y = [], []
for idx in indices:
    img, target, wgt = test_dataset[idx]
    img = img.to(device=device).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        _ = model(img)   # Forward pass -> hook captures features
    y.append(target)  # target = multi-hot labels

# Run t-SNE
tsne = TSNE(n_components=2, random_state=907)
X_2d = tsne.fit_transform(np.array(features))

# Assign colors
# https://matplotlib.org/stable/users/explain/colors/colormaps.html
colors = cm.tab20(np.linspace(0, 1, num_classes))

def get_color(label_vector):
    indices = np.where(label_vector == 1)[0]
    if len(indices) == 1:
        return colors[indices[0]]
    return np.mean(colors[indices], axis=0)


point_colors = np.array([get_color(lbl) for lbl in y])  # (1000, 4)

plt.figure(figsize=(10, 10))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=point_colors, s=10)

for i in range(num_classes):
    plt.scatter([], [], c=colors[i], label=voc_classes[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("t-SNE of VOC features")
plt.savefig('tsne_voc.png', bbox_inches='tight')
plt.show()
