"""
Build a 75-example analysis set (25 original test images + shuffled 16x16/8x8
versions), extract pre-final embeddings from each trained model checkpoint,
run PCA to 2D, and save labeled scatter plots.
"""
import csv
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from main import CIFAR10_MEAN, CIFAR10_STD, build_model


CHECKPOINT_DIR = './checkpoints'
OUTPUT_DIR = './pca_plots'
SAMPLES_PATH = os.path.join(OUTPUT_DIR, 'analysis_samples.npz')
METADATA_PATH = os.path.join(OUTPUT_DIR, 'analysis_samples.csv')
SEED = 0

MODEL_CLASSES = [
    'Plain-Old-CIFAR10-FC',
    'D-shuffletruffle-FC',
    'N-shuffletruffle-FC',
    'Plain-Old-CIFAR10-CNN',
    'D-shuffletruffle-CNN',
    'N-shuffletruffle-CNN',
    'Plain-Old-CIFAR10-Attention',
    'D-shuffletruffle-Attention',
    'N-shuffletruffle-Attention',
]

VARIANT_STYLES = {
    'original':  {'color': '#1f77b4', 'marker': 'o', 'facecolor': 'none', 'linewidth': 1.5},
    'patch16':   {'color': '#e6194b', 'marker': '^', 'facecolor': 'none', 'linewidth': 1.5},
    'patch8':    {'color': '#2ca02c', 'marker': 's', 'facecolor': 'none', 'linewidth': 1.5},
}


def shuffle_image(image, patch_size, rng):
    height, width, channels = image.shape
    assert height % patch_size == 0 and width % patch_size == 0
    h = height // patch_size
    w = width // patch_size
    patches = image.reshape(h, patch_size, w, patch_size, channels)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(h * w, patch_size, patch_size, channels)
    shuffled = patches.copy()
    rng.shuffle(shuffled, axis=0)
    shuffled = shuffled.reshape(h, w, patch_size, patch_size, channels)
    shuffled = shuffled.transpose(0, 2, 1, 3, 4).reshape(height, width, channels)
    return shuffled


def create_analysis_samples():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    chosen_indices = np.sort(rng.choice(len(testset), size=25, replace=False))

    images = []
    sample_ids = []
    class_labels = []
    variants = []
    source_indices = []

    for sample_id, dataset_idx in enumerate(chosen_indices, start=1):
        pil_image, class_label = testset[dataset_idx]
        image = np.asarray(pil_image)
        variant_images = {
            'original': image,
            'patch16': shuffle_image(image, 16, rng),
            'patch8': shuffle_image(image, 8, rng),
        }
        for variant_name, variant_image in variant_images.items():
            images.append(variant_image)
            sample_ids.append(sample_id)
            class_labels.append(class_label)
            variants.append(variant_name)
            source_indices.append(dataset_idx)

    np.savez(
        SAMPLES_PATH,
        images=np.asarray(images, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        class_labels=np.asarray(class_labels, dtype=np.int64),
        variants=np.asarray(variants),
        source_indices=np.asarray(source_indices, dtype=np.int64),
    )

    with open(METADATA_PATH, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'variant', 'cifar10_label', 'source_test_index'])
        for sample_id, variant, class_label, dataset_idx in zip(sample_ids, variants, class_labels, source_indices):
            writer.writerow([sample_id, variant, class_label, dataset_idx])


def load_analysis_samples():
    if not os.path.exists(SAMPLES_PATH):
        create_analysis_samples()
    bundle = np.load(SAMPLES_PATH, allow_pickle=True)
    return (
        bundle['images'],
        bundle['sample_ids'],
        bundle['class_labels'],
        bundle['variants'],
        bundle['source_indices'],
    )


def compute_pca(features, n_dim=2):
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_dim]
    return centered @ components.T


def extract_embeddings(model, images, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    tensors = [transform(image) for image in images]
    batch = torch.stack(tensors).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model.forward_features(batch)
    return embeddings.cpu().numpy()


def plot_model_embeddings(model_name, projected, sample_ids, variants):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(f'{model_name} PCA of Pre-final Embeddings', fontsize=12, fontweight='bold')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.grid(True, alpha=0.3)

    for variant_name, style in VARIANT_STYLES.items():
        mask = variants == variant_name
        coords = projected[mask]
        ids = sample_ids[mask]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            facecolors=style['facecolor'],
            edgecolors=style['color'],
            linewidths=style['linewidth'],
            marker=style['marker'],
            s=70,
            alpha=0.9,
            label=variant_name,
        )
        for x, y, sample_id in zip(coords[:, 0], coords[:, 1], ids):
            ax.text(x + 0.05, y + 0.05, str(int(sample_id)), fontsize=7, color=style['color'])

    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'{model_name}_pca.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images, sample_ids, _, variants, _ = load_analysis_samples()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    missing = [model_name for model_name in MODEL_CLASSES
               if not os.path.exists(os.path.join(CHECKPOINT_DIR, f'{model_name}.pt'))]
    if missing:
        missing_str = ', '.join(missing)
        raise FileNotFoundError(
            f'Missing checkpoints for: {missing_str}. '
            'Run training after checkpoint support is enabled.'
        )

    for model_name in MODEL_CLASSES:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}.pt')
        model = build_model(model_name, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        embeddings = extract_embeddings(model, images, device)
        projected = compute_pca(embeddings, n_dim=2)
        plot_model_embeddings(model_name, projected, sample_ids, variants)


if __name__ == '__main__':
    main()
