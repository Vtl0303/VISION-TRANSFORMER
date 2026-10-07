"""
Quick evaluation script: sample 10 test images, run predictions with the trained ViT model,
visualize a grid with predicted labels and confidences, and print a brief report.
"""

import warnings
warnings.filterwarnings("ignore")

import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from transformers import ViTImageProcessor, ViTForImageClassification

# Ensure project root on sys.path so `config.py` is importable when running directly
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
if (PROJECT_ROOT / 'visualixation').exists():
    PROJECT_ROOT = PROJECT_ROOT  # running from project root
else:
    PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TEST_PATH, OUTPUT_DIR, LABELS_LIST


def load_model(model_dir: str = OUTPUT_DIR) -> Tuple[ViTImageProcessor, ViTForImageClassification, torch.device, dict, dict]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = ViTImageProcessor.from_pretrained(model_dir)
    model = ViTForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    # Mappings
    id2label = getattr(model.config, 'id2label', None)
    label2id = getattr(model.config, 'label2id', None)
    if not isinstance(id2label, dict) or not isinstance(label2id, dict):
        # Fallback to LABELS_LIST order
        id2label = {i: lbl for i, lbl in enumerate(LABELS_LIST)}
        label2id = {lbl: i for i, lbl in enumerate(LABELS_LIST)}
    return processor, model, device, id2label, label2id


def sample_test_images(num_images: int = 10, dataset: str = 'dataset') -> List[Path]:
    """
    Sample test images from specified dataset.
    
    Args:
        num_images: Number of images to sample
        dataset: Dataset to use ('dataset' or 'dataset2')
    """
    if dataset == 'dataset':
        # Original dataset structure: Dataset/Test/Real and Dataset/Test/Fake
        test_real = Path(TEST_PATH) / 'Real'
        test_fake = Path(TEST_PATH) / 'Fake'
    elif dataset == 'dataset2':
        # Dataset2 structure: dataset2/test/real and dataset2/test/fake
        test_real = Path('dataset2/test/real')
        test_fake = Path('dataset2/test/fake')
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'dataset' or 'dataset2'")
    
    images = list(test_real.glob('*.jpg')) + list(test_real.glob('*.png'))
    images += list(test_fake.glob('*.jpg')) + list(test_fake.glob('*.png'))
    
    if len(images) == 0:
        raise FileNotFoundError(f"No images found under {test_real} or {test_fake}")
    
    random.shuffle(images)
    return images[:num_images]


@torch.no_grad()
def predict_image(image_path: Path, processor: ViTImageProcessor, model: ViTForImageClassification, device: torch.device) -> Tuple[int, float, np.ndarray]:
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    pred_label = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_label, confidence, probs


def make_grid(images: List[Path], predictions: List[int], confidences: List[float], save_path: Path, id2label: dict) -> None:
    cols = 5
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, img_path in enumerate(images):
        ax = plt.subplot(rows, cols, idx + 1)
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        pred_name = id2label.get(int(predictions[idx]), str(predictions[idx]))
        title = f"{img_path.parent.name} → {pred_name} ({confidences[idx]*100:.1f}%)"
        color = 'green' if pred_name.strip().lower() == 'real' else 'red'
        ax.set_title(title, fontsize=10, color=color)
        ax.axis('off')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def evaluate(num_images: int = 10, model_dir: str = OUTPUT_DIR, out_image: str = 'quick_eval_grid.png', dataset: str = 'dataset') -> None:
    """
    Evaluate model on test images from specified dataset.
    
    Args:
        num_images: Number of images to evaluate
        model_dir: Directory containing the trained model
        out_image: Output filename for the visualization grid
        dataset: Dataset to use ('dataset' or 'dataset2')
    """
    print(f"Loading model from {model_dir}...")
    processor, model, device, id2label, label2id = load_model(model_dir)

    print(f"Sampling {num_images} test images from {dataset}...")
    image_paths = sample_test_images(num_images, dataset)

    print("Running predictions...")
    preds, confs, gts = [], [], []
    for p in image_paths:
        pred, conf, probs = predict_image(p, processor, model, device)
        preds.append(pred)
        confs.append(conf)
        # Map ground-truth folder name to model's label index robustly
        folder_name = p.parent.name.strip().lower()
        gt_label_name = 'Real' if folder_name == 'real' else ('Fake' if folder_name == 'fake' else folder_name)
        # Try direct match with model's label2id (case-insensitive)
        gt_idx = None
        for k, v in label2id.items():
            if isinstance(k, str) and k.strip().lower() == str(gt_label_name).strip().lower():
                gt_idx = int(v)
                break
        if gt_idx is None:
            # Fallback to conventional mapping
            gt_idx = 0 if folder_name == 'real' else 1
        gts.append(gt_idx)

    # Metrics
    preds_arr = np.array(preds)
    gts_arr = np.array(gts)
    acc = float((preds_arr == gts_arr).mean())

    # Per-class accuracy
    report_lines = []
    report_lines.append(f"=== EVALUATION RESULTS ON {dataset.upper()} ===")
    report_lines.append(f"Accuracy on {len(image_paths)} samples: {acc*100:.2f}%")
    for class_idx, class_name in enumerate(LABELS_LIST):
        mask = gts_arr == class_idx
        if mask.sum() > 0:
            cls_acc = float((preds_arr[mask] == gts_arr[mask]).mean())
            report_lines.append(f"- {class_name} accuracy: {cls_acc*100:.2f}% (n={int(mask.sum())})")
        else:
            report_lines.append(f"- {class_name} accuracy: n=0")

    print("\n".join(report_lines))

    # Save grid with dataset-specific filename
    grid_filename = f'quick_eval_grid_{dataset}.png' if out_image == 'quick_eval_grid.png' else out_image
    grid_path = Path(grid_filename)
    make_grid(image_paths, preds, confs, grid_path, id2label)
    print(f"Saved visualization grid to: {grid_path.resolve()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick evaluation of ViT model on deepfake detection')
    parser.add_argument('--dataset', type=str, choices=['dataset', 'dataset2'], default='dataset',
                       help='Dataset to evaluate on (dataset or dataset2)')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of test images to sample and evaluate')
    parser.add_argument('--model_dir', type=str, default=OUTPUT_DIR,
                       help='Directory containing the trained model')
    parser.add_argument('--out_image', type=str, default='quick_eval_grid.png',
                       help='Output filename for visualization grid')
    
    args = parser.parse_args()
    
    print(f"Starting evaluation on {args.dataset} dataset...")
    evaluate(
        num_images=args.num_images,
        model_dir=args.model_dir,
        out_image=args.out_image,
        dataset=args.dataset
    )
    
    # Also run on the other dataset for comparison if only one was specified
    if args.dataset == 'dataset':
        print("\n" + "="*50)
        print("Running evaluation on dataset2 for comparison...")
        evaluate(
            num_images=args.num_images,
            model_dir=args.model_dir,
            out_image='quick_eval_grid_dataset2.png',
            dataset='dataset2'
        )
    elif args.dataset == 'dataset2':
        print("\n" + "="*50)
        print("Running evaluation on original dataset for comparison...")
        evaluate(
            num_images=args.num_images,
            model_dir=args.model_dir,
            out_image='quick_eval_grid_dataset.png',
            dataset='dataset'
        )


