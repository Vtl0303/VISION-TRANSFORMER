"""
Visualization module wrapper for deepfake detection evaluation
This module provides the Visualizer class for analyzing model predictions
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

class Visualizer:
    """Class for visualizing and analyzing model predictions"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_predictions(self, outputs, id2label):
        """Analyze model predictions and return comprehensive results"""
        print("Analyzing predictions...")
        
        # Extract predictions and labels
        predictions = outputs.predictions
        label_ids = outputs.label_ids
        
        # Get predicted labels
        predicted_labels = predictions.argmax(axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(label_ids, predicted_labels)
        precision = precision_score(label_ids, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(label_ids, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(label_ids, predicted_labels, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(label_ids, predicted_labels)
        
        # Classification report
        class_report = classification_report(label_ids, predicted_labels, 
                                           target_names=list(id2label.values()),
                                           output_dict=True, zero_division=0)
        
        # ROC AUC if probabilities available
        roc_auc = None
        if predictions.shape[1] > 1:
            try:
                # Use probabilities for positive class (assuming Fake = 1)
                y_proba = predictions[:, 1]  # Probability of Fake class
                roc_auc = roc_auc_score(label_ids, y_proba)
            except:
                roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'predictions': predicted_labels,
            'true_labels': label_ids
        }
        
        self.results = results
        return results
    
    def plot_training_history(self, trainer, save_path="training_curves.png"):
        """Plot training history from trainer"""
        print("Creating training history plot...")
        
        try:
            # Get training history from trainer state
            if hasattr(trainer.state, 'log_history'):
                history = trainer.state.log_history
                
                # Extract metrics
                epochs = []
                train_loss = []
                eval_loss = []
                eval_accuracy = []
                
                for log in history:
                    if 'epoch' in log:
                        epochs.append(log['epoch'])
                    if 'train_loss' in log:
                        train_loss.append(log['train_loss'])
                    if 'eval_loss' in log:
                        eval_loss.append(log['eval_loss'])
                    if 'eval_accuracy' in log:
                        eval_accuracy.append(log['eval_accuracy'])
                
                if epochs and (train_loss or eval_loss or eval_accuracy):
                    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Plot loss
                    if train_loss and eval_loss:
                        axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
                        axes[0].plot(epochs, eval_loss, 'r-', label='Validation Loss', linewidth=2)
                        axes[0].set_xlabel('Epoch')
                        axes[0].set_ylabel('Loss')
                        axes[0].set_title('Training and Validation Loss')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                    
                    # Plot accuracy
                    if eval_accuracy:
                        axes[1].plot(epochs, eval_accuracy, 'g-', label='Validation Accuracy', linewidth=2)
                        axes[1].set_xlabel('Epoch')
                        axes[1].set_ylabel('Accuracy')
                        axes[1].set_title('Validation Accuracy')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Training history saved to {save_path}")
                    return
        except Exception as e:
            print(f"Could not create training history plot: {e}")
        
        # Fallback: create placeholder plot
        self._create_placeholder_plot(save_path)
    
    def create_summary_report(self, results, save_path="summary_report.txt"):
        """Create summary text report"""
        print("Creating summary report...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("DEEPFAKE DETECTION MODEL SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n")
            
            if results.get('roc_auc'):
                f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
            
            f.write("\nCONFUSION MATRIX:\n")
            f.write("-" * 20 + "\n")
            cm = results['confusion_matrix']
            f.write(f"True Negatives: {cm[0,0]}\n")
            f.write(f"False Positives: {cm[0,1]}\n")
            f.write(f"False Negatives: {cm[1,0]}\n")
            f.write(f"True Positives: {cm[1,1]}\n")
            
            f.write("\nCLASSIFICATION REPORT:\n")
            f.write("-" * 20 + "\n")
            class_report = results['classification_report']
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):
                    f.write(f"\n{class_name}:\n")
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"  {metric}: {value:.4f}\n")
        
        print(f"Summary report saved to {save_path}")
    
    def test_single_image(self, image, model_path, save_path="single_image_test.png"):
        """Test model on a single image"""
        print("Testing single image...")
        
        try:
            from transformers import ViTImageProcessor, ViTForImageClassification
            import torch
            from PIL import Image
            
            # Load model and processor
            processor = ViTImageProcessor.from_pretrained(model_path)
            model = ViTForImageClassification.from_pretrained(model_path)
            model.eval()
            
            # Process image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            inputs = processor(images=image, return_tensors="pt")
            
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
                predicted_class_id = probabilities.argmax().item()
                confidence = probabilities[predicted_class_id].item()
            
            # Get class name
            id2label = model.config.id2label
            predicted_class = id2label[str(predicted_class_id)]
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Show image
            axes[0].imshow(image)
            axes[0].set_title(f'Input Image')
            axes[0].axis('off')
            
            # Show prediction
            axes[1].bar(['Real', 'Fake'], probabilities.numpy(), color=['blue', 'red'], alpha=0.7)
            axes[1].set_title(f'Prediction: {predicted_class} ({confidence:.3f})')
            axes[1].set_ylabel('Probability')
            axes[1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Single image test completed. Prediction: {predicted_class} ({confidence:.3f})")
            print(f"Result saved to {save_path}")
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.numpy()
            }
            
        except Exception as e:
            print(f"Error testing single image: {e}")
            return None
    
    def _create_placeholder_plot(self, save_path="placeholder_plot.png"):
        """Create a placeholder plot when training history is not available"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'Training History\nNot Available', 
                ha='center', va='center', fontsize=16, 
                transform=ax.transAxes)
        ax.set_title('Training Curves')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Placeholder plot saved to {save_path}")
    
    def plot_confusion_matrix(self, results, save_path="confusion_matrix.png"):
        """Plot confusion matrix"""
        print("Creating confusion matrix plot...")
        
        cm = results['confusion_matrix']
        labels = ['Real', 'Fake']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add percentage annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                percentage = cm[i, j] / cm.sum() * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, results, save_path="roc_curve.png"):
        """Plot ROC curve"""
        if not results.get('roc_auc'):
            print("ROC AUC not available, skipping ROC curve plot")
            return
            
        print("Creating ROC curve plot...")
        
        # This is a simplified version - in practice you'd need the actual probabilities
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {save_path}")


if __name__ == "__main__":
    # Test the visualizer
    print("Visualization module loaded successfully!")
