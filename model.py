"""
Model module for Deepfake vs Real Faces Detection using ViT
"""

import torch
import evaluate
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

from config import (
    MODEL_NAME, OUTPUT_DIR, LOGS_DIR, NUM_EPOCHS, LEARNING_RATE,
    BATCH_SIZE, EVAL_BATCH_SIZE, WEIGHT_DECAY, WARMUP_STEPS,
    IMAGE_SIZE, RANDOM_ROTATION, SHARPNESS_FACTOR, LABELS_LIST,
    DEVICE
)


class ViTModel:
    """Vision Transformer model for deepfake detection"""
    
    def __init__(self, model_name=MODEL_NAME, num_labels=len(LABELS_LIST)):
        self.model_name = model_name
        self.num_labels = num_labels
        self.processor = None
        self.model = None
        self.trainer = None
        self.label2id = None
        self.id2label = None
        
    def load_processor(self):
        """Load ViT image processor"""
        print(f"Loading ViT processor from {self.model_name}...")
        
        # Create a processor for ViT model input from the pre-trained model
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        
        # Retrieve the image mean and standard deviation used for normalization
        self.image_mean, self.image_std = self.processor.image_mean, self.processor.image_std
        
        # Get the size (height) of the ViT model's input images
        self.size = self.processor.size["height"]
        print(f"Image size: {self.size}")
        
        return self.processor
    
    def create_transforms(self):
        """Create image transformation pipelines"""
        print("Creating image transformation pipelines...")
        
        # Define a normalization transformation for the input images
        normalize = Normalize(mean=self.image_mean, std=self.image_std)
        
        # Define a set of transformations for training data
        self.train_transforms = Compose([
            Resize((self.size, self.size)),             # Resize images to the ViT model's input size
            RandomRotation(RANDOM_ROTATION),            # Apply random rotation
            RandomAdjustSharpness(SHARPNESS_FACTOR),    # Adjust sharpness randomly
            ToTensor(),                                 # Convert images to tensors
            normalize                                   # Normalize images using mean and std
        ])
        
        # Define a set of transformations for validation data
        self.val_transforms = Compose([
            Resize((self.size, self.size)),             # Resize images to the ViT model's input size
            ToTensor(),                                 # Convert images to tensors
            normalize                                   # Normalize images using mean and std
        ])
        
        return self.train_transforms, self.val_transforms
    
    def create_transform_functions(self):
        """Create transform functions for dataset processing"""
        def train_transforms(examples):
            examples['pixel_values'] = [self.train_transforms(image.convert("RGB")) for image in examples['image']]
            return examples
        
        def val_transforms(examples):
            examples['pixel_values'] = [self.val_transforms(image.convert("RGB")) for image in examples['image']]
            return examples
        
        return train_transforms, val_transforms
    
    def load_model(self, label2id, id2label):
        """Load ViT model with label mappings"""
        print(f"Loading ViT model from {self.model_name}...")
        
        # Create a ViTForImageClassification model from a pretrained checkpoint
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        
        # Configure the mapping of class labels to their corresponding indices
        self.model.config.id2label = id2label
        self.model.config.label2id = label2id
        
        # Store label mappings
        self.label2id = label2id
        self.id2label = id2label
        
        # Calculate and print the number of trainable parameters in millions
        num_params = self.model.num_parameters(only_trainable=True) / 1e6
        print(f"Number of trainable parameters: {num_params:.2f}M")
        
        return self.model
    
    def create_collate_fn(self):
        """Create collate function for data loading"""
        def collate_fn(examples):
            # Stack the pixel values from individual examples into a single tensor
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            
            # Convert the label strings in examples to corresponding numeric IDs
            labels = torch.tensor([example['label'] for example in examples])
            
            # Return a dictionary containing the batched pixel values and labels
            return {"pixel_values": pixel_values, "labels": labels}
        
        return collate_fn
    
    def create_metrics_function(self):
        """Create metrics computation function"""
        # Load the accuracy metric
        accuracy = evaluate.load("accuracy")
        
        def compute_metrics(eval_pred):
            # Extract model predictions and true labels
            predictions = eval_pred.predictions
            label_ids = eval_pred.label_ids
            
            # Calculate accuracy using the loaded accuracy metric
            predicted_labels = predictions.argmax(axis=1)
            acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']
            
            # Return the computed accuracy as a dictionary
            return {"accuracy": acc_score}
        
        return compute_metrics
    
    def create_training_args(self):
        """Create training arguments"""
        print("Creating training arguments...")
        
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,                    # Directory for model checkpoints
            logging_dir=LOGS_DIR,                     # Directory for training logs
            eval_strategy="epoch",                    # Evaluate at end of each epoch
            learning_rate=LEARNING_RATE,              # Learning rate
            per_device_train_batch_size=BATCH_SIZE,   # Training batch size
            per_device_eval_batch_size=EVAL_BATCH_SIZE, # Evaluation batch size
            num_train_epochs=NUM_EPOCHS,              # Number of training epochs
            weight_decay=WEIGHT_DECAY,                # Weight decay
            warmup_steps=WARMUP_STEPS,               # Warmup steps
            remove_unused_columns=False,              # Don't remove unused columns
            save_strategy='epoch',                    # Save model per epoch
            load_best_model_at_end=True,              # Load best model at end
            save_total_limit=1,                       # Limit saved checkpoints
            report_to="none"                          # Don't report to external services
        )
        
        return args
    
    def create_trainer(self, train_dataset, eval_dataset):
        """Create trainer instance"""
        print("Creating trainer...")
        
        # Create trainer instance
        self.trainer = Trainer(
            self.model,
            self.create_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.create_collate_fn(),
            compute_metrics=self.create_metrics_function(),
            tokenizer=self.processor,
        )
        
        return self.trainer
    
    def evaluate_pretraining(self):
        """Evaluate model before training"""
        print("Evaluating pre-training model performance...")
        results = self.trainer.evaluate()
        print(f"Pre-training evaluation results: {results}")
        return results
    
    def train(self):
        """Train the model"""
        print("Starting model training...")
        results = self.trainer.train()
        print(f"Training completed. Results: {results}")
        return results
    
    def evaluate_posttraining(self):
        """Evaluate model after training"""
        print("Evaluating post-training model performance...")
        results = self.trainer.evaluate()
        print(f"Post-training evaluation results: {results}")
        return results
    
    def predict(self, test_dataset):
        """Make predictions on test dataset"""
        print("Making predictions on test dataset...")
        outputs = self.trainer.predict(test_dataset)
        print(f"Prediction metrics: {outputs.metrics}")
        return outputs
    
    def save_model(self):
        """Save the trained model"""
        print(f"Saving model to {OUTPUT_DIR}...")
        self.trainer.save_model()
        print("Model saved successfully!")
    
    def setup_pipeline(self, train_dataset, eval_dataset, label2id, id2label):
        """Complete setup pipeline for the model"""
        self.load_processor()
        self.create_transforms()
        self.load_model(label2id, id2label)
        self.create_trainer(train_dataset, eval_dataset)
        
        return self.trainer


if __name__ == "__main__":
    # Test the model setup
    from data_processor import DataProcessor
    
    # Process data
    processor = DataProcessor()
    train_data, test_data, label2id, id2label = processor.process_pipeline()
    
    # Setup model
    vit_model = ViTModel()
    trainer = vit_model.setup_pipeline(train_data, test_data, label2id, id2label)
