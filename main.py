"""
Main script for Deepfake vs Real Faces Detection using ViT
This script runs the complete pipeline from data processing to model training and evaluation
"""

import warnings
warnings.filterwarnings("ignore")

from data_processor import DataProcessor
from model import ViTModel
from visualization import Visualizer
from huggingface_upload import HuggingFaceUploader
from config import OUTPUT_DIR, LABELS_LIST


def main():
    """Main function to run the complete pipeline"""
    print("=" * 80)
    print("DEEPFAKE VS REAL FACES DETECTION USING VISION TRANSFORMER")
    print("=" * 80)
    
    try:
        # Step 1: Data Processing
        print("\n1. DATA PROCESSING")
        print("-" * 40)
        processor = DataProcessor()
        train_data, test_data, label2id, id2label = processor.process_pipeline()
        
        # Step 2: Model Setup and Training
        print("\n2. MODEL SETUP AND TRAINING")
        print("-" * 40)
        vit_model = ViTModel()
        
        # Setup model pipeline
        trainer = vit_model.setup_pipeline(train_data, test_data, label2id, id2label)
        
        # Apply transforms to datasets
        train_transforms, val_transforms = vit_model.create_transform_functions()
        train_data.set_transform(train_transforms)
        test_data.set_transform(val_transforms)
        
        # Evaluate pre-training performance
        print("\nEvaluating pre-training model...")
        vit_model.evaluate_pretraining()
        
        # Train the model
        print("\nTraining model...")
        vit_model.train()
        
        # Evaluate post-training performance
        print("\nEvaluating post-training model...")
        vit_model.evaluate_posttraining()
        
        # Step 3: Model Evaluation and Visualization
        print("\n3. MODEL EVALUATION AND VISUALIZATION")
        print("-" * 40)
        
        # Make predictions
        outputs = vit_model.predict(test_data)
        
        # Create visualizations
        visualizer = Visualizer()
        results = visualizer.analyze_predictions(outputs, id2label)
        
        # Plot training history
        visualizer.plot_training_history(trainer)
        
        # Create summary report
        visualizer.create_summary_report(results)
        
        # Step 4: Save Model
        print("\n4. SAVING MODEL")
        print("-" * 40)
        vit_model.save_model()
        
        # Step 5: Test Single Image (Optional)
        print("\n5. TESTING SINGLE IMAGE")
        print("-" * 40)
        if len(test_data) > 0:
            sample_image = test_data[0]["image"]
            print(f"Testing sample image from test dataset...")
            result = visualizer.test_single_image(sample_image, OUTPUT_DIR)
            if result:
                print("Single image test completed successfully!")
        
        # Step 6: Hugging Face Upload (Optional)
        print("\n6. HUGGING FACE UPLOAD")
        print("-" * 40)
        print("Would you like to upload the model to Hugging Face Hub? (y/n)")
        response = input().lower().strip()
        
        if response == 'y':
            uploader = HuggingFaceUploader()
            uploader.upload_pipeline()
        else:
            print("Skipping Hugging Face upload.")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Model saved to: {OUTPUT_DIR}")
        print(f"Final accuracy: {results['accuracy']:.4f}")
        print(f"Final F1 score: {results['f1_score']:.4f}")
        
    except Exception as e:
        print(f"\nError occurred during pipeline execution: {e}")
        print("Please check your dataset path and dependencies.")
        raise


def run_data_processing_only():
    """Run only the data processing part"""
    print("Running data processing only...")
    processor = DataProcessor()
    train_data, test_data, label2id, id2label = processor.process_pipeline()
    return train_data, test_data, label2id, id2label


def run_training_only(train_data, test_data, label2id, id2label):
    """Run only the training part with pre-processed data"""
    print("Running training only...")
    vit_model = ViTModel()
    trainer = vit_model.setup_pipeline(train_data, test_data, label2id, id2label)
    
    # Apply transforms
    train_transforms, val_transforms = vit_model.create_transform_functions()
    train_data.set_transform(train_transforms)
    test_data.set_transform(val_transforms)
    
    # Train and evaluate
    vit_model.evaluate_pretraining()
    vit_model.train()
    vit_model.evaluate_posttraining()
    
    return vit_model, trainer


def run_evaluation_only(model_path=OUTPUT_DIR):
    """Run only the evaluation part with a saved model"""
    print("Running evaluation only...")
    
    # Load test data
    processor = DataProcessor()
    train_data, test_data, label2id, id2label = processor.process_pipeline()
    
    # Setup model
    vit_model = ViTModel()
    vit_model.load_processor()
    vit_model.create_transforms()
    vit_model.load_model(label2id, id2label)
    
    # Apply transforms
    train_transforms, val_transforms = vit_model.create_transform_functions()
    test_data.set_transform(val_transforms)
    
    # Create trainer
    trainer = vit_model.create_trainer(test_data, test_data)  # Using test_data for both
    
    # Make predictions
    outputs = vit_model.predict(test_data)
    
    # Visualize results
    visualizer = Visualizer()
    results = visualizer.analyze_predictions(outputs, id2label)
    visualizer.create_summary_report(results)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "data":
            run_data_processing_only()
        elif mode == "train":
            # Load pre-processed data
            train_data, test_data, label2id, id2label = run_data_processing_only()
            run_training_only(train_data, test_data, label2id, id2label)
        elif mode == "eval":
            run_evaluation_only()
        else:
            print("Invalid mode. Use: data, train, eval, or no argument for full pipeline")
    else:
        # Run full pipeline
        main()
