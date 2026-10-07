"""
Data processing module for Deepfake vs Real Faces Detection
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from datasets import Dataset, Image, ClassLabel
from PIL import ImageFile

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import DATASET_PATH, LABELS_LIST, TEST_SIZE, RANDOM_STATE


class DataProcessor:
    """Class for processing and preparing the dataset"""
    
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = Path(dataset_path)
        self.file_names = []
        self.labels = []
        self.df = None
        self.dataset = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """Load image files and labels from dataset directory"""
        print("Loading data from dataset...")
        
        # Iterate through all image files in the specified directory
        for file in sorted((self.dataset_path.glob('*/*/*.*'))):
            # Extract the label from the file path (works on both Windows and Unix)
            label = file.parts[-2]  # Get the second-to-last part of the path
            self.labels.append(label)  # Add the label to the list
            self.file_names.append(str(file))  # Add the file path to the list
        
        print(f"Total files loaded: {len(self.file_names)}")
        print(f"Labels found: {set(self.labels)}")
        
        # Normalize labels to match expected format (Real/Fake)
        normalized_labels = []
        for label in self.labels:
            if label.lower() in ['real', 'genuine', 'authentic']:
                normalized_labels.append('Real')
            elif label.lower() in ['fake', 'synthetic', 'generated', 'artificial']:
                normalized_labels.append('Fake')
            else:
                # Default to original label if not recognized
                normalized_labels.append(label)
        
        # Create a pandas dataframe from the collected file names and normalized labels
        self.df = pd.DataFrame.from_dict({"image": self.file_names, "label": normalized_labels})
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Normalized labels: {set(normalized_labels)}")
        
        return self.df
    
    def balance_dataset(self):
        """Apply random oversampling to balance the dataset"""
        print("Balancing dataset using random oversampling...")
        
        # 'y' contains the target variable (label) we want to predict
        y = self.df[['label']]
        
        # Drop the 'label' column from the DataFrame 'df' to separate features from the target variable
        df_features = self.df.drop(['label'], axis=1)
        
        # Create a RandomOverSampler object with a specified random seed
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        
        # Use the RandomOverSampler to resample the dataset by oversampling the minority class
        df_resampled, y_resampled = ros.fit_resample(df_features, y)
        
        # Delete the original 'y' variable to save memory as it's no longer needed
        del y
        
        # Add the resampled target variable 'y_resampled' as a new 'label' column in the DataFrame 'df'
        df_resampled['label'] = y_resampled
        
        # Delete the 'y_resampled' variable to save memory as it's no longer needed
        del y_resampled
        
        # Perform garbage collection to free up memory used by discarded variables
        gc.collect()
        
        self.df = df_resampled
        print(f"Balanced dataset shape: {self.df.shape}")
        
        return self.df
    
    def create_dataset(self):
        """Create HuggingFace dataset from DataFrame"""
        print("Creating HuggingFace dataset...")
        
        # Create a dataset from a Pandas DataFrame
        self.dataset = Dataset.from_pandas(self.df).cast_column("image", Image())
        
        return self.dataset
    
    def create_label_mappings(self):
        """Create label to ID mappings"""
        # Initialize empty dictionaries to map labels to IDs and vice versa
        label2id, id2label = dict(), dict()
        
        # Iterate over the unique labels and assign each label an ID, and vice versa
        for i, label in enumerate(LABELS_LIST):
            label2id[label] = i  # Map the label to its corresponding ID
            id2label[i] = label  # Map the ID to its corresponding label
        
        print("Mapping of IDs to Labels:", id2label)
        print("Mapping of Labels to IDs:", label2id)
        
        return label2id, id2label
    
    def prepare_dataset(self, label2id, id2label):
        """Prepare dataset with label mappings and train/test split"""
        print("Preparing dataset with label mappings...")
        
        # Creating classlabels to match labels to IDs
        ClassLabels = ClassLabel(num_classes=len(LABELS_LIST), names=LABELS_LIST)
        
        # Mapping labels to IDs
        def map_label2id(example):
            example['label'] = ClassLabels.str2int(example['label'])
            return example
        
        self.dataset = self.dataset.map(map_label2id, batched=True)
        
        # Casting label column to ClassLabel Object
        self.dataset = self.dataset.cast_column('label', ClassLabels)
        
        # Splitting the dataset into training and testing sets
        self.dataset = self.dataset.train_test_split(
            test_size=TEST_SIZE, 
            shuffle=True, 
            stratify_by_column="label"
        )
        
        # Extracting the training and testing data from the split dataset
        self.train_data = self.dataset['train']
        self.test_data = self.dataset['test']
        
        print(f"Training data size: {len(self.train_data)}")
        print(f"Testing data size: {len(self.test_data)}")
        
        return self.train_data, self.test_data
    
    def get_data_info(self):
        """Get information about the dataset"""
        if self.df is not None:
            print("\nDataset Information:")
            print(f"Total samples: {len(self.df)}")
            print(f"Label distribution:")
            print(self.df['label'].value_counts())
            
            if self.train_data is not None and self.test_data is not None:
                print(f"\nTrain/Test split:")
                print(f"Training samples: {len(self.train_data)}")
                print(f"Testing samples: {len(self.test_data)}")
    
    def process_pipeline(self):
        """Run the complete data processing pipeline"""
        self.load_data()
        self.balance_dataset()
        self.create_dataset()
        label2id, id2label = self.create_label_mappings()
        train_data, test_data = self.prepare_dataset(label2id, id2label)
        self.get_data_info()
        
        return train_data, test_data, label2id, id2label


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    train_data, test_data, label2id, id2label = processor.process_pipeline()
