#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.models.cnn_gru import CNNGRUModel

def create_augmented_data(X, y, augmentation_factor=2):
    """Create augmented data by adding noise and variations."""
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        # Original sample
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Augmented samples
        for _ in range(augmentation_factor - 1):
            # Add small random noise
            noise = np.random.normal(0, 0.01, X[i].shape)
            X_aug.append(X[i] + noise)
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

def train_ultimate_model():
    """Train the ULTIMATE model to achieve 99%+ accuracy with advanced techniques."""
    
    print("Training ULTIMATE 99%+ accuracy model with advanced techniques...")
    
    # Load balanced dataset
    df = pd.read_csv('data/5class_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(df['category'].value_counts())
    
    # Extract features (exclude non-feature columns)
    exclude_cols = ['category', 'label', 'info', 'destination']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Feature columns: {feature_columns}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Clean data
    df_clean = df.dropna()
    X = df_clean[feature_columns]
    y = df_clean['category']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save label encoder
    np.save('data/processed/label_encoder_classes.npy', le.classes_)
    print(f"Label encoder classes: {list(le.classes_)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    print(f"Scaler saved with {len(scaler.feature_names_in_)} features")
    
    # Create sequences
    seq_len = 20
    X_sequences, y_sequences = [], []
    
    for i in range(len(X_scaled) - seq_len + 1):
        X_sequences.append(X_scaled[i:i + seq_len])
        y_sequences.append(y_encoded[i + seq_len - 1])
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    
    print(f"Created {len(X_seq)} sequences with shape {X_seq.shape}")
    
    # Data augmentation for better generalization
    print("Applying data augmentation...")
    X_aug, y_aug = create_augmented_data(X_seq, y_seq, augmentation_factor=3)
    print(f"Augmented data shape: {X_aug.shape}")
    
    # Split data with stratification - more training data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Data split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Check class distribution in splits
    print("Train class distribution:")
    print(np.bincount(y_train, minlength=len(le.classes_)))
    print("Val class distribution:")
    print(np.bincount(y_val, minlength=len(le.classes_)))
    print("Test class distribution:")
    print(np.bincount(y_test, minlength=len(le.classes_)))
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create data loaders with optimal batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Smaller batch for stability
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize ULTIMATE model with larger capacity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = X_train.shape[2]
    num_classes = len(le.classes_)
    
    # ULTIMATE model architecture for 99%+ accuracy
    model = CNNGRUModel(
        input_dim=input_dim,
        hidden_dim=1024,  # Larger for better capacity
        num_layers=4,     # More layers for complex patterns
        num_classes=num_classes,
        dropout_prob=0.3  # Slightly higher dropout
    ).to(device)
    
    print(f"ULTIMATE Model: input_dim={input_dim}, hidden_dim=1024, num_layers=4, num_classes={num_classes}")
    
    # Advanced training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # AdamW with lower LR
    
    # Advanced learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Training loop with advanced techniques
    print("Starting ULTIMATE training for 99%+ accuracy...")
    
    best_val_accuracy = 0.0
    patience = 30
    epochs_no_improve = 0
    max_epochs = 300
    
    # Training history for monitoring
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Advanced gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping based on accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            
            # Save best model
            torch.save(model.state_dict(), 'models/ultimate_model.pth')
            print(f"  Saved best model (epoch {epoch+1}) - Val Acc: {val_accuracy:.4f}")
        else:
            epochs_no_improve += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("ULTIMATE training completed!")
    
    # Load best model and test
    model.load_state_dict(torch.load('models/ultimate_model.pth'))
    model.eval()
    
    # Test on test set
    print("Testing on test set...")
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = test_correct / test_total
    print(f"FINAL TEST ACCURACY: {test_accuracy:.4f} ({test_correct}/{test_total})")
    
    # Per-class accuracy
    print("Per-Class Test Accuracy:")
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    class_accuracies = {}
    for i, class_name in enumerate(le.classes_):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_correct = (all_preds[class_mask] == all_labels[class_mask]).sum()
            class_accuracy = class_correct / class_mask.sum()
            class_count = class_mask.sum()
            class_accuracies[class_name] = class_accuracy
            print(f"  {class_name}: {class_correct}/{class_count} ({class_accuracy:.4f})")
    
    # Check if we achieved 99%+ accuracy
    if test_accuracy >= 0.99:
        print("SUCCESS! Achieved 99%+ accuracy!")
    else:
        print(f"Target not reached. Current accuracy: {test_accuracy:.4f}")
    
    # Update app components
    print("Updating app components...")
    joblib.dump(scaler, 'models/scaler.joblib')
    np.save('data/processed/label_encoder_classes.npy', le.classes_)
    
    print("App components updated!")
    print(f"ULTIMATE model trained! Achieves {test_accuracy:.4f} accuracy")
    
    return model, scaler, le

if __name__ == "__main__":
    train_ultimate_model()
