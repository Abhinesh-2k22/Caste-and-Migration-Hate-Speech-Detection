
# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import random
import re
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')

# Text Preprocessing
def clean_text(text):
    """Clean text while preserving both Tamil and Romanized Tamil characters."""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s\u0B80-\u0BFF!?.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

# Dataset Class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self):
        #Return the length of the dataset.
        return len(self.texts)
        
    def augment_text(self, text):

        if random.random() < 0.3:  # 30% chance of augmentation
            if random.random() < 0.5:
                words = text.split()
                if len(words) > 3:  
                    idx = random.randint(1, len(words)-2)
                    del words[idx]
                    text = ' '.join(words)
            else:
                words = text.split()
                if len(words) > 3:
                    middle_words = words[1:-1]
                    random.shuffle(middle_words)
                    words[1:-1] = middle_words
                    text = ' '.join(words)
        return text
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = clean_text(text)
        
        if self.augment:
            text = self.augment_text(text)
            
        label = self.labels[idx]
        
        # Tokenize with special handling for both Tamil and Romanized Tamil
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Model Architecture
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class HateSpeechClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=2, dropout=0.3):  # Increased dropout
        super(HateSpeechClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

#  Visualization Functions
def plot_metrics(train_metrics, val_metrics, output_dir):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(val_metrics['accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics['f1'], label='Train F1', marker='o')
    plt.plot(val_metrics['f1'], label='Validation F1', marker='o')
    plt.title('F1 Score Curves')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# Training and Evaluation
def print_dataset_stats(train_df, val_df, test_df):
    """Print detailed statistics about the datasets."""
    print("\nDataset Statistics:")
    print("-" * 50)
    
    print("\nTraining Set:")
    print(f"Total samples: {len(train_df)}")
    print("Class distribution:")
    train_class_counts = train_df['label'].value_counts()
    for label, count in train_class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(train_df)*100:.2f}%)")
    
    print("\nText Length Statistics:")
    train_lengths = train_df['text'].str.len()
    print(f"  Average length: {train_lengths.mean():.2f} characters")
    print(f"  Min length: {train_lengths.min()} characters")
    print(f"  Max length: {train_lengths.max()} characters")

    def count_tamil_chars(text):
        return len(re.findall(r'[\u0B80-\u0BFF]', str(text)))
    
    print("\nLanguage Mix in Training Set:")
    train_tamil_chars = train_df['text'].apply(count_tamil_chars).sum()
    total_chars = train_df['text'].str.len().sum()
    print(f"  Tamil characters: {train_tamil_chars} ({train_tamil_chars/total_chars*100:.2f}%)")
    print(f"  Romanized Tamil/English: {total_chars - train_tamil_chars} ({(total_chars - train_tamil_chars)/total_chars*100:.2f}%)")
    
    print("\nValidation Set:")
    print(f"Total samples: {len(val_df)}")
    print("Class distribution:")
    val_class_counts = val_df['label'].value_counts()
    for label, count in val_class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(val_df)*100:.2f}%)")

    print("\nTest Set:")
    print(f"Total samples: {len(test_df)}")
    print("Class distribution:")
    test_class_counts = test_df['label'].value_counts()
    for label, count in test_class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(test_df)*100:.2f}%)")
    
    print("\nCombined Training + Validation:")
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Total samples: {len(combined_df)}")
    print("Class distribution:")
    combined_class_counts = combined_df['label'].value_counts()
    for label, count in combined_class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(combined_df)*100:.2f}%)")
    
    print("-" * 50)

def train_and_evaluate(train_file, dev_file, test_with_labels_file, output_dir='output', 
                      model_name="xlm-roberta-base",
                      num_epochs=10,
                      batch_size=8,
                      learning_rate=1e-5,
                      gradient_accumulation_steps=8):


    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(dev_file)
    test_df = pd.read_csv(test_with_labels_file)
    
    print_dataset_stats(train_df, val_df, test_df)
    
    # Calculate class weights with stronger emphasis on minority class
    class_counts = train_df['label'].value_counts()
    total_samples = len(train_df)
    class_weights = torch.FloatTensor([
        (total_samples / (2 * count)) ** 0.75
        for count in class_counts
    ]).to(device)
    print("\nClass weights:", class_weights.cpu().numpy())
    
    # Initialize tokenizer and model
    print(f"\nInitializing {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HateSpeechClassifier(model_name)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:")
    print(f"Total parameters: {total_params:,}")
    print("Using XLM-RoBERTa-base for efficient training")
    
    # Create datasets and dataloaders
    train_dataset = HateSpeechDataset(train_df['text'].values, train_df['label'].values, 
                                    tokenizer, augment=True)
    val_dataset = HateSpeechDataset(val_df['text'].values, val_df['label'].values, 
                                  tokenizer, augment=False)
    test_dataset = HateSpeechDataset(test_df['text'].values, test_df['label'].values,
                                   tokenizer, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Use CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Initialize optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': learning_rate
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': learning_rate * 2
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)
    

    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    

    train_metrics = {'accuracy': [], 'f1': []}
    val_metrics = {'accuracy': [], 'f1': []}
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    # Initialize mixed precision training
    scaler = GradScaler()
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_preds = []
        train_labels = []
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Mixed precision training
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item() * gradient_accumulation_steps
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        train_pred_counts = pd.Series(train_preds).value_counts()
        print(f"\nTraining predictions distribution: {train_pred_counts.to_dict()}")
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        

        val_pred_counts = pd.Series(val_preds).value_counts()
        print(f"Validation predictions distribution: {val_pred_counts.to_dict()}")
        
        # Compute metrics with macro average
        train_metrics['accuracy'].append(accuracy_score(train_labels, train_preds))
        train_metrics['f1'].append(precision_recall_fscore_support(
            train_labels, train_preds, average='macro', zero_division=0)[2])
        
        val_metrics['accuracy'].append(accuracy_score(val_labels, val_preds))
        val_metrics['f1'].append(precision_recall_fscore_support(
            val_labels, val_preds, average='macro', zero_division=0)[2])
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train - Accuracy: {train_metrics['accuracy'][-1]:.4f}, Macro F1: {train_metrics['f1'][-1]:.4f}")
        print(f"Val - Accuracy: {val_metrics['accuracy'][-1]:.4f}, Macro F1: {val_metrics['f1'][-1]:.4f}")
        

        plot_metrics(train_metrics, val_metrics, output_dir)
        
        # Early stopping and model saving based on macro F1
        if val_metrics['f1'][-1] > best_val_f1:
            best_val_f1 = val_metrics['f1'][-1]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print(f"New best model saved with validation macro F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    print(f"\nLoaded best model with validation macro F1: {best_val_f1:.4f}")
    
    # Evaluate on test set
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            test_preds.extend(preds)
            test_labels.extend(labels.numpy())

    test_pred_counts = pd.Series(test_preds).value_counts()
    print(f"\nTest predictions distribution: {test_pred_counts.to_dict()}")

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='macro', zero_division=0
    )
    
    print("\nFinal Test Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro Precision: {test_precision:.4f}")
    print(f"Macro Recall: {test_recall:.4f}")
    print(f"Macro F1 Score: {test_f1:.4f}")

    plot_confusion_matrix(test_labels, test_preds, output_dir)
    
    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }


if __name__ == "__main__":

    results = train_and_evaluate(
        train_file='train.csv',
        dev_file='dev.csv',
        test_with_labels_file='test_with_labels.csv',
        output_dir='output',
        model_name="xlm-roberta-base", 
        num_epochs=10,
        batch_size=8, 
        learning_rate=1e-5, 
        gradient_accumulation_steps=8
    ) 