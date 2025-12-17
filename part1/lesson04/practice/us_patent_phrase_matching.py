import pandas as pd
import numpy as np
from pathlib import Path
import os
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# --- CONFIGURATION ---
MODEL_NAME = 'microsoft/deberta-v3-small'
DATA_DIR = Path('data')
TRAIN_FILE = 'train.csv'
OUTPUT_DIR = 'outputs'

# --- METRICS ---
def compute_metrics(eval_pred):
    """Calculates Pearson correlation between predictions and labels."""
    predictions, labels = eval_pred
    # Deberta returns a 2D array for regression; flatten to 1D
    predictions = predictions.flatten()
    return {"pearson": np.corrcoef(predictions, labels)[0, 1]}

# --- DATA LOADING AND PREPROCESSING ---

def load_and_prepare_data(data_dir: Path, filename: str) -> pd.DataFrame:
    file_path = data_dir / filename
    try:
        df = pd.read_csv(file_path)
        # Combine context, target, and anchor into a single string
        # We use lowercase to ensure consistency
        df['input'] = (
            'TEXT1: ' + df['context'].str.lower() + 
            '; TEXT2: ' + df['target'].str.lower() + 
            '; ANC1: ' + df['anchor'].str.lower()
        )
        # Rename 'score' to 'labels' as required by the Trainer API
        return df.rename(columns={'score': 'labels'})
    except FileNotFoundError:
        print(f"ERROR: {file_path} not found.")
        return pd.DataFrame()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Load and Prepare
    df = load_and_prepare_data(DATA_DIR, TRAIN_FILE)
    
    if not df.empty:
        # 2. Tokenization
        tokz = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        def tok_func(x): 
            return tokz(x['input'], truncation=True, max_length=512)

        # 3. Create Dataset and Split
        # Convert to HF Dataset and split into 75% train / 25% validation
        ds = Dataset.from_pandas(df).map(tok_func, batched=True)
        dds = ds.train_test_split(0.25, seed=42)
        
        # 4. Training Arguments
        # We use a cosine learning rate scheduler and FP16 for speed
        args = TrainingArguments(
            OUTPUT_DIR,
            learning_rate=8e-5,
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            fp16=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=256,
            num_train_epochs=4,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="pearson",
            report_to='none'
        )

        # 5. Initialize Model
        # num_labels=1 tells the model this is a regression task
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=1
        )

        # 6. Training
        trainer = Trainer(
            model,
            args,
            train_dataset=dds['train'],
            eval_dataset=dds['test'],
            tokenizer=tokz,
            compute_metrics=compute_metrics
        )

        print("Starting training...")
        trainer.train()
        
        # 7. Save the final model
        trainer.save_model("./final_patent_model")
        print("Model saved to ./final_patent_model")