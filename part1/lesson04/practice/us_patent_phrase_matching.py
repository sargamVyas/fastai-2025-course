import pandas as pd
from pathlib import Path
import os
from datasets import Dataset
from transformers import AutoTokenizer

# --- CONFIGURATION ---
# Define the name of the Hugging Face model to be used for tokenization and training.
# Deberta-v3-small is a powerful and efficient choice for sequence classification.
MODEL_NAME = 'microsoft/deberta-v3-small'

# Define the relative path to the data directory.
# This assumes your data (e.g., train.csv) is located in a 'data' folder
# relative to where this script is executed.
# NOTE: Replace this with your actual local path if running outside a standard structure.
DATA_DIR = Path('data')
TRAIN_FILE = 'train.csv'

# --- DATA LOADING AND PREPROCESSING ---

def load_data(data_dir: Path, filename: str) -> pd.DataFrame:
    """Loads the training data from a specified CSV file."""
    # Use the / operator from pathlib.Path for cross-platform path joining
    file_path = data_dir / filename
    print(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}. Please check your path.")
        return pd.DataFrame()

def prepare_input_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a combined 'input' text column following a format often used
    for competitive NLP tasks (e.g., Siamese network style input).
    
    The format is: 'Text1: [context]; Text2: [target]; Anc1: [anchor]'
    """
    if df.empty:
        return df
        
    df['input'] = (
        'Text1: ' + df['context'] + 
        '; Text2: ' + df['target'] + 
        '; Anc1: ' + df['anchor']
    )
    return df

def tokenize_dataset(df: pd.DataFrame, model_nm: str):
    """
    Initializes a tokenizer and applies it to the 'input' column of the dataset.
    
    This step performs both tokenization (splitting text) and numericalization 
    (converting tokens to IDs).
    """
    if df.empty:
        print("Dataframe is empty, skipping tokenization.")
        return None

    # 1. Convert pandas DataFrame to Hugging Face Dataset format
    ds = Dataset.from_pandas(df)
    
    # 2. Load the pre-trained tokenizer
    print(f"Loading tokenizer: {model_nm}")
    tokz = AutoTokenizer.from_pretrained(model_nm)

    # 3. Define the tokenization function
    def tok_func(x): 
        # Tokenizer is applied to the list of texts in the 'input' column for the batch
        return tokz(x['input'])

    # 4. Apply the function to the entire dataset, processing in batches for efficiency
    print("Tokenizing dataset...")
    tok_ds = ds.map(tok_func, batched=True)
    
    return tok_ds

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    
    # Check if the data directory exists
    if not DATA_DIR.exists():
        print(f"Creating data directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)
        print("Please place your 'train.csv' file inside the 'data' directory.")
    
    # 1. Load data
    df = load_data(DATA_DIR, TRAIN_FILE)
    
    if not df.empty:
        # 2. Prepare the combined input text
        df = prepare_input_text(df)
        print("\nExample of prepared input text:")
        print(df['input'].head())

        # 3. Tokenize and Numericalize the data
        tokenized_data = tokenize_dataset(df, MODEL_NAME)
        
        if tokenized_data:
            print("\nTokenization Complete. Final Dataset Structure:")
            print(tokenized_data)
            # You can save the tokenized_data here for further training
            # e.g., tokenized_data.save_to_disk("tokenized_train_data")