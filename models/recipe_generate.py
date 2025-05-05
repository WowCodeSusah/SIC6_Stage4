import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import os
from os.path import dirname, abspath

# ========== Configuration ==========
csv_path = 'models/recipe/combined_recipes.csv'  # Change to your actual CSV file path
model_checkpoint = 't5-small'
output_dir_prefix = 't5_recipe_trained_until_chunk_'
chunk_size = 100000  # Adjust based on your system
num_train_epochs = 2
batch_size = 4
max_input_length = 512
max_target_length = 512

# ========== Step 1: Load and Prepare Data ==========
df = pd.read_csv(csv_path)
df = df.sample(frac=1).reset_index(drop=True)
chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

# ========== Step 5: Testing Function ==========
def generate_recipe(ingredient_text, model_subdir=f"{output_dir_prefix}4"):
    base_path = dirname(abspath(__file__))
    model_path = os.path.join(base_path, model_subdir)

    model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)

    input_text = "generate recipe: " + ingredient_text
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True,
                          padding="max_length", max_length=512).input_ids
    outputs = model.generate(input_ids=input_ids, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)