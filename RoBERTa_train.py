import os
from transformers import (
    RobertaConfig, 
    RobertaForMaskedLM, 
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from tokenizers import Tokenizer
from datasets import load_dataset

# --- Configuration ---
# Path to the tokenizer file you trained.
TOKENIZER_FILE = 'morpheme_tokenizer/tokenizer.json'
# Path to the segmented corpus file.
CORPUS_FILE = 'babylm_10M_morphemes.txt'
# Directory to save the trained model and training checkpoints.
MODEL_OUTPUT_DIR = 'roberta-base-morphemes'


def configure_model_and_tokenizer(tokenizer_path):
    """
    Loads the custom tokenizer and configures a RoBERTa-base model 
    for training from scratch. This combines the logic from Step 5.
    """
    print(f"⚙️  Loading tokenizer and configuring RoBERTa model...")

    # 1. Load custom tokenizer from file and wrap it for transformers compatibility
    raw_tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )
    
    # 2. Load standard RoBERTa-base configuration
    config = RobertaConfig.from_pretrained('roberta-base')

    # 3. Update configuration to match the custom tokenizer
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    
    # 4. Instantiate the model with random weights
    model = RobertaForMaskedLM(config=config)
    
    print(f"✅ Model configured successfully. Parameters: {model.num_parameters():,}")
    return model, tokenizer

def main():
    """
    Main function to orchestrate the data loading, model configuration,
    and training process.
    """
    print("--- Starting RoBERTa Training Pipeline ---")

    # Step 1: Configure the model and tokenizer
    model, tokenizer = configure_model_and_tokenizer(TOKENIZER_FILE)
    if not model:
        return # Exit if configuration fails

    # Step 2: Load and prepare the dataset
    print(f"\n📚 Loading and tokenizing the dataset from '{CORPUS_FILE}'...")
    # Load the text file as a dataset object
    dataset = load_dataset('text', data_files={'train': CORPUS_FILE})

    def tokenize_function(examples):
        # The tokenizer will process the text and we truncate to a max length.
        # 512 is a standard max length for BERT-style models.
        return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

    # Apply the tokenization to the entire dataset
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=4, # Use multiple processes to speed up tokenization
        remove_columns=['text'] # Remove the original text column
    )
    print("✅ Dataset prepared.")

    # Step 3: Set up the Data Collator
    # The data collator is a crucial component for Masked Language Modeling (MLM).
    # It takes a batch of tokenized examples and dynamically creates the masks.
    # This is the "dynamic masking" that makes RoBERTa effective.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15 # Standard masking probability
    )
    print("✅ Data collator for MLM configured.")

    # Step 4: Define Training Arguments
    # These arguments control every aspect of the training run.
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=10,  # As per BabyLM 2025 competition rules
        per_device_train_batch_size=16, # Adjust based on your GPU memory
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16= False, #For NVIDIA GPUs
        use_mps_device=True, # For Apple Silicon
        logging_steps=500,
    )
    print("✅ Training arguments defined.")

    # Step 5: Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
    )

    # Step 6: Start Training!
    print("\n🚀🚀🚀 Starting model training... 🚀🚀🚀")
    trainer.train()
    print("🏁 Training finished!")

    # Step 7: Save the final model
    print(f"💾 Saving final model to '{MODEL_OUTPUT_DIR}'...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    print("✅ Model saved. Your custom-trained RoBERTa is ready!")


if __name__ == '__main__':
    main()