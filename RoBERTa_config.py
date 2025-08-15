import os
from transformers import RobertaConfig, RobertaForMaskedLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer

# --- Configuration ---
# Path to the directory where you saved your trained tokenizer.
TOKENIZER_DIR = 'morpheme_tokenizer'
TOKENIZER_FILE = os.path.join(TOKENIZER_DIR, 'tokenizer.json')

def configure_roberta_from_scratch(tokenizer_path):
    """
    Configures a RoBERTa-base model for training from scratch with a custom tokenizer.
    
    This function performs three key steps:
    1. Loads your custom-trained morpheme tokenizer from its JSON file.
    2. Loads the standard RoBERTa-base architecture configuration.
    3. Instantiates the model with randomly initialized weights, ensuring its
       vocabulary size and special token IDs match your tokenizer.
    """
    print(f"⚙️  Configuring RoBERTa model using tokenizer from: '{tokenizer_path}'")

    if not os.path.exists(tokenizer_path):
        print(f"❌ Error: Tokenizer file not found at '{tokenizer_path}'.")
        print("Please make sure the path is correct and you have run the tokenizer training script.")
        return None, None

    # 1. Load your custom morpheme tokenizer directly from the .json file
    # This avoids the AutoTokenizer error which looks for a config.json file.
    print("Loading custom tokenizer from file...")
    raw_tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Wrap the loaded tokenizer with PreTrainedTokenizerFast to make it
    # compatible with the rest of the Hugging Face ecosystem.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )
    print(f"Tokenizer loaded successfully. Vocabulary size: {tokenizer.vocab_size}")

    # 2. Load the standard RoBERTa-base configuration
    # This gives us the standard architecture settings (number of layers, heads, etc.)
    print("Loading RoBERTa-base standard configuration...")
    config = RobertaConfig.from_pretrained('roberta-base')

    # 3. CRITICAL STEP: Update the configuration to match your tokenizer
    # The model's embedding layer size and its understanding of special tokens
    # MUST match the tokenizer's vocabulary and settings.
    print("Updating configuration to match custom tokenizer...")
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    
    # You can also adjust other parameters here if needed for your experiments,
    # for example, reducing the number of layers for a smaller model.
    # config.num_hidden_layers = 6 

    # 4. Instantiate the model from the MODIFIED configuration
    # This creates the RoBERTa architecture with the correct embedding size,
    # but with all weights randomly initialized. It is NOT pre-trained.
    print("Instantiating model from scratch with the new configuration...")
    model = RobertaForMaskedLM(config=config)
    
    print(f"✅ Model configured successfully!")
    print(f"   - Number of parameters: {model.num_parameters():,}")
    
    return model, tokenizer

def test_configuration(model, tokenizer):
    """
    A simple test to verify that the model and tokenizer are compatible.
    """
    print("\n🔬 Running a quick test...")
    
    # Your morphemes will look different, this is just an example.
    test_sentence = "un happy ness is bake d" 
    
    print(f"Original text: '{test_sentence}'")
    
    # Tokenize the text
    inputs = tokenizer(test_sentence, return_tensors="pt")
    print(f"Tokenized IDs: {inputs['input_ids']}")
    
    # Pass through the model
    try:
        outputs = model(**inputs)
        print("Model forward pass successful. Output shape:", outputs.logits.shape)
        print("✅ Configuration test passed!")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")


if __name__ == '__main__':
    # Execute the configuration and testing
    model, tokenizer = configure_roberta_from_scratch(TOKENIZER_FILE)
    
    if model and tokenizer:
        test_configuration(model, tokenizer)
        print("\nNext step: Use this configured 'model' and 'tokenizer' in your training script (Step 6).")