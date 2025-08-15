import os
import glob
import re
import morfessor

# --- Configuration ---
# Path to the directory containing the BabyLM .train files
# Example: './babylm_data/babylm_10M/'
CORPUS_DIR = './Dataset/osfstorage-archive/text_data/train_10M'

# --- Important Installation Note ---
# This script requires the Morfessor library. Install it using pip:
# pip install morfessor

# --- Output File Paths ---
# This script will generate these files.
CONSOLIDATED_CORPUS_FILE = 'babylm_10M_consolidated.txt'
MODEL_OUTPUT_FILE = 'morfessor_10M.model'
SEGMENTED_CORPUS_FILE = 'babylm_10M_morphemes.txt'


def prepare_data(corpus_dir, consolidated_output):
    """
    Consolidates all .train files from the BabyLM corpus into a single,
    lowercase text file for Morfessor training and segmentation.
    """
    print(f"🔍 Starting data preparation from '{corpus_dir}'...")
    
    train_files = glob.glob(os.path.join(corpus_dir, '*.train'))
    
    if not train_files:
        print(f"❌ Error: No .train files found in '{corpus_dir}'. Please check the CORPUS_DIR path.")
        return False

    print(f"Found {len(train_files)} training files to process.")

    # Consolidate the entire corpus into a single file.
    # Morfessor works well with raw, clean text.
    with open(consolidated_output, 'w', encoding='utf-8') as outfile:
        for filepath in train_files:
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # We'll write the cleaned, lowercased line.
                    # This removes punctuation and ensures consistency.
                    clean_line = ' '.join(re.findall(r'\b\w+\b', line.lower()))
                    if clean_line: # Avoid writing empty lines
                        outfile.write(clean_line + '\n')
    
    print(f"✅ Consolidated and cleaned corpus saved to '{consolidated_output}'")
    return True

def train_morfessor(corpus_file, model_output):
    """
    Trains a Morfessor model directly using its Python API.
    """
    print("\n💪 Starting Morfessor model training...")
    print("This is generally faster than StateMorph but can still take time.")
    
    try:
        # Initialize Morfessor's I/O handler
        io = morfessor.MorfessorIO()
        
        # Read the training data from the consolidated corpus file as a generator
        train_data_generator = io.read_corpus_file(corpus_file)
        
        # Create a Morfessor Baseline model instance
        model = morfessor.BaselineModel()
        
        # Train the model on the data generator
        model.train_online(train_data_generator)
        
        # Save the trained model to a file
        io.write_binary_model_file(model_output, model)
        
        print(f"✅ Training complete. Model saved to '{model_output}'.")
        return True
    except Exception as e:
        print(f"❌ An error occurred during Morfessor training: {e}")
        print("Please ensure the 'morfessor' library is installed correctly ('pip install morfessor').")
        return False


def segment_corpus(consolidated_corpus, model_file, segmented_output):
    """
    Uses the trained Morfessor model to segment the entire corpus via the Python API.
    """
    print("\n✍️  Segmenting the full corpus with the trained model...")
    
    try:
        # Initialize Morfessor's I/O handler
        io = morfessor.MorfessorIO()
        
        # Load the trained model from the file
        model = io.read_binary_model_file(model_file)
        
        # Open the input and output files
        with open(consolidated_corpus, 'r', encoding='utf-8') as infile, \
             open(segmented_output, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # Process each line word by word
                words = line.strip().split()
                segmented_words = []
                for word in words:
                    # Segment each word and join the morphemes with spaces
                    morphs, _ = model.viterbi_segment(word)
                    segmented_words.append(' '.join(morphs))
                
                # Write the fully segmented line to the output file
                outfile.write(' '.join(segmented_words) + '\n')

        print(f"✅ Segmentation complete. Output saved to '{segmented_output}'.")
        return True
    except Exception as e:
        print(f"❌ An error occurred during corpus segmentation: {e}")
        return False

if __name__ == '__main__':
    # --- Main Execution ---
    print("--- Starting Morphological Segmentation Pipeline (using Morfessor) ---")
    
    # Step 1: Prepare the consolidated corpus from the BabyLM data
    if prepare_data(CORPUS_DIR, CONSOLIDATED_CORPUS_FILE):
        
        # Step 2: Train the Morfessor model on the prepared corpus
        if train_morfessor(CONSOLIDATED_CORPUS_FILE, MODEL_OUTPUT_FILE):
            
            # Step 3: Segment the entire corpus using the trained model
            if segment_corpus(CONSOLIDATED_CORPUS_FILE, MODEL_OUTPUT_FILE, SEGMENTED_CORPUS_FILE):
                print("\n🎉 Pipeline finished successfully!")
                print(f"Your final, segmented corpus is ready at: {SEGMENTED_CORPUS_FILE}")
                print("You can now use this file to train your Hugging Face tokenizer.")
    
    print("--- Pipeline Ended ---")