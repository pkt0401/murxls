import argparse
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from FlagEmbedding import BGEM3FlagModel  # BGE-M3 model for multilingual retrieval

# LLM imports based on selected model
def load_model_and_tokenizer(model_name):
    """
    Load the specified model and tokenizer based on model_name.
    Returns the model, tokenizer, and a client object for API-based models.
    """
    model, tokenizer, client = None, None, None
    
    if model_name.startswith("mistral"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_name} model...")
        try:
            # Try to load locally first
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                local_files_only=True
            )
            print("Models loaded from local cache")
        except Exception as e:
            print(f"Local loading failed: {e}")
            print("Attempting to download models (this may take a while)...")
            # Download if local load fails
            model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
            
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token
            
    elif model_name.startswith("llama"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_name} model...")
        try:
            # Try to load locally first
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3-70B-Instruct",
                local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3-70B-Instruct",
                local_files_only=True
            )
            print("Models loaded from local cache")
        except Exception as e:
            print(f"Local loading failed: {e}")
            print("Attempting to download models (this may take a while)...")
            # Download if local load fails
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70B-Instruct")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70B-Instruct")
            
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token
            
    elif model_name.startswith(("gpt-3.5", "gpt3.5")):
        import openai
        from openai import OpenAI
        
        # Get API key with interactive input if not in environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Please enter your OpenAI API key: ")
            if not api_key:
                raise ValueError("API key is required for GPT models")
            
        openai.api_key = api_key
        client = OpenAI(api_key=api_key)
        model_id = "gpt-3.5-turbo-0125"
        print(f"Using OpenAI API with model: {model_id}")
        
    elif model_name.startswith(("gpt-4", "gpt4")):
        import openai
        from openai import OpenAI
        
        # Get API key with interactive input if not in environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Please enter your OpenAI API key: ")
            if not api_key:
                raise ValueError("API key is required for GPT models")
            
        openai.api_key = api_key
        client = OpenAI(api_key=api_key)
        model_id = "gpt-4o-mini"  # Using GPT-4o-mini as default
        print(f"Using OpenAI API with model: {model_id}")
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return model, tokenizer, client

def load_data(src_lang, tgt_lang):
    """
    Load validation and test data for the given language pair.
    Uses data from the CrossSum_dataset directory.
    """
    data_name = f"{src_lang}-{tgt_lang}"  # e.g., "english-thai"
    
    # Try to find the CrossSum_dataset directory
    crosssum_dirs = [
        "./CrossSum_dataset/",
        "../CrossSum_dataset/",
        "../../CrossSum_dataset/"
    ]
    
    crosssum_dir = None
    for d in crosssum_dirs:
        if os.path.exists(d):
            crosssum_dir = d
            break
    
    if not crosssum_dir:
        raise FileNotFoundError("Could not find CrossSum_dataset directory. Please make sure it exists in the current or parent directory.")
    
    # Test and val paths in CrossSum_dataset structure
    val_path = os.path.join(crosssum_dir, "val", f"{data_name}.csv")
    test_path = os.path.join(crosssum_dir, "test", f"{data_name}.csv")
    
    # Check if alternate paths exist
    if not (os.path.exists(val_path) and os.path.exists(test_path)):
        # Try with swapped language order
        reversed_data_name = f"{tgt_lang}-{src_lang}"
        val_path = os.path.join(crosssum_dir, "val", f"{reversed_data_name}.csv")
        test_path = os.path.join(crosssum_dir, "test", f"{reversed_data_name}.csv")
    
    # Check if files exist
    if not os.path.exists(val_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find data files for {src_lang}-{tgt_lang} in CrossSum_dataset directory structure.")
    
    print(f"Loading data from: \nVal: {val_path} \nTest: {test_path}")
    
    try:
        test_samples = pd.read_csv(test_path, encoding="utf-8")
        shot_samples = pd.read_csv(val_path, encoding='utf-8')
        print(f"Loaded {len(test_samples)} test samples and {len(shot_samples)} validation samples")
        return test_samples, shot_samples, val_path, test_path
    except Exception as e:
        print(f"Error loading files: {e}")
        raise FileNotFoundError("Could not load the required data files.")


def setup_output_directories(model_name, lang_pair):
    """
    Create necessary output directories for results and logs.
    """
    output_name = f"{model_name}_{lang_pair}"
    
    # Main output directory
    output_dir = f"./outputs/{model_name}/{lang_pair}"
    
    # Log directory
    log_dir = f"./outputs/{model_name}/logs"
    
    # Individual sample outputs
    samples_dir = f"./outputs/{model_name}/{lang_pair}/samples"
    
    # Create all directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    return {
        "base": output_dir,
        "output_path": f"{output_dir}/summaries.txt",
        "log_path": f"{log_dir}/{lang_pair}_log.txt",
        "samples_dir": samples_dir,
        "final_metrics_path": f"{output_dir}/generation_stats.txt"
    }

def find_most_similar_example(query_text, shot_samples, retriever):
    """
    Find the most similar example from validation set for retrieval-based ICL.
    """
    try:
        # Compute query embedding
        query_embedding = retriever.encode([query_text], batch_size=1, max_length=8192)['dense_vecs']
        
        # Compute embeddings for all examples
        example_texts = shot_samples['text'].tolist()
        example_embeddings = retriever.encode(example_texts, batch_size=8, max_length=8192)['dense_vecs']
        
        # Calculate cosine similarity
        similarities = np.dot(query_embedding, example_embeddings.T)[0]
        
        # Get most similar example
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[most_similar_idx]
        
        print(f"Selected example index: {most_similar_idx} with similarity score: {similarity_score:.4f}")
        
        return most_similar_idx, float(similarity_score)
    except Exception as e:
        print(f"Error finding similar example: {e}")
        return 0, 0.0  # Return first example as fallback

def create_prompt(test_text, shot_text, shot_summary, target_language, num_shots=1):
    """
    Create the prompt for the model based on the number of shots.
    """
    one_shot_example = f"""Please summarize the following text in {target_language}
Text: {shot_text}
Translated summary: {shot_summary}
"""
    
    query_prompt = f"""Please summarize the following text in {target_language}
Text: {test_text}
Translated summary:"""
    
    final_prompt = one_shot_example + "\n" + query_prompt
    
    return final_prompt

def generate_summary_mistral_llama(model, tokenizer, prompt, device):
    """
    Generate summary using local Mistral or Llama model.
    """
    try:
        # Prepare chat input for the model
        messages = [{"role": "user", "content": prompt}]
        chat_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
        chat_input = chat_input.to(device)
        
        # Generate response with temperature 0.1 for more deterministic output
        with torch.no_grad():
            generated_ids = model.generate(
                chat_input,
                max_new_tokens=512,
                do_sample=True,  # Changed to True to enable temperature
                temperature=0.1,  # Added temperature parameter
                top_p=0.9,        # Added top_p parameter
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode response
        full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        # Clean up memory
        del chat_input, generated_ids
        torch.cuda.empty_cache()
        
        return full_output
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

def generate_summary_gpt(client, prompt, model_id):
    """
    Generate summary using OpenAI API models (GPT-3.5, GPT-4).
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.1,  # Using consistent temperature with Mistral/Llama
            top_p=0.9,
            max_tokens=512
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return ""

def extract_summary_from_response(response, model_type="mistral"):
    """
    Extract the actual summary from model responses, which may contain extra text.
    Different models require different extraction methods.
    """
    if model_type.startswith("llama"):
        return clean_summary_llama(response)
    else:
        # For Mistral, GPT and other models
        # Look for "Translated summary:" pattern
        trans_idx = response.rfind("Translated summary:")
        if trans_idx != -1:
            # Get text after this marker
            # Check if there's another prompt after this
            next_prompt_idx = response.find("Please summarize", trans_idx)
            if next_prompt_idx != -1:
                return response[trans_idx + len("Translated summary:"):next_prompt_idx].strip()
            else:
                return response[trans_idx + len("Translated summary:"):].strip()
        
        # Look for assistant tag (for chat models)
        assist_idx = response.rfind("<|im_start|>assistant")
        if assist_idx != -1:
            end_idx = response.find("<|im_end|>", assist_idx)
            if end_idx != -1:
                assistant_text = response[assist_idx + len("<|im_start|>assistant"):end_idx].strip()
            else:
                assistant_text = response[assist_idx + len("<|im_start|>assistant"):].strip()
                
            # Check for "Translated summary:" in assistant text
            trans_idx = assistant_text.rfind("Translated summary:")
            if trans_idx != -1:
                return assistant_text[trans_idx + len("Translated summary:"):].strip()
            else:
                return assistant_text
        
        # Look for [/INST] tag (for Mistral)
        inst_idx = response.rfind("[/INST]")
        if inst_idx != -1:
            next_prompt_idx = response.find("Please summarize", inst_idx)
            if next_prompt_idx != -1:
                return response[inst_idx + len("[/INST]"):next_prompt_idx].strip()
            else:
                return response[inst_idx + len("[/INST]"):].strip()
        
        # If no patterns matched, return the whole response
        return response.strip()

def clean_summary_llama(summary):
    """
    Post-process summaries generated by Llama 3 models
    """
    import re
    
    # Remove special tokens and tags
    cleaned = (summary.replace("<s>", "").replace("</s>", "")
              .replace("<|im_start|>", "").replace("<|im_end|>", "")
              .replace("<|eot_id|>", "").replace("<|start_header_id|>", "")
              .replace("<|end_header_id|>", "").replace("assistant", "")
              .replace("<assistant>", "").replace("</assistant>", "")
              .replace("[/INST]", "").replace("[INST]", ""))
    
    # Remove English introduction text
    intro_patterns = [
        r"Here is the summary of the text in.*?:",
        r"Here's the summary in.*?:",
        r"The summary in.*?:",
        r"Summary in.*?:",
        r"Translated summary:"
    ]
    
    for pattern in intro_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Handle incomplete Unicode characters
    # Process incomplete summaries that contain '�' character
    if '�' in cleaned:
        # Keep only up to the last complete sentence
        sentences = re.split(r'([.!?।\n])', cleaned)
        valid_sentences = []
        current = ""
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                segment = sentences[i] + sentences[i+1]
            else:
                segment = sentences[i]
                
            if '�' not in segment:
                valid_sentences.append(segment)
            else:
                # Add only text before the '�'
                valid_part = segment.split('�')[0].strip()
                if valid_part:
                    valid_sentences.append(valid_part)
                break
        
        cleaned = ''.join(valid_sentences)
    
    # Remove repeated text (Llama3 sometimes repeats the same sentence)
    sentences = re.split(r'([.!?।\n])', cleaned)
    processed_sentences = []
    seen_sentences = set()
    
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i+1]).strip()
        else:
            sentence = sentences[i].strip()
            
        if sentence and sentence not in seen_sentences:
            processed_sentences.append(sentence)
            seen_sentences.add(sentence)
    
    cleaned = ' '.join(processed_sentences)
    
    # Replace multiple newlines with a single newline
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    return cleaned.strip()

# Removed the Rouge evaluation function since we're only doing generation

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Cross-Lingual Summarization with Retrieval-Based ICL')
    
    # Required arguments - but with defaults as None to allow interactive mode
    parser.add_argument('--model', type=str, choices=['mistral', 'llama', 'gpt3.5', 'gpt4'],
                        help='Model to use for summarization')
    
    parser.add_argument('--src_lang', type=str,
                        choices=['english', 'thai', 'gujarati', 'marathi', 'pashto', 'burmese', 'sinhala'],
                        help='Source language')
    
    parser.add_argument('--tgt_lang', type=str,
                        choices=['english', 'thai', 'gujarati', 'marathi', 'pashto', 'burmese', 'sinhala'],
                        help='Target language')
    
    # Optional arguments
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    parser.add_argument('--shots', type=int, default=1, choices=[0, 1, 2],
                        help='Number of shots for in-context learning')
    
    parser.add_argument('--retrieval', type=str, default='similarity', 
                        choices=['similarity', 'shortest'],
                        help='Method for selecting examples: similarity or shortest length')
    
    args = parser.parse_args()
    
    # Interactive mode if required arguments are not provided
    if args.model is None:
        print("Please select a model:")
        print("1. mistral (Mistral-7B-Instruct-v0.3)")
        print("2. llama (Llama-3-70B-Instruct)")
        print("3. gpt3.5 (GPT-3.5-turbo)")
        print("4. gpt4 (GPT-4o-mini)")
        model_choice = input("Enter your choice (1-4): ")
        model_options = {
            '1': 'mistral',
            '2': 'llama',
            '3': 'gpt3.5',
            '4': 'gpt4'
        }
        args.model = model_options.get(model_choice, 'mistral')
    
    if args.src_lang is None:
        print("\nPlease select a source language:")
        print("1. english")
        print("2. thai")
        print("3. gujarati")
        print("4. marathi")
        print("5. pashto")
        print("6. burmese")
        print("7. sinhala")
        lang_choice = input("Enter your choice (1-7): ")
        lang_options = {
            '1': 'english',
            '2': 'thai',
            '3': 'gujarati',
            '4': 'marathi',
            '5': 'pashto',
            '6': 'burmese',
            '7': 'sinhala'
        }
        args.src_lang = lang_options.get(lang_choice, 'english')
    
    if args.tgt_lang is None:
        print("\nPlease select a target language:")
        print("1. english")
        print("2. thai")
        print("3. gujarati")
        print("4. marathi")
        print("5. pashto")
        print("6. burmese")
        print("7. sinhala")
        lang_choice = input("Enter your choice (1-7): ")
        lang_options = {
            '1': 'english',
            '2': 'thai',
            '3': 'gujarati',
            '4': 'marathi',
            '5': 'pashto',
            '6': 'burmese',
            '7': 'sinhala'
        }
        args.tgt_lang = lang_options.get(lang_choice, 'english')
    
    # Validate languages are different
    if args.src_lang == args.tgt_lang:
        raise ValueError("Source and target languages must be different. Please run again with different languages.")
    
    # Print configuration
    print("\n===== Configuration =====")
    print(f"Model: {args.model}")
    print(f"Source Language: {args.src_lang}")
    print(f"Target Language: {args.tgt_lang}")
    print(f"Number of shots: {args.shots}")
    print(f"Retrieval method: {args.retrieval}")
    if args.max_samples:
        print(f"Processing {args.max_samples} samples (limited for testing)")
    print("=========================\n")
    
    # Set up paths and language format
    model_name = args.model
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    lang_pair = f"{src_lang}-{tgt_lang}"
    
    # Target language formatting (capitalize first letter for prompt)
    target_language = tgt_lang.capitalize()
    
    # Load data
    test_samples, shot_samples, shot_data_path, test_data_path = load_data(src_lang, tgt_lang)
    
    # Limit samples if specified
    if args.max_samples:
        test_samples = test_samples[:args.max_samples]
    
    # Set up output directories
    output_paths = setup_output_directories(model_name, lang_pair)
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer, client = load_model_and_tokenizer(model_name)
    
    # Move model to device if using local model
    if model is not None:
        model = model.to(device)
    
    # Load BGE-M3 model for retrieval
    print("Loading BGE-M3 model for retrieval...")
    retriever = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    print("BGE-M3 model loaded successfully")
    
    # Initialize trackers
    similarity_scores = []
    example_indices = []
    
    # Set up logger
    with open(output_paths["log_path"], "w", encoding="utf-8") as log_file:
        log_file.write(f"Starting cross-lingual summarization for {lang_pair} using {model_name}\n")
        log_file.write(f"Retrieval method: {args.retrieval}, Shots: {args.shots}\n\n")
    
    # Process each test sample
    print(f"Processing {len(test_samples)} test samples...")
    
    # Open output file for writing summaries
    with open(output_paths["output_path"], "w", encoding="utf-8") as output_file:
        # Process each test sample
        for i in tqdm(range(len(test_samples))):
            try:
                # Select example based on retrieval method
                if args.retrieval == 'similarity':
                    # Find most similar example using retrieval
                    example_idx, similarity = find_most_similar_example(test_samples['text'][i], shot_samples, retriever)
                else:
                    # Use shortest example (length-based selection)
                    text_lengths = [len(text) for text in shot_samples['text']]
                    example_idx = np.argmin(text_lengths)
                    # Calculate similarity anyway for comparison
                    _, similarity = find_most_similar_example(test_samples['text'][i], shot_samples, retriever)
                
                # Store example index and similarity
                example_indices.append(example_idx)
                similarity_scores.append(similarity)
                
                # Create the prompt
                prompt = create_prompt(
                    test_samples['text'][i],
                    shot_samples['text'][example_idx],
                    shot_samples['summary'][example_idx],
                    target_language,
                    args.shots
                )
                
                # Save prompt to file
                prompt_path = os.path.join(output_paths["samples_dir"], f"sample_{i}_prompt.txt")
                with open(prompt_path, "w", encoding="utf-8") as prompt_file:
                    prompt_file.write(prompt)
                
                # Generate summary based on model type
                if model_name.startswith(("mistral", "llama")):
                    full_output = generate_summary_mistral_llama(model, tokenizer, prompt, device)
                else:  # GPT models
                    model_id = "gpt-3.5-turbo-0125" if model_name.startswith("gpt3.5") else "gpt-4o-mini"
                    full_output = generate_summary_gpt(client, prompt, model_id)
                
                # Save full output to file
                full_output_path = os.path.join(output_paths["samples_dir"], f"sample_{i}_full_output.txt")
                with open(full_output_path, "w", encoding="utf-8") as full_output_file:
                    full_output_file.write(full_output)
                
                # Extract summary from response
                summary = extract_summary_from_response(full_output)
                
                # Save to output file
                output_file.write(summary + "\n")
                
                # Log basic information
                log_msg = f"Sample {i} - Similarity: {similarity:.4f}, Example used: {example_idx}"
                print(log_msg)
                with open(output_paths["log_path"], "a", encoding="utf-8") as log_file:
                    log_file.write(log_msg + "\n")
            
            except Exception as e:
                error_msg = f"Error processing sample {i}: {str(e)}"
                print(error_msg)
                with open(output_paths["log_path"], "a", encoding="utf-8") as log_file:
                    log_file.write(error_msg + "\n")
                    
                # Write empty line to output file
                output_file.write("\n")
    
    # Save basic statistics
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    with open(output_paths["final_metrics_path"], "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Language Pair: {lang_pair}\n")
        f.write(f"Retrieval Method: {args.retrieval}\n")
        f.write(f"Shots: {args.shots}\n\n")
        f.write(f"Average Similarity: {avg_similarity:.4f}\n")
        f.write(f"Processed: {len(similarity_scores)}/{len(test_samples)} samples\n")
    
    print("\n===== Generation Complete =====")
    print(f"Average Similarity: {avg_similarity:.4f}")
    print(f"Processed: {len(similarity_scores)}/{len(test_samples)} samples")
    print(f"\nProcessing complete. Results saved to {output_paths['base']}")

if __name__ == "__main__":
    main()
