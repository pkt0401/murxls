import fasttext
import langid
from huggingface_hub import hf_hub_download
import os
import pandas as pd
import glob
import argparse
import re

class LID:
    """
    Language identification class that supports FLORES language codes.
    """
    def __init__(self, target_lang_code):
        try:
            model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
            self.model = fasttext.load_model(model_path)
        except Exception as e:
            print(f"Error loading fasttext model: {str(e)}")
            print("Trying to use local model if available...")
            # Try loading model from local paths
            local_paths = ["./lid.176.bin", "./lid.176.ftz", "./model.bin"]
            model_loaded = False
            for path in local_paths:
                if os.path.exists(path):
                    try:
                        self.model = fasttext.load_model(path)
                        print(f"Loaded model from local path: {path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"Error loading from {path}: {str(e)}")
            
            if not model_loaded:
                raise ValueError("Could not load fasttext model from any location")
        
        # Target language code in FLORES format
        self.target_lang_code = target_lang_code
        
        # Expected fastText label format for the target language (e.g., __label__eng_Latn)
        self.target_fasttext_label = f"__label__{self.target_lang_code}"
        
        print(f"Target language code: {self.target_lang_code}")
        print(f"Expected fastText label: {self.target_fasttext_label}")
    
    def __call__(self, predictions, *, source_texts=None):
        def correct_lang(idx, response):
            # Remove newline characters
            response = response.replace("\n", " ")
            prediction = self.model.predict(response)
            detected_label = prediction[0][0]  # Example: __label__eng_Latn
            confidence = prediction[1][0]
            
            # Debug output (only for first 5 samples)
            if idx < 5:
                print(f"\nSample {idx}: {response[:50]}...")
                print(f"Detected label: {detected_label} with confidence: {confidence:.4f}")
                print(f"Expected label: {self.target_fasttext_label}")
                print(f"Match: {detected_label == self.target_fasttext_label}")
                
                # Validate with a second method using langid
                langid_lang, langid_conf = langid.classify(response)
                print(f"langid detection: {langid_lang} with confidence: {langid_conf:.4f}")
                print("---")
            
            return int(detected_label == self.target_fasttext_label)
        
        scores = []
        correct = 0
        denom = 0
        skip = 0
        
        for i, response in enumerate(predictions):
            if len(response) > 20:
                is_correct = correct_lang(i, response)
                scores.append(is_correct)
                correct += is_correct
                denom += 1
            else:
                scores.append(-1)
                skip += 1
        
        print(f"\nPercentage of skipped samples: {float(skip)/len(predictions):.2%}")
        accuracy = correct / denom if denom > 0 else 0
        print(f"Language accuracy: {accuracy:.2%}")
        
        # Additional analysis: Display the most frequently detected languages
        print("\nLanguage detection statistics:")
        detected_langs = {}
        for response in predictions:
            if len(response) > 20:
                lang_label = self.model.predict(response)[0][0]
                detected_langs[lang_label] = detected_langs.get(lang_label, 0) + 1
        
        print("Most detected languages:")
        for lang, count in sorted(detected_langs.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"{lang}: {count} ({count/denom:.1%})")
        
        return accuracy, scores

def find_text_files(output_dir, pattern):
    """
    Find files matching the given pattern in the output directory.
    Also checks for files in the directory structure created by main.py.
    """
    # Direct pattern matching
    files = glob.glob(os.path.join(output_dir, pattern))
    
    # Check for files in the 'outputs' directory structure from main.py
    model_dirs = glob.glob(os.path.join(output_dir, "outputs", "*"))
    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            lang_dirs = glob.glob(os.path.join(model_dir, "*-*"))
            for lang_dir in lang_dirs:
                if os.path.basename(lang_dir).lower() in pattern.lower():
                    summary_file = os.path.join(lang_dir, "summaries.txt")
                    if os.path.exists(summary_file):
                        files.append(summary_file)
    
    return files

def is_data_file(filename):
    """
    Determine whether the filename is a data file that should be evaluated.
    """
    # Exclude filenames that contain 'metrics' or '_results'
    if 'metrics' in filename or '_results' in filename:
        return False
    
    # Include the summaries.txt file (generated by main.py)
    if filename == 'summaries.txt':
        return True
        
    # Check for language pair pattern (e.g., english-thai, thai-english, etc.)
    lang_pair_pattern = r'(english|thai|gujarati|marathi|pashto|burmese|sinhala)-(english|thai|gujarati|marathi|pashto|burmese|sinhala)'
    if re.search(lang_pair_pattern, filename):
        return True
    
    return False

def process_text_file(file_path, target_lang_code):
    """
    Read a text file and perform language identification evaluation.
    The file should have one text sample per line.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    
    # Read text file by trying multiple encodings
    encodings_to_try = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
    texts = None
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                texts = [line.strip() for line in f.readlines()]
            break
        except UnicodeDecodeError:
            continue
    
    if texts is None:
        # If all encoding attempts fail, try binary mode
        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
                texts = [line.strip() for line in binary_data.decode('utf-8', errors='replace').splitlines()]
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None, None
    
    # Remove empty lines
    texts = [t for t in texts if t.strip()]
    
    if not texts:
        print(f"No text content found in {file_path}")
        return None, None
    
    print(f"Loaded {len(texts)} text samples from {file_path}")
    
    # Print sample texts
    print("\nSample texts:")
    for i, text in enumerate(texts[:2]):
        print(f"{i+1}: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Initialize the LID evaluator
    try:
        evaluator = LID(target_lang_code)
    except Exception as e:
        print(f"Error initializing LID evaluator: {str(e)}")
        return None, None
    
    # Perform language identification evaluation
    accuracy, scores = evaluator(texts)
    
    # Analyze and save results to a DataFrame
    results = pd.DataFrame({
        'text': texts,
        'is_correct_language': scores
    })
    
    # Save results as a CSV file
    output_dir = os.path.dirname(file_path)
    output_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_lid_results.csv"
    output_path = os.path.join(output_dir, output_name)
    
    try:
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {str(e)}")
    
    return accuracy, results

def process_language_pairs(files_dir, language_pairs, specific_pair=None):
    """
    Perform language identification on multiple language pair files.
    
    Args:
        files_dir: Directory containing the files.
        language_pairs: Mapping of language pairs and file patterns.
        specific_pair: Process only this specific language pair if specified (default: None means all pairs).
    """
    results = {}
    
    # Process a specific language pair or all pairs
    pairs_to_process = {specific_pair: language_pairs[specific_pair]} if specific_pair else language_pairs
    
    for pair_name, (source_lang, target_lang, file_pattern) in pairs_to_process.items():
        print(f"\n{'='*50}")
        print(f"Processing language pair: {pair_name} ({source_lang} -> {target_lang})")
        print(f"{'='*50}")
        
        # Find matching files in the directory
        file_paths = find_text_files(files_dir, file_pattern)
        file_paths = [f for f in file_paths if is_data_file(os.path.basename(f))]
        
        if not file_paths:
            print(f"No files found matching pattern: {file_pattern}")
            continue
        
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            accuracy, _ = process_text_file(file_path, target_lang)
            
            if accuracy is not None:
                results[file_path] = {
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'accuracy': accuracy
                }
    
    # Summarize the results
    print("\n" + "="*120)
    print("Summary of language evaluation:")
    print("="*120)
    print(f"{'File':<70} | {'Language Pair':<20} | {'Accuracy':<10}")
    print("-"*120)
    
    for file_path, info in results.items():
        file_name = os.path.basename(file_path)
        pair = f"{info['source_lang']} -> {info['target_lang']}"
        print(f"{file_name:<70} | {pair:<20} | {info['accuracy']:.2%}")
    
    # Save summary results to a CSV file
    df_results = pd.DataFrame([
        {
            'file': file_path,
            'source_language': info['source_lang'],
            'target_language': info['target_lang'],
            'accuracy': info['accuracy']
        }
        for file_path, info in results.items()
    ])
    
    output_path = os.path.join(files_dir, "language_accuracy_results.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results

# Language pairs definition (source language, target language, file pattern)
# Using FLORES language codes for language identification.
language_pairs = {
    'English-Thai': ('eng_Latn', 'tha_Thai', '*english-thai*'),
    'English-Gujarati': ('eng_Latn', 'guj_Gujr', '*english-gujarati*'),
    'English-Marathi': ('eng_Latn', 'mar_Deva', '*english-marathi*'),
    'English-Pashto': ('eng_Latn', 'pbt_Arab', '*english-pashto*'),
    'English-Burmese': ('eng_Latn', 'mya_Mymr', '*english-burmese*'),
    'English-Sinhala': ('eng_Latn', 'sin_Sinh', '*english-sinhala*'),
    'Thai-English': ('tha_Thai', 'eng_Latn', '*thai-english*'),
    'Gujarati-English': ('guj_Gujr', 'eng_Latn', '*gujarati-english*'),
    'Marathi-English': ('mar_Deva', 'eng_Latn', '*marathi-english*'),
    'Pashto-English': ('pbt_Arab', 'eng_Latn', '*pashto-english*'),
    'Burmese-English': ('mya_Mymr', 'eng_Latn', '*burmese-english*'),
    'Sinhala-English': ('sin_Sinh', 'eng_Latn', '*sinhala-english*')
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate language identification for generated summaries")
    parser.add_argument('--dir', type=str, default=".", help="Directory containing generated summary files")
    parser.add_argument('--language_pair', type=str, default=None, 
                        choices=list(language_pairs.keys()), 
                        help="Specific language pair to evaluate (default: all)")
    parser.add_argument('--file', type=str, default=None, 
                        help="Specific file to evaluate (overrides the language_pair option)")
    
    args = parser.parse_args()
    
    # Process a single file if specified
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        
        # Infer language pair from the file name
        file_name = os.path.basename(args.file)
        language_pair = None
        target_lang_code = None
        
        for pair_name, (source_lang, target_lang, pattern) in language_pairs.items():
            if re.search(pattern.replace('*', '.*'), file_name, re.IGNORECASE):
                language_pair = pair_name
                target_lang_code = target_lang
                break
        
        if not target_lang_code:
            print("Could not determine target language from file name.")
            print("Please specify the language pair manually using the --language_pair option.")
            return
        
        print(f"Processing single file: {args.file}")
        print(f"Detected language pair: {language_pair}")
        process_text_file(args.file, target_lang_code)
    
    # Otherwise, process files based on language pairs
    else:
        process_language_pairs(args.dir, language_pairs, args.language_pair)

if __name__ == "__main__":
    main()
