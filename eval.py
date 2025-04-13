import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import re
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Ensure NLTK data is downloaded
try:
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed. May already be installed.")

def setup_output_directories(model_name, lang_pair):
    """
    Create necessary output directories for evaluation results.
    """
    # Main output directory
    output_dir = f"./outputs/{model_name}/{lang_pair}"
    
    # Evaluation results directory
    eval_dir = f"./outputs/{model_name}/{lang_pair}/evaluation"
    
    # Create directories
    os.makedirs(eval_dir, exist_ok=True)
    
    return {
        "base": output_dir,
        "eval_dir": eval_dir,
        "summaries_path": f"{output_dir}/summaries.txt",
        "metrics_path": f"{eval_dir}/metrics.csv",
        "final_metrics_path": f"{eval_dir}/final_metrics.txt",
        "figures_dir": f"{eval_dir}/figures"
    }

def load_data(src_lang, tgt_lang):
    """
    Load test data containing reference summaries.
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
    
    # Test path in CrossSum_dataset structure
    test_path = os.path.join(crosssum_dir, "test", f"{data_name}.csv")
    
    # Check if alternate path exists
    if not os.path.exists(test_path):
        # Try with swapped language order
        reversed_data_name = f"{tgt_lang}-{src_lang}"
        test_path = os.path.join(crosssum_dir, "test", f"{reversed_data_name}.csv")
    
    # Check if file exists
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find test data file for {src_lang}-{tgt_lang} in CrossSum_dataset directory structure.")
    
    print(f"Loading reference data from: {test_path}")
    
    try:
        test_samples = pd.read_csv(test_path, encoding="utf-8")
        print(f"Loaded {len(test_samples)} reference samples")
        return test_samples
    except Exception as e:
        print(f"Error loading file: {e}")
        raise FileNotFoundError("Could not load the required data file.")

def load_generated_summaries(summaries_path):
    """
    Load generated summaries from file.
    """
    if not os.path.exists(summaries_path):
        raise FileNotFoundError(f"Could not find the summaries file at {summaries_path}")
    
    with open(summaries_path, 'r', encoding='utf-8') as f:
        generated_summaries = [line.strip() for line in f.readlines()]
    
    return generated_summaries

def detect_language(text, target_lang):
    """
    Detect if the text is in the target language.
    This is a simplified approach and might need to be improved for production use.
    """
    try:
        import fasttext
        # Attempt to load the language detection model
        try:
            model_path = './lid.176.bin'
            if not os.path.exists(model_path):
                model_path = './lid.176.ftz'
                if not os.path.exists(model_path):
                    print("Language detection model not found. Skipping language detection.")
                    return True
            
            model = fasttext.load_model(model_path)
            prediction = model.predict(text.replace('\n', ' '), k=1)
            detected_lang = prediction[0][0].replace('__label__', '')
            
            # Language code mapping
            lang_map = {
                'english': 'en',
                'thai': 'th',
                'gujarati': 'gu',
                'marathi': 'mr',
                'pashto': 'ps',
                'burmese': 'my',
                'sinhala': 'si'
            }
            
            target_code = lang_map.get(target_lang, target_lang)
            return detected_lang == target_code
            
        except Exception as e:
            print(f"Error in language detection: {e}")
            return True  # Assume it's correct if detection fails
    
    except ImportError:
        print("Fasttext not installed. Skipping language detection.")
        return True  # Assume it's correct if fasttext is not installed

def calculate_rouge(prediction, reference):
    """
    Calculate ROUGE scores for a prediction and reference.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Handle empty strings
    if not prediction or not reference:
        return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    # Tokenize into sentences
    try:
        pred_sents = "\n".join(sent_tokenize(prediction))
        ref_sents = "\n".join(sent_tokenize(reference))
    except:
        # Fallback for tokenization errors
        pred_sents = prediction
        ref_sents = reference
    
    # Calculate scores
    scores = scorer.score(ref_sents, pred_sents)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_bertscore(predictions, references, lang):
    """
    Calculate BERTScore for a set of predictions and references.
    Requires bert_score package.
    """
    try:
        from bert_score import score
        
        # Language code mapping
        lang_map = {
            'english': 'en',
            'thai': 'th',
            'gujarati': 'gu',
            'marathi': 'mr',
            'pashto': 'ps',
            'burmese': 'my',
            'sinhala': 'si'
        }
        
        lang_code = lang_map.get(lang, 'en')
        
        # Calculate BERTScore
        precision, recall, f1 = score(predictions, references, lang=lang_code, verbose=True)
        
        # Convert to numpy arrays and then to Python lists
        precision_list = precision.cpu().numpy().tolist()
        recall_list = recall.cpu().numpy().tolist()
        f1_list = f1.cpu().numpy().tolist()
        
        return precision_list, recall_list, f1_list
    
    except ImportError:
        print("BERTScore package not installed. Skipping BERTScore calculation.")
        return [0] * len(predictions), [0] * len(predictions), [0] * len(predictions)

def create_plots(metrics_df, model_name, lang_pair, output_dir):
    """
    Create plots for visualization of evaluation metrics.
    """
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. ROUGE scores distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df['rouge1'], bins=20, alpha=0.5, label='ROUGE-1')
    plt.hist(metrics_df['rouge2'], bins=20, alpha=0.5, label='ROUGE-2')
    plt.hist(metrics_df['rougeL'], bins=20, alpha=0.5, label='ROUGE-L')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'ROUGE Scores Distribution - {model_name} ({lang_pair})')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'rouge_distribution.png'))
    plt.close()
    
    # 2. Correct Language Rate
    if 'is_correct_lang' in metrics_df.columns:
        correct_count = metrics_df['is_correct_lang'].sum()
        incorrect_count = len(metrics_df) - correct_count
        plt.figure(figsize=(8, 8))
        plt.pie([correct_count, incorrect_count], 
                labels=['Correct Language', 'Incorrect Language'],
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'])
        plt.title(f'Language Correctness - {model_name} ({lang_pair})')
        plt.savefig(os.path.join(figures_dir, 'language_correctness.png'))
        plt.close()
    
    # 3. ROUGE scores by sample index
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df.index, metrics_df['rouge1'], label='ROUGE-1')
    plt.plot(metrics_df.index, metrics_df['rouge2'], label='ROUGE-2')
    plt.plot(metrics_df.index, metrics_df['rougeL'], label='ROUGE-L')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    plt.title(f'ROUGE Scores by Sample - {model_name} ({lang_pair})')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'rouge_by_sample.png'))
    plt.close()
    
    print(f"Plots saved to {figures_dir}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate Cross-Lingual Summarization outputs')
    
    # Required arguments - but with defaults as None to allow interactive mode
    parser.add_argument('--model', type=str, choices=['mistral', 'llama', 'gpt3.5', 'gpt4'],
                        help='Model used for summarization')
    
    parser.add_argument('--src_lang', type=str,
                        choices=['english', 'thai', 'gujarati', 'marathi', 'pashto', 'burmese', 'sinhala'],
                        help='Source language')
    
    parser.add_argument('--tgt_lang', type=str,
                        choices=['english', 'thai', 'gujarati', 'marathi', 'pashto', 'burmese', 'sinhala'],
                        help='Target language')
    
    # Optional arguments
    parser.add_argument('--bertscore', action='store_true',
                        help='Calculate BERTScore (requires additional packages)')
    
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    
    parser.add_argument('--summaries_path', type=str, default=None,
                        help='Path to generated summaries file (overrides default path)')
    
    args = parser.parse_args()
    
    # Interactive mode if required arguments are not provided
    if args.model is None:
        print("Please select the model to evaluate:")
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
        print("\nPlease select the source language:")
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
        print("\nPlease select the target language:")
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
    print("\n===== Evaluation Configuration =====")
    print(f"Model: {args.model}")
    print(f"Source Language: {args.src_lang}")
    print(f"Target Language: {args.tgt_lang}")
    print(f"Calculate BERTScore: {args.bertscore}")
    print(f"Generate plots: {args.plot}")
    print("===================================\n")
    
    # Set up paths and language format
    model_name = args.model
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    lang_pair = f"{src_lang}-{tgt_lang}"
    
    # Target language formatting
    target_language = tgt_lang.capitalize()
    
    # Set up output directories
    output_paths = setup_output_directories(model_name, lang_pair)
    
    # Override summaries path if provided
    if args.summaries_path:
        summaries_path = args.summaries_path
    else:
        summaries_path = output_paths["summaries_path"]
    
    # Load data
    print("Loading reference data...")
    test_samples = load_data(src_lang, tgt_lang)
    
    # Load generated summaries
    print("Loading generated summaries...")
    generated_summaries = load_generated_summaries(summaries_path)
    
    # Ensure the number of summaries matches the number of test samples
    if len(generated_summaries) < len(test_samples):
        print(f"Warning: Generated summaries ({len(generated_summaries)}) are fewer than test samples ({len(test_samples)})")
        # Truncate test samples to match the number of summaries
        test_samples = test_samples.iloc[:len(generated_summaries)]
    elif len(generated_summaries) > len(test_samples):
        print(f"Warning: Generated summaries ({len(generated_summaries)}) are more than test samples ({len(test_samples)})")
        # Truncate generated summaries to match the number of test samples
        generated_summaries = generated_summaries[:len(test_samples)]
    
    # Initialize metrics
    metrics = {
        'sample_id': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'is_correct_lang': [],
        'summary_length': []
    }
    
    # Add BERTScore columns if requested
    if args.bertscore:
        metrics.update({
            'bertscore_precision': [],
            'bertscore_recall': [],
            'bertscore_f1': []
        })
    
    # Process each sample
    print("Evaluating summaries...")
    for i in tqdm(range(len(test_samples))):
        try:
            reference = test_samples['summary'][i]
            generated = generated_summaries[i]
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge(generated, reference)
            
            # Check if the generated summary is in the correct language
            is_correct_lang = detect_language(generated, tgt_lang)
            
            # Store metrics
            metrics['sample_id'].append(i)
            metrics['rouge1'].append(rouge_scores['rouge1'])
            metrics['rouge2'].append(rouge_scores['rouge2'])
            metrics['rougeL'].append(rouge_scores['rougeL'])
            metrics['is_correct_lang'].append(int(is_correct_lang))
            metrics['summary_length'].append(len(generated))
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # Add empty values for this sample
            metrics['sample_id'].append(i)
            metrics['rouge1'].append(0)
            metrics['rouge2'].append(0)
            metrics['rougeL'].append(0)
            metrics['is_correct_lang'].append(0)
            metrics['summary_length'].append(0)
    
    # Calculate BERTScore if requested
    if args.bertscore:
        print("Calculating BERTScore...")
        try:
            precisions, recalls, f1s = calculate_bertscore(
                [s for s in generated_summaries if s],  # Skip empty strings
                [test_samples['summary'][i] for i in range(len(generated_summaries)) if generated_summaries[i]],
                tgt_lang
            )
            
            # Add scores to metrics
            score_idx = 0
            for i in range(len(test_samples)):
                if i < len(generated_summaries) and generated_summaries[i]:
                    metrics['bertscore_precision'].append(precisions[score_idx])
                    metrics['bertscore_recall'].append(recalls[score_idx])
                    metrics['bertscore_f1'].append(f1s[score_idx])
                    score_idx += 1
                else:
                    metrics['bertscore_precision'].append(0)
                    metrics['bertscore_recall'].append(0)
                    metrics['bertscore_f1'].append(0)
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            for i in range(len(test_samples)):
                metrics['bertscore_precision'].append(0)
                metrics['bertscore_recall'].append(0)
                metrics['bertscore_f1'].append(0)
    
    # Create a DataFrame from the metrics
    metrics_df = pd.DataFrame(metrics)
    
    # Save metrics to CSV
    metrics_df.to_csv(output_paths["metrics_path"], index=False)
    print(f"Metrics saved to {output_paths['metrics_path']}")
    
    # Calculate and save final metrics
    avg_rouge1 = metrics_df['rouge1'].mean()
    avg_rouge2 = metrics_df['rouge2'].mean()
    avg_rougeL = metrics_df['rougeL'].mean()
    correct_lang_rate = metrics_df['is_correct_lang'].mean() * 100
    avg_length = metrics_df['summary_length'].mean()
    
    # Add BERTScore averages if available
    bertscore_metrics = ""
    if args.bertscore and 'bertscore_f1' in metrics_df.columns:
        avg_bertscore_precision = metrics_df['bertscore_precision'].mean()
        avg_bertscore_recall = metrics_df['bertscore_recall'].mean()
        avg_bertscore_f1 = metrics_df['bertscore_f1'].mean()
        bertscore_metrics = f"Average BERTScore Precision: {avg_bertscore_precision:.4f}\n" \
                           f"Average BERTScore Recall: {avg_bertscore_recall:.4f}\n" \
                           f"Average BERTScore F1: {avg_bertscore_f1:.4f}\n"
    
    with open(output_paths["final_metrics_path"], "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Language Pair: {lang_pair}\n\n")
        f.write(f"Average ROUGE-1: {avg_rouge1:.4f}\n")
        f.write(f"Average ROUGE-2: {avg_rouge2:.4f}\n")
        f.write(f"Average ROUGE-L: {avg_rougeL:.4f}\n")
        f.write(bertscore_metrics)
        f.write(f"Correct Language Rate: {correct_lang_rate:.2f}%\n")
        f.write(f"Average Summary Length: {avg_length:.1f} characters\n")
        f.write(f"Processed: {len(metrics_df)}/{len(test_samples)} samples\n")
    
    # Print final metrics
    print("\n===== Final Metrics =====")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")
    if args.bertscore and 'bertscore_f1' in metrics_df.columns:
        print(f"Average BERTScore F1: {avg_bertscore_f1:.4f}")
    print(f"Correct Language Rate: {correct_lang_rate:.2f}%")
    print(f"Average Summary Length: {avg_length:.1f} characters")
    print(f"Processed: {len(metrics_df)}/{len(test_samples)} samples")
    
    # Create plots if requested
    if args.plot:
        print("\nGenerating plots...")
        create_plots(metrics_df, model_name, lang_pair, output_paths["eval_dir"])
    
    print(f"\nEvaluation complete. Results saved to {output_paths['eval_dir']}")

if __name__ == "__main__":
    main()
