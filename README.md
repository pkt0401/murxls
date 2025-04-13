# MuRXLS: Cross-Lingual Summarization for Low-Resource Languages using Multilingual Retrieval-Based In-Context Learning

This repository implements the Cross-Lingual Summarization for Low-Resource Languages using Multilingual Retrieval-Based In-Context Learning (MuRXLS) approach described in the paper "Cross-Lingual Summarization for Low-Resource Languages using Multilingual Retrieval-Based In-Context Learning".

MuRXLS is a robust framework that dynamically selects the most relevant summarization examples for each article using multilingual retrieval, enhancing cross-lingual summarization performance particularly for low-resource languages.

## Features

- Support for 12 language pairs (6 languages paired with English in both directions)
- Compatible with multiple LLMs (Mistral-7B, Llama-3-70B, GPT-3.5, GPT-4)
- Similarity-based retrieval for optimal example selection
- Comprehensive evaluation tools (ROUGE, BERTScore, Language Identification)
- Interactive or command-line usage

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for Mistral and Llama models)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/murxls.git
   cd murxls
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the CrossSum_dataset directory with the following structure:
   ```
   CrossSum_dataset/
   ├── train/
   ├── test/
   │   ├── english-thai.csv
   │   ├── english-gujarati.csv
   │   └── [other language pairs].csv
   └── val/
       ├── english-thai.csv
       ├── english-gujarati.csv
       └── [other language pairs].csv
   ```

4. Additional setup for language identification (optional):
   ```bash
   # Download the fastText language identification model
   mkdir -p models
   wget -P models https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
   ```

## Dataset Structure

The CrossSum_dataset contains cross-lingual summarization data with the following structure:

```
CrossSum_dataset/
├── train/                       # Training split
│   ├── english-thai.csv
│   ├── english-gujarati.csv
│   └── ...
├── test/                        # Test split
│   ├── english-thai.csv
│   ├── english-gujarati.csv
│   └── ...
└── val/                         # Validation split
    ├── english-thai.csv
    ├── english-gujarati.csv
    └── ...
```

Each CSV file contains the following columns:
- `text`: Source language content
- `summary`: Target language summary

## Usage

The toolkit consists of four main components:

1. `main.py`: Generates cross-lingual summaries
2. `eval.py`: Evaluates summaries using ROUGE metrics
3. `bertscore_eval.py`: Evaluates summaries using BERTScore
4. `lid_eval.py`: Evaluates language identification accuracy

### Generation (main.py)

#### Using the bash script:

```bash
# Basic usage
./run_main.sh -m gpt3.5 -s english -t thai

# With more options
./run_main.sh --model mistral --source burmese --target english --shots 2 --retrieval similarity
```

#### Direct python usage:

```bash
# Interactive mode
python main.py

# Command-line arguments
python main.py --model llama --src_lang gujarati --tgt_lang english --shots 1 --retrieval similarity
```

### Evaluation (eval.py)

#### Using the bash script:

```bash
# Basic evaluation
./run_eval.sh -m gpt3.5 -s english -t thai

# With visualization
./run_eval.sh --model mistral --source thai --target english --bertscore --plot
```

#### Direct python usage:

```bash
# Interactive mode
python eval.py

# Command-line arguments
python eval.py --model llama --src_lang english --tgt_lang marathi --plot
```

### BERTScore Evaluation (bertscore_eval.py)

#### Using the bash script:

```bash
# Basic usage
./run_bertscore.sh -g ./outputs/gpt3.5 -l Thai-English

# With custom model
./run_bertscore.sh --generated-dir ./outputs/mistral --reference-dir ./CrossSum_dataset --model-type microsoft/mdeberta-v3-base
```

#### Direct python usage:

```bash
python bertscore_eval.py --generated_dir ./outputs --language_pair English-Thai
```

### Language Identification Evaluation (lid_eval.py)

#### Using the bash script:

```bash
# Evaluate all language pairs in a directory
./run_lid.sh -d ./outputs/mistral

# Evaluate a specific file
./run_lid.sh -f ./outputs/gpt3.5/english-thai/summaries.txt
```

#### Direct python usage:

```bash
python lid_eval.py --dir ./outputs/llama --language_pair Marathi-English
```

## Supported Languages

The toolkit supports the following language pairs:

- English ↔ Thai
- English ↔ Gujarati
- English ↔ Marathi
- English ↔ Pashto
- English ↔ Burmese
- English ↔ Sinhala

## Supported Models

- **Mistral-7B-Instruct-v0.3**: Open-source 7B parameter instruction-tuned model
- **Llama-3-70B-Instruct**: Meta's 70B parameter instruction-tuned model
- **GPT-3.5-Turbo**: OpenAI's GPT-3.5 model (requires API key)
- **GPT-4o-mini**: OpenAI's GPT-4 model (requires API key)

## Output Structure

Generated outputs and evaluation results are organized as follows:

```
outputs/
├── [model_name]/
│   ├── [src_lang]-[tgt_lang]/
│   │   ├── summaries.txt             # Generated summaries
│   │   ├── samples/                  # Individual sample files
│   │   └── evaluation/               # Evaluation results
│   │       ├── metrics.csv           # Detailed metrics
│   │       ├── final_metrics.txt     # Summary metrics
│   │       └── figures/              # Visualization plots
│   └── logs/                         # Log files
└── bertscore_evaluation_results.csv  # Overall BERTScore results
```

## Citation

If you use this code in your research, please cite:

```
@article{park2025crosslingual,
  title={Cross-Lingual Summarization for Low-Resource Languages using Multilingual Retrieval-Based In-Context Learning},
  author={Park, Gyutae and Park, Jeonghyun and Lee, Hwanhee},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.main.py)

#### Using the bash script:

```bash
# Basic usage
./run_main.sh -m gpt3.5 -s english -t thai

# With more options
./run_main.sh --model mistral --source burmese --target english --shots 2 --retrieval similarity
```

#### Direct python usage:

```bash
# Interactive mode
python main.py

# Command-line arguments
python main.py --model llama --src_lang gujarati --tgt_lang english --shots 1 --retrieval similarity
```

### Evaluation (eval.py)

#### Using the bash script:

```bash
# Basic evaluation
./run_eval.sh -m gpt3.5 -s english -t thai

# With visualization
./run_eval.sh --model mistral --source thai --target english --bertscore --plot
```

#### Direct python usage:

```bash
# Interactive mode
python eval.py

# Command-line arguments
python eval.py --model llama --src_lang english --tgt_lang marathi --plot
```

### BERTScore Evaluation (bertscore_eval.py)

#### Using the bash script:

```bash
# Basic usage
./run_bertscore.sh -g ./outputs/gpt3.5 -l Thai-English

# With custom model
./run_bertscore.sh --generated-dir ./outputs/mistral --reference-dir ./CrossSum_low --model-type microsoft/mdeberta-v3-base
```

#### Direct python usage:

```bash
python bertscore_eval.py --generated_dir ./outputs --language_pair English-Thai
```

### Language Identification Evaluation (lid_eval.py)

#### Using the bash script:

```bash
# Evaluate all language pairs in a directory
./run_lid.sh -d ./outputs/mistral

# Evaluate a specific file
./run_lid.sh -f ./outputs/gpt3.5/english-thai/summaries.txt
```

#### Direct python usage:

```bash
python lid_eval.py --dir ./outputs/llama --language_pair Marathi-English
```

## Supported Languages

The toolkit supports the following language pairs:

- English ↔ Thai
- English ↔ Gujarati
- English ↔ Marathi
- English ↔ Pashto
- English ↔ Burmese
- English ↔ Sinhala

## Supported Models

- **Mistral-7B-Instruct-v0.3**: Open-source 7B parameter instruction-tuned model
- **Llama-3-70B-Instruct**: Meta's 70B parameter instruction-tuned model
- **GPT-3.5-Turbo**: OpenAI's GPT-3.5 model (requires API key)
- **GPT-4o-mini**: OpenAI's GPT-4 model (requires API key)

## Output Structure

Generated outputs and evaluation results are organized as follows:

```
outputs/
├── [model_name]/
│   ├── [src_lang]-[tgt_lang]/
│   │   ├── summaries.txt             # Generated summaries
│   │   ├── samples/                  # Individual sample files
│   │   └── evaluation/               # Evaluation results
│   │       ├── metrics.csv           # Detailed metrics
│   │       ├── final_metrics.txt     # Summary metrics
│   │       └── figures/              # Visualization plots
│   └── logs/                         # Log files
└── bertscore_evaluation_results.csv  # Overall BERTScore results
```

## Citation

If you use this code in your research, please cite:

```
@article{park2025crosslingual,
  title={Cross-Lingual Summarization for Low-Resource Languages using Multilingual Retrieval-Based In-Context Learning},
  author={Park, Gyutae and Park, Jeonghyun and Lee, Hwanhee},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
