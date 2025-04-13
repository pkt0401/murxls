#!/bin/bash

# Script to run the main.py for cross-lingual summarization generation

# Default values
MODEL="gpt3.5"
SRC_LANG="english"
TGT_LANG="thai"
SHOTS=1
RETRIEVAL="similarity"
MAX_SAMPLES=""

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL       Specify model (mistral, llama, gpt3.5, gpt4)"
    echo "  -s, --source LANGUAGE   Source language (english, thai, gujarati, marathi, pashto, burmese, sinhala)"
    echo "  -t, --target LANGUAGE   Target language (english, thai, gujarati, marathi, pashto, burmese, sinhala)"
    echo "  -n, --shots NUMBER      Number of shots for in-context learning (0, 1, or 2)"
    echo "  -r, --retrieval METHOD  Retrieval method (similarity or shortest)"
    echo "  -x, --max-samples NUM   Max number of samples to process (for testing)"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Example: $0 -m mistral -s english -t thai -n 1 -r similarity"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -s|--source)
            SRC_LANG="$2"
            shift 2
            ;;
        -t|--target)
            TGT_LANG="$2"
            shift 2
            ;;
        -n|--shots)
            SHOTS="$2"
            shift 2
            ;;
        -r|--retrieval)
            RETRIEVAL="$2"
            shift 2
            ;;
        -x|--max-samples)
            MAX_SAMPLES="--max_samples $2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate model
if [[ ! "$MODEL" =~ ^(mistral|llama|gpt3.5|gpt4)$ ]]; then
    echo "Error: Invalid model. Choose from mistral, llama, gpt3.5, or gpt4."
    exit 1
fi

# Validate languages
VALID_LANGS=("english" "thai" "gujarati" "marathi" "pashto" "burmese" "sinhala")
VALID_SRC=false
VALID_TGT=false

for lang in "${VALID_LANGS[@]}"; do
    if [[ "$SRC_LANG" == "$lang" ]]; then
        VALID_SRC=true
    fi
    if [[ "$TGT_LANG" == "$lang" ]]; then
        VALID_TGT=true
    fi
done

if [[ "$VALID_SRC" == "false" ]]; then
    echo "Error: Invalid source language."
    exit 1
fi

if [[ "$VALID_TGT" == "false" ]]; then
    echo "Error: Invalid target language."
    exit 1
fi

if [[ "$SRC_LANG" == "$TGT_LANG" ]]; then
    echo "Error: Source and target languages must be different."
    exit 1
fi

# Validate shots
if [[ ! "$SHOTS" =~ ^[0-2]$ ]]; then
    echo "Error: Shots must be 0, 1, or 2."
    exit 1
fi

# Validate retrieval method
if [[ ! "$RETRIEVAL" =~ ^(similarity|shortest)$ ]]; then
    echo "Error: Retrieval method must be 'similarity' or 'shortest'."
    exit 1
fi

# Check if CrossSum_dataset directory exists
if [[ ! -d "./CrossSum_dataset" && ! -d "../CrossSum_dataset" && ! -d "../../CrossSum_dataset" ]]; then
    echo "Warning: CrossSum_dataset directory not found. Make sure it exists with the correct structure:"
    echo "CrossSum_dataset/"
    echo "├── train/"
    echo "├── test/"
    echo "└── val/"
fi

# Run the script
echo "Running cross-lingual summarization with:"
echo "- Model: $MODEL"
echo "- Source language: $SRC_LANG"
echo "- Target language: $TGT_LANG"
echo "- Shots: $SHOTS"
echo "- Retrieval method: $RETRIEVAL"
if [[ -n "$MAX_SAMPLES" ]]; then
    echo "- Max samples: ${MAX_SAMPLES#--max_samples }"
fi

# Execute Python script
python main.py --model "$MODEL" --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" --shots "$SHOTS" --retrieval "$RETRIEVAL" $MAX_SAMPLES
