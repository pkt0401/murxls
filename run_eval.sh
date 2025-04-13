#!/bin/bash

# Script to run the eval.py for evaluating cross-lingual summarization outputs

# Default values
MODEL="gpt3.5"
SRC_LANG="english"
TGT_LANG="thai"
BERTSCORE=false
PLOT=false
SUMMARIES_PATH=""

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL       Specify model (mistral, llama, gpt3.5, gpt4)"
    echo "  -s, --source LANGUAGE   Source language (english, thai, gujarati, marathi, pashto, burmese, sinhala)"
    echo "  -t, --target LANGUAGE   Target language (english, thai, gujarati, marathi, pashto, burmese, sinhala)"
    echo "  -b, --bertscore         Calculate BERTScore (requires additional packages)"
    echo "  -p, --plot              Generate visualization plots"
    echo "  -f, --file PATH         Path to generated summaries file (overrides default path)"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Example: $0 -m mistral -s english -t thai -b -p"
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
        -b|--bertscore)
            BERTSCORE=true
            shift
            ;;
        -p|--plot)
            PLOT=true
            shift
            ;;
        -f|--file)
            SUMMARIES_PATH="--summaries_path $2"
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

# Check if CrossSum_dataset directory exists
if [[ ! -d "./CrossSum_dataset" && ! -d "../CrossSum_dataset" && ! -d "../../CrossSum_dataset" ]]; then
    echo "Warning: CrossSum_dataset directory not found. Make sure it exists with the correct structure:"
    echo "CrossSum_dataset/"
    echo "├── train/"
    echo "├── test/"
    echo "└── val/"
fi

# Build command with options
CMD="python eval.py --model $MODEL --src_lang $SRC_LANG --tgt_lang $TGT_LANG"

if [[ "$BERTSCORE" == "true" ]]; then
    CMD="$CMD --bertscore"
fi

if [[ "$PLOT" == "true" ]]; then
    CMD="$CMD --plot"
fi

if [[ -n "$SUMMARIES_PATH" ]]; then
    CMD="$CMD $SUMMARIES_PATH"
fi

# Run the script
echo "Running evaluation with:"
echo "- Model: $MODEL"
echo "- Source language: $SRC_LANG"
echo "- Target language: $TGT_LANG"
echo "- BERTScore: $BERTSCORE"
echo "- Plot: $PLOT"
if [[ -n "$SUMMARIES_PATH" ]]; then
    echo "- Summaries path: ${SUMMARIES_PATH#--summaries_path }"
fi

# Execute Python script
$CMD
