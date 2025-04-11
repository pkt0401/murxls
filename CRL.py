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
    언어 식별을 위한 클래스로, FLORES 형식의 언어 코드를 지원합니다.
    """
    def __init__(self, target_lang_code):
        try:
            model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
            self.model = fasttext.load_model(model_path)
        except Exception as e:
            print(f"Error loading fasttext model: {str(e)}")
            print("Trying to use local model if available...")
            # 로컬 경로에서 모델 로드 시도
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
        
        # 대상 언어 코드 (FLORES 형식)
        self.target_lang_code = target_lang_code
        
        # 타겟 언어 감지를 위한 fastText 레이블 형식 (예: __label__eng_Latn)
        self.target_fasttext_label = f"__label__{self.target_lang_code}"
        
        print(f"Target language code: {self.target_lang_code}")
        print(f"Expected fastText label: {self.target_fasttext_label}")
    
    def __call__(self, predictions, *, source_texts=None):
        def correct_lang(idx, response):
            response = response.replace("\n", " ")
            prediction = self.model.predict(response)
            detected_label = prediction[0][0]  # 예: __label__eng_Latn
            confidence = prediction[1][0]
            
            # 디버깅용 출력 (처음 5개 샘플만)
            if idx < 5:
                print(f"\nSample {idx}: {response[:50]}...")
                print(f"Detected label: {detected_label} with confidence: {confidence:.4f}")
                print(f"Expected label: {self.target_fasttext_label}")
                print(f"Match: {detected_label == self.target_fasttext_label}")
                
                # 두 번째 방식으로도 검증 (langid 사용)
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
                iscorrect = correct_lang(i, response)
                scores.append(iscorrect)
                correct += iscorrect
                denom += 1
            else:
                scores.append(-1)
                skip += 1
        
        print(f"\nPercentage of skipped samples: {float(skip)/len(predictions):.2%}")
        accuracy = correct/denom if denom > 0 else 0
        print(f"Language accuracy: {accuracy:.2%}")
        
        # 추가 분석: 가장 많이 감지된 언어 출력
        print("\nLanguage detection statistics:")
        detected_langs = {}
        for i, response in enumerate(predictions):
            if len(response) > 20:
                lang_label = self.model.predict(response)[0][0]
                detected_langs[lang_label] = detected_langs.get(lang_label, 0) + 1
        
        print("Most detected languages:")
        for lang, count in sorted(detected_langs.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"{lang}: {count} ({count/denom:.1%})")
        
        return accuracy, scores

def find_text_files(output_dir, pattern):
    """
    주어진 출력 디렉토리와 파일 패턴에 맞는 파일들을 찾습니다.
    """
    # 직접 패턴 일치 검색
    files = glob.glob(os.path.join(output_dir, pattern))
    
    # main.py 출력 구조에 맞는 파일 검색 (summaries.txt)
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
    파일명이 평가해야 할 데이터 파일인지 확인합니다.
    """
    # metrics 또는 results가 포함된 파일명은 제외
    if 'metrics' in filename or '_results' in filename:
        return False
    
    # summaries.txt 파일 포함 (main.py에서 생성된 요약 파일)
    if filename == 'summaries.txt':
        return True
        
    # 언어 쌍 패턴 확인 (예: english-thai, thai-english 등)
    lang_pair_pattern = r'(english|thai|gujarati|marathi|pashto|burmese|sinhala)-(english|thai|gujarati|marathi|pashto|burmese|sinhala)'
    if re.search(lang_pair_pattern, filename):
        return True
    
    return False

def process_text_file(file_path, target_lang_code):
    """
    텍스트 파일을 읽고 LID 평가를 수행합니다.
    파일은 각 줄이 하나의 텍스트 샘플을 포함하는 형식이어야 합니다.
    """
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    
    # 텍스트 파일 읽기 (여러 인코딩 시도)
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
        # 모든 인코딩 시도 실패 시 바이너리 모드로 시도
        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
                texts = [line.strip() for line in binary_data.decode('utf-8', errors='replace').splitlines()]
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None, None
    
    # 빈 줄 제거
    texts = [t for t in texts if t.strip()]
    
    if not texts:
        print(f"No text content found in {file_path}")
        return None, None
    
    print(f"Loaded {len(texts)} text samples from {file_path}")
    
    # 샘플 텍스트 출력
    print("\nSample texts:")
    for i, text in enumerate(texts[:2]):
        print(f"{i+1}: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # LID 평가기 초기화
    try:
        evaluator = LID(target_lang_code)
    except Exception as e:
        print(f"Error initializing LID evaluator: {str(e)}")
        return None, None
    
    # 언어 식별 평가 수행
    accuracy, scores = evaluator(texts)
    
    # 결과 분석 및 저장
    results = pd.DataFrame({
        'text': texts,
        'is_correct_language': scores
    })
    
    # 결과 저장
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
    여러 언어 쌍 파일에 대해 언어 식별을 수행합니다.
    
    Args:
        files_dir: 파일들이 위치한 디렉토리
        language_pairs: 언어 쌍과 파일명 패턴의 매핑
        specific_pair: 특정 언어 쌍만 처리하려면 지정 (기본값: None, 모든 쌍 처리)
    """
    results = {}
    
    # 특정 언어 쌍 처리 또는 모든 언어 쌍 처리
    pairs_to_process = {specific_pair: language_pairs[specific_pair]} if specific_pair else language_pairs
    
    for pair_name, (source_lang, target_lang, file_pattern) in pairs_to_process.items():
        print(f"\n{'='*50}")
        print(f"Processing language pair: {pair_name} ({source_lang} -> {target_lang})")
        print(f"{'='*50}")
        
        # 패턴에 맞는 파일 찾기
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
    
    # 결과 요약
    print("\n" + "="*120)
    print("Summary of language evaluation:")
    print("="*120)
    print(f"{'File':<70} | {'Language Pair':<20} | {'Accuracy':<10}")
    print("-"*120)
    
    for file_path, info in results.items():
        file_name = os.path.basename(file_path)
        pair = f"{info['source_lang']} -> {info['target_lang']}"
        print(f"{file_name:<70} | {pair:<20} | {info['accuracy']:.2%}")
    
    # 결과를 CSV로 저장
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

# 언어 쌍 정의 (소스 언어, 타겟 언어, 파일 패턴)
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
                        help="Specific file to evaluate (overrides language_pair option)")
    
    args = parser.parse_args()
    
    # 단일 파일 처리
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        
        # 파일 이름에서 언어 쌍 추론
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
            print("Please specify language pair manually using --language_pair option.")
            return
        
        print(f"Processing single file: {args.file}")
        print(f"Detected language pair: {language_pair}")
        process_text_file(args.file, target_lang_code)
    
    # 언어 쌍 기반 처리
    else:
        process_language_pairs(args.dir, language_pairs, args.language_pair)

if __name__ == "__main__":
    main()