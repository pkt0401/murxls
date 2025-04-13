import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn as nn
import os
import warnings
import re
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
import csv
from FlagEmbedding import BGEM3FlagModel  # SentenceTransformer 대신 BGE-M3 모델 임포트

# NLTK 데이터 다운로드 (첫 실행 시 필요)
try:
    import nltk
    nltk.download('punkt', quiet=True)
except:
    print("NLTK 다운로드 중 오류가 발생했습니다. 이미 설치되어 있을 수 있습니다.")

# 모델명과 언어쌍 설정
model_name = "mis3"      # 모델명 (예: "mis3" for Mistral-3)
src_lang = "english"     # 원본 언어: 영어
tgt_lang = "thai"        # 대상 언어: 태국어
lang_pair = f"{src_lang}-{tgt_lang}"  # "english-thai"

# Try to import deepspeed, but have a fallback if it fails
try:
    import deepspeed
    use_deepspeed = True
    # Try DeepSpeed initialization, but catch errors
    try:
        deepspeed.init_distributed()
        deepspeed_initialized = True
    except (ImportError, RuntimeError, OSError) as e:
        print(f"DeepSpeed initialization failed: {e}")
        print("Continuing without DeepSpeed distributed mode...")
        deepspeed_initialized = False
except ImportError:
    print("DeepSpeed not available. Using standard PyTorch.")
    use_deepspeed = False
    deepspeed_initialized = False

# 모델 및 토크나이저 로드 - v0.3으로 변경
print("Loading Mistral model v0.3...")
try:
    # 먼저 로컬에서 로드 시도
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",  # v0.3 사용
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",  # v0.3 사용
        local_files_only=True
    )
    print("Models loaded from local cache")
except Exception as e:
    print(f"Local loading failed: {e}")
    print("Attempting to download models (this may take a while)...")
    # 로컬 로드 실패시 다운로드 시도
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")  # v0.3 사용
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")  # v0.3 사용

# BGE-M3 모델 로드 - SentenceTransformer 대신
print("Loading BGE-M3 model...")
retriever = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("BGE-M3 model loaded")

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Setup for multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Move model to device
model = model.to(device)

# Initialize DeepSpeed engine if available and initialized
if use_deepspeed and deepspeed_initialized:
    try:
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config={
                "train_micro_batch_size_per_gpu": 1,
                "fp16": {"enabled": True}
            }
        )
        use_model_engine = True
        print("DeepSpeed engine initialized successfully")
    except Exception as e:
        print(f"DeepSpeed engine initialization failed: {e}")
        print("Falling back to standard model...")
        use_model_engine = False
else:
    use_model_engine = False
    print("Using standard PyTorch (no DeepSpeed)")

# 데이터 경로 설정
data_name = f"{src_lang}-{tgt_lang}"
shot_data = f"./english_csv/val/{data_name}_val.csv"
test_data = f"./english_csv/test/{data_name}_test.csv"

# 출력 파일 경로 설정
output_name = f"{model_name}_v03_{lang_pair}_dynamic_example_bgem3"
output_dir = f"./mistral_outputs/dynamic_example/{output_name}"

# 디버깅 로그 디렉토리 생성
log_dir = f"./mistral_outputs/retrieval_logs"

# 개별 출력 디렉토리 생성 - 각 샘플별 전체 출력을 저장
samples_dir = f"./mistral_outputs/samples_output"

# 클렌징된 출력 디렉토리 생성
cleansed_output_dir = f"./mistral_outputs/cleansed_outputs"

# CSV 결과 파일 경로
csv_output_file = f"{output_dir}_results.csv"

# Create output directories if they don't exist
os.makedirs(os.path.dirname(output_dir), exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(cleansed_output_dir, exist_ok=True)

# 데이터 로드 - 인덱스 재설정 적용
print(f"Loading data from {shot_data} and {test_data}")
try:
    # 슬라이싱 시에도 인덱스를 0부터 시작하도록 reset_index 적용
    test_samples = pd.read_csv(test_data, encoding="utf-8")
    shot_samples = pd.read_csv(shot_data, encoding='utf-8')
    print(f"Loaded {len(test_samples)} test samples and {len(shot_samples)} validation samples")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    raise

# Define a variable to track any skipped items due to OOM
oom_list = []

# ROUGE 점수와 유사도 점수를 누적하기 위한 변수들 초기화
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
rougeLsum_scores = []

# 토크나이저 설정
tokenizer.pad_token = tokenizer.eos_token

# 샘플 수 제한 - 디버깅 목적으로 적은 수의 샘플만 처리
MAX_SAMPLES = len(test_samples)  # 전체 샘플 처리, 필요에 따라 조정

# 언어 설정
language = tgt_lang.capitalize()  # 첫 글자 대문자로 변환 (예: "thai" -> "Thai")

# 디버깅용 출력 내용 검사 함수
def inspect_response(response, index, debug_file):
    debug_file.write(f"===== SAMPLE {index} RESPONSE INSPECTION =====\n")
    debug_file.write(f"Original Length: {len(response)}\n")
    debug_file.write(f"First 200 chars: {response[:200]}...\n")
    debug_file.write(f"Last 200 chars: {response[-200:] if len(response) > 200 else response}\n")
    
    # 프롬프트 반복 패턴 검사
    prompt_repetitions = response.count("Please summarize the following text in")
    debug_file.write(f"Number of 'Please summarize' occurrences: {prompt_repetitions}\n")
    
    # 'Translated summary:' 패턴 검사
    translated_occurrences = response.count("Translated summary:")
    debug_file.write(f"Number of 'Translated summary:' occurrences: {translated_occurrences}\n")
    
    # Assistant/User 태그 검사
    assistant_tags = response.count("<|im_start|>assistant")
    user_tags = response.count("<|im_start|>user")
    debug_file.write(f"Number of assistant tags: {assistant_tags}\n")
    debug_file.write(f"Number of user tags: {user_tags}\n\n")
    
    return {
        "prompt_repetitions": prompt_repetitions,
        "translated_occurrences": translated_occurrences,
        "assistant_tags": assistant_tags,
        "user_tags": user_tags
    }

# 언어에 독립적인 추출 로직
def extract_translation_from_response(response, index, debug_file):
    # 응답 구조 분석
    inspection_result = inspect_response(response, index, debug_file)
    
    # 응답에서 요약 추출
    debug_file.write(f"===== EXTRACTION ATTEMPT FOR SAMPLE {index} =====\n")
    
    # 가능한 추출 방법들
    extracted_summary = None
    
    # 방법 1: 마지막 "Translated summary:" 이후의 모든 텍스트
    if inspection_result["translated_occurrences"] > 0:
        last_idx = response.rfind("Translated summary:")
        if last_idx != -1:
            # 다음 프롬프트까지의 텍스트만 추출
            next_prompt_idx = response.find("Please summarize", last_idx)
            if next_prompt_idx != -1:
                extracted_summary = response[last_idx + len("Translated summary:"):next_prompt_idx].strip()
            else:
                extracted_summary = response[last_idx + len("Translated summary:"):].strip()
            
            debug_file.write(f"Method 1 (Last 'Translated summary:'): Found at position {last_idx}\n")
            debug_file.write(f"Extracted: {extracted_summary[:100]}...\n\n")
    
    # 방법 2: assistant 태그 이후의 텍스트
    if extracted_summary is None and inspection_result["assistant_tags"] > 0:
        last_assistant_idx = response.rfind("<|im_start|>assistant")
        if last_assistant_idx != -1:
            end_idx = response.find("<|im_end|>", last_assistant_idx)
            if end_idx != -1:
                assistant_text = response[last_assistant_idx + len("<|im_start|>assistant"):end_idx].strip()
            else:
                assistant_text = response[last_assistant_idx + len("<|im_start|>assistant"):].strip()
            
            # 어시스턴트 텍스트에서 Translated summary: 검색
            trans_idx = assistant_text.rfind("Translated summary:")
            if trans_idx != -1:
                extracted_summary = assistant_text[trans_idx + len("Translated summary:"):].strip()
            else:
                extracted_summary = assistant_text
            
            debug_file.write(f"Method 2 (Assistant tag): Found at position {last_assistant_idx}\n")
            debug_file.write(f"Extracted: {extracted_summary[:100]}...\n\n")
    
    # 방법 3: [/INST] 태그 이후의 텍스트 (Mistral의 경우)
    if extracted_summary is None:
        inst_idx = response.rfind("[/INST]")
        if inst_idx != -1:
            # 다음 프롬프트까지만 추출
            next_prompt_idx = response.find("Please summarize", inst_idx)
            if next_prompt_idx != -1:
                extracted_summary = response[inst_idx + len("[/INST]"):next_prompt_idx].strip()
            else:
                extracted_summary = response[inst_idx + len("[/INST]"):].strip()
            
            debug_file.write(f"Method 3 ([/INST] tag): Found at position {inst_idx}\n")
            debug_file.write(f"Extracted: {extracted_summary[:100]}...\n\n")
    
    # 방법 4: 특수 패턴 - 반복되는 프롬프트 사이의 텍스트
    if extracted_summary is None and inspection_result["prompt_repetitions"] > 1:
        # 첫 번째 프롬프트 이후의 인덱스
        first_prompt_end = response.find("Translated summary:", response.find("Please summarize"))
        if first_prompt_end != -1:
            # 두 번째 프롬프트의 시작 인덱스
            second_prompt_start = response.find("Please summarize", first_prompt_end)
            if second_prompt_start != -1:
                extracted_summary = response[first_prompt_end + len("Translated summary:"):second_prompt_start].strip()
                debug_file.write(f"Method 4 (Between prompts): Extracted from {first_prompt_end} to {second_prompt_start}\n")
                debug_file.write(f"Extracted: {extracted_summary[:100]}...\n\n")
    
    # 추출 결과 정리
    if extracted_summary is None:
        debug_file.write("No extraction method worked. Using full response.\n\n")
        return response.strip()
    
    return extracted_summary

# 간소화된 후처리 함수
def clean_summary(summary):
    # 여러 줄 공백을 한 줄로 변경
    cleaned = re.sub(r'\n\s*\n', '\n', summary)
    
    return cleaned.strip()

# 가장 유사한 예제 찾기 함수 - BGE-M3 모델용으로 수정
def find_most_similar_example(query_text, shot_samples, retriever):
    """
    주어진 쿼리 텍스트와 가장 유사한 예제를 찾는 함수
    """
    try:
        # 쿼리 텍스트 임베딩
        query_embedding = retriever.encode([query_text], batch_size=1, max_length=8192)['dense_vecs']
        
        # 모든 예제 텍스트 임베딩
        example_texts = shot_samples['text'].tolist()
        example_embeddings = retriever.encode(example_texts, batch_size=12, max_length=8192)['dense_vecs']
        
        # 코사인 유사도 계산 - BGE-M3는 이미 정규화된 벡터를 반환하므로 내적(dot product)으로 계산
        similarities = np.dot(query_embedding, example_embeddings.T)[0]
        
        # 가장 유사한 예제의 인덱스 찾기
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[most_similar_idx]
        
        print(f"Selected example index: {most_similar_idx} with similarity score: {similarity_score:.4f}")
        
        return most_similar_idx, float(similarity_score)
    except Exception as e:
        print(f"Error finding similar example: {e}")
        # 오류가 발생하면 첫 번째 예제 사용
        return 0, 0.0

# CSV 결과 파일 생성 및 헤더 작성
with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['sample_id', 'query_text', 'example_text', 'example_summary', 
                         'raw_model_output', 'cleaned_output', 'reference_summary', 
                         'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'example_idx', 'similarity_score'])

# 디버깅 로그 파일 생성
debug_log_path = os.path.join(log_dir, f"{output_name}_extraction_debug.txt")
debug_file = open(debug_log_path, "w", encoding="utf-8")

# 예시 인덱스와 유사도 점수를 저장할 리스트
example_indices = []
similarity_scores = []

# 메인 생성 및 결과 저장
print(f"Starting inference for {MAX_SAMPLES} samples, writing results to {output_dir}.txt")
with open(output_dir + '.txt', 'w', encoding='utf-8') as fp:
    for i in tqdm(range(min(MAX_SAMPLES, len(test_samples)))):
        if i not in oom_list:
            try:
                # BGE-M3 모델로 가장 유사한 예시 찾기
                example_idx, similarity_score = find_most_similar_example(test_samples['text'][i], shot_samples, retriever)
                example_indices.append(example_idx)
                similarity_scores.append(similarity_score)
                
                # 이전 스타일의 간단한 프롬프트 형식 사용
                one_shot_sample = f"""Please summarize the following text in {language}
Text:{shot_samples['text'][example_idx]}
Translated summary:{shot_samples['summary'][example_idx]}
"""
                
                prompt = f"""Please summarize the following text in {language}
Text:{test_samples['text'][i]}
Translated summary:"""
                
                final_prompt = one_shot_sample + "\n" + prompt
                
                # 각 샘플의 프롬프트를 별도의 파일로 저장
                prompt_file_path = os.path.join(samples_dir, f"sample_{i}_prompt.txt")
                with open(prompt_file_path, "w", encoding="utf-8") as prompt_file:
                    prompt_file.write(final_prompt)
                
                # 채팅 템플릿 사용 (Mistral v0.3에 적합)
                messages = [
                    {"role": "user", "content": final_prompt}
                ]
                chat_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
                chat_input = chat_input.to(device)
                
                with torch.no_grad():
                    if use_model_engine:
                        # Use DeepSpeed engine
                        generated_ids = model_engine.module.generate(
                            chat_input, 
                            max_new_tokens=512, 
                            do_sample=False, 
                            pad_token_id=tokenizer.eos_token_id
                        )
                    else:
                        # Standard model generation
                        generated_ids = model.generate(
                            chat_input, 
                            max_new_tokens=512, 
                            do_sample=False, 
                            pad_token_id=tokenizer.eos_token_id
                        )
                
                # 전체 출력 디코딩
                full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                
                # 각 샘플의 완전한 출력을 별도의 파일로 저장
                sample_file_path = os.path.join(samples_dir, f"sample_{i}_full_output.txt")
                with open(sample_file_path, "w", encoding="utf-8") as sample_file:
                    sample_file.write("="*80 + "\n")
                    sample_file.write(f"PROMPT:\n")
                    sample_file.write("="*80 + "\n")
                    sample_file.write(final_prompt)
                    sample_file.write("\n\n")
                    
                    sample_file.write("="*80 + "\n")
                    sample_file.write(f"RAW MODEL OUTPUT:\n")
                    sample_file.write("="*80 + "\n")
                    sample_file.write(full_output)
                    sample_file.write("\n\n")
                
                # 메인 출력 파일에 저장
                fp.write(full_output.replace("\n", " ") + "\n")
                
                # Free up memory
                del chat_input, generated_ids
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM error for index {i}, skipping...")
                    oom_list.append(i)
                    fp.write('\n')
                    example_indices.append(-1)
                    similarity_scores.append(0.0)
                    
                    # CSV에 오류 행 추가
                    with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([i, "OOM ERROR", "", "", "", "", "", 0, 0, 0, 0, -1, 0])
                    
                    torch.cuda.empty_cache()
                else:
                    print(f"Error processing index {i}: {str(e)}")
                    fp.write('\n')
                    example_indices.append(-1)
                    similarity_scores.append(0.0)
                    
                    # CSV에 오류 행 추가
                    with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([i, f"ERROR: {str(e)[:100]}", "", "", "", "", "", 0, 0, 0, 0, -1, 0])
                    
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Unexpected error for index {i}: {str(e)}")
                fp.write('\n')
                example_indices.append(-1)
                similarity_scores.append(0.0)
                
                # CSV에 오류 행 추가
                with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([i, f"ERROR: {str(e)[:100]}", "", "", "", "", "", 0, 0, 0, 0, -1, 0])
        else:
            fp.write('\n')
            example_indices.append(-1)
            similarity_scores.append(0.0)
            
            # CSV에 건너뛴 행 추가
            with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([i, "SKIPPED", "", "", "", "", "", 0, 0, 0, 0, -1, 0])

# 출력 데이터를 미리 저장
original_responses = []
with open(output_dir + '.txt', "r", encoding="utf-8") as file:
    original_responses = file.readlines()

# 후처리 및 결과 저장
print("Processing generated outputs...")
cleansed_output_path = os.path.join(cleansed_output_dir, f"{os.path.basename(output_dir)}.txt")

# 인덱스 문제 해결: test_samples의 길이만큼만 처리
summaries = []
for i in range(min(len(original_responses), len(test_samples))):
    if i < len(original_responses):
        extracted_summary = extract_translation_from_response(original_responses[i], i, debug_file)
        cleaned_summary = clean_summary(extracted_summary)
        summaries.append(cleaned_summary)
    else:
        summaries.append("")  # 부족한 부분은 빈 문자열로 채움

# 추출된 요약을 파일로 저장
with open(cleansed_output_path, "w", encoding="utf-8") as file:
    file.write("\n".join(summaries))

print(f"Cleansed summaries saved to {cleansed_output_path}")

# 디버그 파일 닫기
debug_file.close()

# ROUGE 평가 - 수정된 방식으로 구현
print("Calculating ROUGE metrics...")

# 수정된 방식으로 참조 요약 가져오기
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
dec_summaries = summaries

# 인덱스 문제 해결: test_samples에서 직접 참조 요약 추출
rouge_scores = []
for i in range(len(summaries)):
    # test_samples 범위를 벗어나면 건너뜀
    if i >= len(test_samples):
        print(f"Skipping ROUGE calculation for sample {i}: index out of range")
        continue
        
    # 빈 요약 또는 오류가 있는 경우 건너뜀
    if not summaries[i]:
        print(f"Skipping ROUGE calculation for sample {i}: empty summary")
        continue
    
    try:
        # 직접 test_samples에서 참조 요약 추출
        ref_summary = test_samples['summary'][i]
        
        # 문장 토큰화 적용
        pred_sents = "\n".join(sent_tokenize(summaries[i]))
        ref_sents = "\n".join(sent_tokenize(ref_summary))
        
        scores = scorer.score(ref_sents, pred_sents)
        
        # 점수 추출
        rouge1 = scores['rouge1'][2]  # F1 점수
        rouge2 = scores['rouge2'][2]  # F1 점수
        rougeL = scores['rougeL'][2]  # F1 점수
        rougeLsum = scores['rougeLsum'][2]  # F1 점수
        
        rouge_scores.append({
            'rouge-1_f': rouge1, 
            'rouge-2_f': rouge2, 
            'rouge-l_f': rougeL,
            'rouge-l-sum_f': rougeLsum
        })
        
        # 예제 인덱스와 유사도 점수 가져오기
        example_idx = example_indices[i] if i < len(example_indices) else -1
        similarity_score = similarity_scores[i] if i < len(similarity_scores) else 0.0
        
        # CSV에 개별 점수 추가
        with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # 텍스트 필드에서 CSV 구분자 문제를 방지하기 위한 전처리
            query_text = test_samples['text'][i].replace('\n', ' ').replace('\r', '')
            
            if example_idx >= 0 and example_idx < len(shot_samples):
                example_text = shot_samples['text'][example_idx].replace('\n', ' ').replace('\r', '')
                example_summary = shot_samples['summary'][example_idx].replace('\n', ' ').replace('\r', '')
            else:
                example_text = ""
                example_summary = ""
                
            raw_output = original_responses[i].replace('\n', ' ').replace('\r', '') if i < len(original_responses) else ""
            cleaned_output_csv = summaries[i].replace('\n', ' ').replace('\r', '')
            reference_summary_csv = ref_summary.replace('\n', ' ').replace('\r', '')
            
            # CSV 행 작성
            csv_writer.writerow([
                i,  # sample_id
                query_text,  # query_text
                example_text,  # example_text
                example_summary,  # example_summary
                raw_output,  # raw_model_output
                cleaned_output_csv,  # cleaned_output
                reference_summary_csv,  # reference_summary
                rouge1,  # rouge1
                rouge2,  # rouge2
                rougeL,  # rougeL
                rougeLsum,  # rougeLsum
                example_idx,  # 사용된 예제 인덱스
                similarity_score  # 유사도 점수
            ])
        
    except Exception as e:
        print(f"Error calculating ROUGE for sample {i}: {e}")
        continue

# 평균 ROUGE 점수 계산 및 출력
if rouge_scores:
    print("eval_rouge1: ", round(sum([item['rouge-1_f'] for item in rouge_scores]) / len(rouge_scores), 4))
    print("eval_rouge2: ", round(sum([item['rouge-2_f'] for item in rouge_scores]) / len(rouge_scores), 4))
    print("eval_rougeL: ", round(sum([item['rouge-l_f'] for item in rouge_scores]) / len(rouge_scores), 4))
    print("eval_rougeLsum: ", round(sum([item['rouge-l-sum_f'] for item in rouge_scores]) / len(rouge_scores), 4))
    
    avg_rouge1 = round(sum([item['rouge-1_f'] for item in rouge_scores]) / len(rouge_scores), 4)
    avg_rouge2 = round(sum([item['rouge-2_f'] for item in rouge_scores]) / len(rouge_scores), 4)
    avg_rougeL = round(sum([item['rouge-l_f'] for item in rouge_scores]) / len(rouge_scores), 4)
    avg_rougeLsum = round(sum([item['rouge-l-sum_f'] for item in rouge_scores]) / len(rouge_scores), 4)
    
    # 최종 메트릭을 CSV에 추가
    with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([])  # 빈 행 추가
        csv_writer.writerow(["SUMMARY", "", "", "", "", "", "", "", "", "", "", "", ""])
        csv_writer.writerow(["Average", "", "", "", "", "", "", avg_rouge1, avg_rouge2, avg_rougeL, avg_rougeLsum, "", ""])
        csv_writer.writerow(["Processed", len(rouge_scores), "out of", MAX_SAMPLES, "", "", "", "", "", "", "", "", ""])
        csv_writer.writerow(["Skipped/Errors", len(oom_list), "", "", "", "", "", "", "", "", "", "", ""])
else:
    print("No valid samples for ROUGE calculation")

print(f"\nProcessing complete.")
print(f"All results are saved in CSV format at: {csv_output_file}")
print(f"Sample outputs are saved in: {samples_dir}")
print(f"Cleansed outputs are saved in: {cleansed_output_path}")
