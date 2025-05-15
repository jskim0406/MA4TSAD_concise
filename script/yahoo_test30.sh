#!/bin/bash
# yahoo_test30.sh - 기존 파일 사용 버전

TRANS_BETA=0
TRANS_ALPHA=95
TEST_RATIO=0.5
WINDOW_SIZE=400
RETRIEVE_POSITIVE_NUM=2
RETRIEVE_NEGATIVE_NUM=1
RETRIEVE_DATABASE_RATIO=1
DELETE_ZERO=1

MODEL_ENGINE="gemini-2.5-flash-preview-04-17" 

DATA_ROOT_DIR="data/Yahoo/ydata-labeled-time-series-anomalies-v1_0"
# 결과 저장 디렉토리 - 평가 스크립트와 일치시킴
SAVE_DIR="result_yahoo_gemini_A1_test30"
PROMPT_MODE=3
SUB_COMPANY="real"

INFER_PATH="${DATA_ROOT_DIR}/A1Benchmark"
RETRIEVE_PATH="${DATA_ROOT_DIR}/A1Benchmark"

RUN_NAME="Yahoo_30files_prompt_${PROMPT_MODE}_win_${WINDOW_SIZE}_${MODEL_ENGINE}"

# 이미 존재하는 파일 사용
echo "실행: Yahoo A1 벤치마크 데이터 30개 파일 테스트"
python run_test_yahoo1_30.py \
  --trans_beta $TRANS_BETA \
  --trans_alpha $TRANS_ALPHA \
  --test_ratio $TEST_RATIO \
  --window_size $WINDOW_SIZE \
  --retrieve_positive_num $RETRIEVE_POSITIVE_NUM \
  --retrieve_database_ratio $RETRIEVE_DATABASE_RATIO \
  --retrieve_negative_num $RETRIEVE_NEGATIVE_NUM \
  --prompt_mode $PROMPT_MODE \
  --run_name $RUN_NAME \
  --infer_data_path "${INFER_PATH}" \
  --retreive_data_path "${RETRIEVE_PATH}" \
  --result_save_dir $SAVE_DIR \
  --sub_company $SUB_COMPANY \
  --delete_zero $DELETE_ZERO \
  --model_engine $MODEL_ENGINE

# 결과 디렉토리 경로 조합
RESULT_PATH="${SAVE_DIR}/${RUN_NAME}"
echo "결과 경로: ${RESULT_PATH}"

# 평가 실행 - 같은 경로 사용
echo "평가 스크립트 실행"
python Eval/Eval_yahoo_test30.py --path "${RESULT_PATH}"

echo "작업 완료"