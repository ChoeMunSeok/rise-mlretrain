# 회의 기준 v8 스키마

앱 13컬럼, PAPE CSV 14컬럼, 전체 혼동행렬 및 운영 입력 확장은 이전 버전과 동일하다.
프로토콜은 apigraph_meeting_validation2015_v8이다.
기본 분할은 2012 Train / 2013-2014 Source / 2015 Validation / 2016-2018 Test 보류다.
이번 수행 범위는 2015 Validation뿐이다. Test는 통합 담당자의 후속 범위다.

source_context는 source_version=validation_2013_2014, source_period=[2013-01,2014-12], source_available_from_month=2015-01이다.
회사의 연간 Source 갱신 원칙과 라벨 공개 선행 조건은 source_schedule.py에 명시했다.
Source 갱신은 XGBoost RETRAIN과 구별한다. 현재 실험에서는 초기 Source만 사용한다.

app_features.parquet는 sample_id,origin_file,original_row,feature_group,observed_month,project_role,xgb_score,
final_prediction,decision_threshold,model_hash,threshold_policy_hash,schema_version,feature_schema_hash 13컬럼이다.
binary_features.parquet는 sample_id+binary_0000..binary_1158, labels_for_evaluation.parquet는 sample_id,y_true다.
Target y_true는 추정 뒤 평가에만 결합한다. 병렬 추정 worker는 Target 라벨 파일을 읽지 않는다.

pape_estimates.csv의 14컬럼 순서:
target_chunk_id,input_variant,estimated_fn,estimated_fp,estimated_recall,estimated_f1,estimated_loss,
actual_fn,actual_fp,actual_recall,actual_f1,dre_auc,ess_ratio,max_weight.
input_variant는 all_binary__uniform/all_binary__dre. seed는 summary의 csv_row_index로 연결한다.
confusion_matrix.csv는 [[TN,FP],[FN,TP]], 행=true/열=predicted, label_order=[0,1]이다.
gain_inputs.parquet/rccda_inputs.json에는 추정 특징만 포함하고 실제값·정답·평가 오차를 넣지 않는다.

병렬 추정은 최대 4개 worker로 제한하고 요청된 월·seed·방법 순서대로 저장한다.
execution.workers를 바꾸면 설정 hash가 달라지므로 기존 export/모델과 혼합하지 않는다.
부분 Validation으로 완성 인계 ZIP을 생성할 수 없다.
