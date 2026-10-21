# 사용한 특징 데이터셋

`dataset.npz`는 원본 CSV 64개, 시행 1,920회에서 만든 25차원 시간영역 특징이다. 원자료는 [sEMG Hand Gesture and Elbow Angle Dataset](https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset)의 `data/` 파일이다.

- `Xtr_raw`, `Xte_raw`: 모델 Pipeline에 넣는 표준화 전 특징
- `ytr`, `yte`: 0~7 클래스 ID
- `groups_tr`, `groups_te`: 창이 속한 시행 ID
- `subjects_tr`, `subjects_te`: 피험자 a~h
- `class_names`, `feature_names`: 실제 라벨·특징 열 순서
- `Xtr`, `Xte`: 지난주 저장한 전체 학습셋 통계 기반 표준화 결과. 이번 교차검증에서는 사용하지 않는다.
- `scaler_mean`, `scaler_scale`, `scaler_var`: 지난주 저장한 표준화 통계

`results/trial_split.csv`는 학습·테스트 시행 목록, `summary.json`은 원본 파일 SHA-256과 전처리 요약, `feature_columns.csv`는 25개 열의 순서다.

원본부터 다시 계산하려면 CSV 64개를 `data/raw/`에 넣고, 필요한 패키지가 설치된 가상환경을 활성화한 뒤 저장소 루트에서 다음을 실행한다.

```bash
python3 preprocessing/build_dataset.py
python3 main.py
```

다른 CSV 경로는 `SEMG_RAW_DIR`, 다른 특징 저장 폴더는 `SEMG_OUTPUT_DIR` 환경변수로 지정할 수 있다. 두 변수는 이번 실행에만 적용하며 시스템 설정을 바꾸지 않는다.
