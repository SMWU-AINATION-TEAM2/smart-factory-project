# 🏭 Smart Factory Energy Prediction

> **스마트 팩토리 에너지 사용량 예측 프로젝트** — 시계열 분석 모델을 활용한 에너지 소비 예측 및 AWS 배포

---

## 📌 프로젝트 개요

스마트 팩토리 환경에서 수집된 시계열 데이터를 기반으로 에너지 사용량을 예측하는 머신러닝 모델을 개발하고, AWS SageMaker 엔드포인트로 배포한 프로젝트입니다.

---

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python |
| Modeling | ARIMA, LSTM |
| Data Processing | Pandas, NumPy |
| Deployment | AWS SageMaker |

---

## 📁 프로젝트 구조

```
smart-factory-project/
├── Scripts/          # 모델 학습 및 실행 스크립트
├── utils/            # 데이터 전처리 유틸리티
├── requirements.txt  # 의존성 패키지
└── README.md
```

---

## ⚙️ 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

- [모듈별 데이터셋](https://drive.google.com/drive/folders/1_DV-VvgT8bbkhdXmRaSQQ7elCIN4G_AN?usp=sharing)
- [모듈별 하프 데이터셋](https://drive.google.com/drive/folders/1hbGdExUh9pZpo6uYQ0psNWzqKo1jfKJW?usp=sharing) — 데이터셋을 홀수/짝수로 분리, localtime 컬럼을 date로 수정한 버전

### 3. 모델 실행

```bash
python Scripts/main.py
```

---

## 🔍 모델 설명

- **ARIMA** — 선형적 시계열 패턴 예측
- **LSTM** — 비선형적 장기 의존성 학습

두 모델의 예측 결과를 비교하여 스마트 팩토리 에너지 소비 패턴을 분석합니다.

---

## ☁️ AWS 배포

학습된 모델은 AWS SageMaker 엔드포인트로 배포하여 실시간 예측이 가능하도록 구성했습니다.
