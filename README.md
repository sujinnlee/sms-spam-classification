<div align="center">

# 스팸 문자 분류 (SMS Spam Classification)

**TF-IDF와 머신러닝 세 가지 모델로 스팸 문자를 분류하고, 오분류 사례까지 비교한 프로젝트**

2026 명지대학교 MCC 코딩동아리 개인 프로젝트

[📓 노트북](spam_classification.ipynb) | [📊 혼동 행렬](confusion_matrix.png)

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)&nbsp;![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)&nbsp;![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)&nbsp;![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)&nbsp;![TF-IDF](https://img.shields.io/badge/TF--IDF-7952B3?style=for-the-badge)

</div>

<br>

## 🎯 한눈에 보기

| | |
|---|---|
| **문제** | 스팸이 13.4%뿐인 불균형 데이터라, 전부 정상이라 답해도 정확도가 약 86.6%가 나온다 |
| **내가 한 일** | 전처리, TF-IDF 변환, 모델 3종 비교, 놓친 스팸(FN)과 정상 오분류(FP) 분석 (개인 프로젝트) |
| **핵심 결과** | Linear SVM 선택, F1-score **0.936**, 놓친 스팸 **33 → 17개**, 정상 오분류 1개 |
| **선택 기준** | 정확도가 아니라 정상 문자를 스팸으로 차단하는 오류를 줄이는 쪽을 우선했다 |
| **AI 활용** | 2026.10에 Claude로 코드를 재구현하고 실행 결과를 검증했다 |

<br>

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 기간 | 2026.03 ~ 2026.06 (최초 수행), 2026.10 (재구현과 검증) |
| 형태 | MCC 코딩동아리 개인 프로젝트 |
| 데이터 | SMS Spam Collection, 5,572개 |
| 최종 모델 | Linear SVM (F1-score 0.936) |

문자 메시지를 스팸(Spam)과 정상(Ham)으로 분류하는 프로젝트이다. 문자 데이터를 전처리하고 TF-IDF로 수치화한 뒤, 세 가지 모델의 성능을 비교했다.

전체 데이터의 스팸 비율이 13.4%로 클래스 불균형이 있어서, 정확도만 보지 않고 Precision, Recall, F1-score와 실제 오분류 사례를 함께 분석했다.

> **AI 활용 안내**: 2026년 3~6월에 진행한 개인 프로젝트를 바탕으로, 2026년 10월에 Claude를 활용해 코드를 재구현하고 실행 결과를 검증했다.

<br>

## 🏗 처리 과정

```text
SMS 데이터 → 전처리 → 학습·테스트 분할 → TF-IDF 변환 → 모델 학습 → 성능 평가 → 오분류 분석 → 모델 선택
                                                      (Naive Bayes / Logistic Regression / Linear SVM)
```

| 구분 | 내용 |
|---|---|
| 전체 데이터 | 5,572개 (정상 약 86.6%, 스팸 약 13.4%) |
| 학습 데이터 | 4,457개 (80%) |
| 테스트 데이터 | 1,115개 (20%), 클래스 비율을 유지한 Stratified Split |
| 전처리 | 소문자 변환, 영문과 숫자 외 문자 정제, 영어 불용어 제거 |
| 평가 방법 | 학습 데이터 5회 교차검증, 별도 테스트 데이터로 최종 평가 |

### 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 데이터 처리 | Pandas |
| 특징 추출과 모델 | scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, Linear SVM) |
| 시각화 | Matplotlib |

### 데이터 누출 방지

TF-IDF 변환기는 학습 데이터에만 적합(`fit`)하고, 테스트 데이터에는 변환(`transform`)만 적용해서 테스트 정보가 학습에 들어가지 않게 했다.

```python
vec = TfidfVectorizer(stop_words="english")

X_train_v = vec.fit_transform(X_train)
X_test_v = vec.transform(X_test)
```

<br>

## 📊 결과

동일한 테스트 데이터 1,115개 기준이다.

| 모델 | F1-score | 놓친 스팸 (FN) | 정상 메시지 오분류 (FP) |
|---|---:|---:|---:|
| Naive Bayes | 0.875 | 33개 | 0개 |
| Logistic Regression | 0.863 | 36개 | 0개 |
| **Linear SVM (선택)** | **0.936** | **17개** | 1개 |
| 가중치 적용 Logistic Regression | | 13개 | 13개 |

Linear SVM은 세 가지 기본 모델 중 F1-score가 가장 높았고, Naive Bayes보다 놓친 스팸이 16개 적었다. 정상 메시지를 스팸으로 잘못 분류한 것은 1개뿐이어서, 스팸 탐지와 정상 메시지 보호 사이의 균형을 고려해 최종 모델로 골랐다.

### 실제 오분류 사례

| 종류 | 메시지 |
|---|---|
| 스팸을 정상으로 분류 (FN) | `ringtoneking 84484` |
| 정상을 스팸으로 분류 (FP) | `K..u also dont msg or reply to his msg..` |

<br>

## 🔥 트러블슈팅

<details open>
<summary><b>1. 정확도만으로는 스팸 탐지 성능을 알 수 없던 문제</b></summary>

- **문제**: 정상 메시지가 약 86.6%라서, 전부 정상이라고 예측해도 정확도가 약 86.6%가 나온다. 정확도만으로는 실제 스팸 탐지 성능을 평가하기 어려웠다.
- **해결**: Precision, Recall, F1-score를 함께 비교하고, 스팸을 놓친 경우(FN)와 정상을 차단한 경우(FP)를 따로 분석했다. 스팸에 가중치를 준 Logistic Regression도 추가로 비교했다.
- **결과**: Linear SVM은 Naive Bayes보다 놓친 스팸을 33개에서 17개로 줄였다. 가중치 Logistic Regression은 놓친 스팸이 13개로 더 적었지만 정상 메시지 13개를 스팸으로 잘못 분류해서, 정상 메시지 차단을 최소화할 수 있는 Linear SVM을 골랐다.

</details>

<br>

## 🌱 개선 방향

- 문자 단위 n-gram 특징 적용
- 전화번호와 특수 패턴 정규화 검토
- 오분류 메시지 유형별 성능 분석 확대

<br>

## 📂 폴더 구조

| 파일 | 내용 |
|---|---|
| `spam_classification.ipynb` | 데이터 불러오기부터 모델 비교, 오분류 분석까지의 전체 코드 |
| `confusion_matrix.png` | 혼동 행렬 |

<br>

## 🚀 실행 방법

노트북 맨 위의 Colab 버튼으로 열고, 위에서부터 한 칸씩 `Shift + Enter`로 실행한다. 데이터는 노트북이 온라인에서 직접 불러온다.
