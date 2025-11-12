# 🧾 Document Type Classification (CV Project)

## 👥 Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [김예인](https://github.com/yeondu-0) | [김재록](https://github.com/Raprok612) | [이재윤](https://github.com/LEEJY0126) | [정용재](https://github.com/YongJae-AI) | [허예경](https://github.com/yekyung821) |
| 팀장 - 모델링 | 모델링 | 모델링 | 모델링 | 모델링 |

---

## 0. Overview

문서 이미지(등록증, 계기판, 보험서류 등)를 입력받아  
17개 클래스로 분류하는 **Computer Vision Classification** 프로젝트입니다.  
문서마다 해상도·밝기·형태가 다양하고, 일부 클래스는 샘플 수가 적은 불균형 데이터셋이었습니다.

### 🧩 Environment
- Python 3.10  
- timm 0.9.12  
- Albumentations 1.4.2  
- WandB for experiment tracking  

### 📦 Requirements
```bash
pip install -r requirements.txt

albumentations==1.3.1
ipykernel==6.27.1
ipython==8.15.0
ipywidgets==8.1.1
jupyter==1.0.0
matplotlib-inline==0.1.6
numpy==1.26.0
pandas==2.1.4
Pillow==9.4.0
timm==0.9.12

````

---

## 1. Competition Info

### Overview

* 주최: Fastcampus AI Bootcamp
* 주제: 문서 이미지 자동 분류
* 평가 지표: **Macro F1-score**

### Timeline

* 2025.10.31 : 데이터 및 베이스라인 코드 공개
* 2025.11.12 : 최종 제출

---

## 2. Components

### Directory

```
cv_project/
├── data/
│   ├── train/               # 원본 학습 이미지
│   ├── train_aug/           # 증강 이미지
│   ├── test/                # 테스트 이미지
│   ├── train_aug.csv
│   └── sample_submission.csv
├── source/
│   ├── train.ipynb          # 학습 코드 (K-Fold + AMP)
│   ├── inference_tta.ipynb  # 추론 + TTA
│   ├── data.py              # 데이터셋 & 증강 정의
│   ├── utils.py             # 학습 루프 함수
├── checkpoints/             # fold별 best model 저장
├── output/                  # 예측 결과
└── README.md
```

---

## 3. Data Description

### Dataset Overview

* 총 이미지: 1,570장 (train), 3,140장 (test)
* 클래스: 17종
* 문서·증명서·차량 사진 등 다양한 형식 존재

### 🔍 EDA Summary

| 항목        | 관찰 내용                                    | 해석                           |
| --------- | ---------------------------------------- | ---------------------------- |
| 밝기 분포     | 왼쪽 꼬리형 분포 — 대부분 밝은 문서, 일부 어두운 이미지 존재     | 조도 편차로 인해 모델이 문서/계기판 간 혼동 가능 |
| 이미지 크기    | width 400 ~ 700, height 350 ~ 650 두 개의 중심 분포 | 문서류(정형)와 사진류(비정형) 혼합         |
| 클래스별 개수   | 1, 13, 14번은 50~80장, 나머지 100장             | 불균형 존재 → class weight 보정     |
| 비율(세로:가로) | 문서류는 세로형, 차량류는 가로형                       | 구조적 특징 차이로 도메인 분리됨           |

### 📊 EDA Visualization

<p align="center">
  <img src="./assets/width_height_distribution.png" width="70%">
</p>

### 클래스 분포 확인

<img width="842" height="470" alt="image" src="https://github.com/user-attachments/assets/47a1a188-04ca-40df-afc4-2b4c20c0c3d1" />


- 1, 13, 14 클래스 수가 적음 → 클래스 불균형
    - **1, 13**: ~50장 (심각한 소수)
    - **14**: ~80장
    - 나머지: ~100장
    
    **해석**
    
    - **강한 클래스 불균형** → macro-F1에서 특히 치명적(소수 클래스 recall↓).
    
    **적용**
    
    - 학습 손실: **class weight + label smoothing(0.05~0.1)**
    - 오프라인 증강: 소수 클래스에만 **강한 aug 비율**(rotation은 텍스트 훼손 주의, 대신 **ShiftScaleRotate의 rotate_limit 낮게(≤15°)**)
    - 검증: **Stratified K-Fold**로 각 fold의 클래스 분포 일치

### 클래스별 이미지 크기: 박스플롯

<img width="850" height="474" alt="image" src="https://github.com/user-attachments/assets/1c1ff01f-7b11-4fa7-875a-5127d4c27187" />

- **0/5/6/8/9/11**: 박스플롯 존재 → **크기 분산 큼**(여러 해상도 혼재)
    - **2번**: **가로 600px로 고정**
    - **그 외 다수**: **약 440px로 고정**
    
    **해석**
    
    - 데이터가 **해상도 이질성**을 강하게 가짐.
    - 작은 글자/도장/마이크로 텍스트가 포함되는 클래스(분산 큰 그룹)는 **고해상도 입력에서 유리**.
    
    **적용**
    
    - Albumentations: `LongestMaxSize` + `PadIfNeeded`로 **종횡비 유지 + 정사각 패딩**
    - 모델별로 `img_size`를 스위치

### 밝기 분포: 히스토그램

<img width="698" height="393" alt="image" src="https://github.com/user-attachments/assets/71081a19-85e8-47d3-b7a3-1ec961eac406" />

- 왼쪽 꼬리 길게 나타남
    
    **해석**
    
    - 전체는 ‘밝은 문서류’가 다수지만, 소수의 어두운 이미지(계기판/번호판 계열)가 섞여 있어 분포의 왼쪽 꼬리가 길어짐.
    - 한 도메인(밝은 문서)에서 학습된 모델이 **어두운 도메인**에서 성능이 급락할 수 있는 전형적 패턴.
    
    **적용**
    
    - 학습: `RandomBrightnessContrast(±0.2~0.4)` 로 **조도 다양화**, 어두운 샘플 비율을 소폭 up.
    - 과한 색변형/CLAHE는 텍스트를 깨뜨릴 수 있으니 **확률 낮게(≤0.2)**.
    - 모듈성: “어두운 후보”(평균 밝기 임계치 이하)만 **light-equalization**(예: 약한 CLAHE) 적용하는 **조건부 전처리**도 고려. → 적용 안함
---

## 4. Modeling

### Model Description

* **EfficientNet-B5** (timm pretrained)
  : 작은 글자가 많은 문서에 적합한 구조, 456x456 해상도 사용
* **K-Fold (5-Fold Stratified CV)**
  : 클래스 불균형 완화 및 일반화 향상
* **AMP (Automatic Mixed Precision)**
  : GPU 메모리 절약 + 학습 속도 개선
* **TTA (Test Time Augmentation)**
  : flip + brightness 조정 기반 예측 안정화

### Modeling Process

1. 데이터 증강
2. K-Fold Stratified Split
3. Fold별 모델 학습 → best F1 저장
4. Inference 시 fold별 예측 평균 앙상블
5. TTA 적용 후 평균 확률 산출

---

## 5. Result

### Leader Board

<img width="966" height="429" alt="image" src="https://github.com/user-attachments/assets/b40c94ca-16e4-4d9c-a6d6-92d55ea8cbde" />

---

## etc

### 🧭 Meeting Log

* [Notion Link](https://www.notion.so/29940cb3731d8146b5bada104fb46a83?v=29940cb3731d81a9886e000c8ca742e8&source=copy_link)
* 회의 내용: 데이터 분포 분석 → 모델 선택 → 실험 관리(W&B)

### 📚 Reference

* [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
* [Albumentations Docs](https://albumentations.ai/docs/)
* [EfficientNet Paper (ICML 2019)](https://arxiv.org/abs/1905.11946)

---

🧩 **Summary**

> 본 프로젝트는 단순한 이미지 분류를 넘어,
> 실제 문서 환경(밝기·방향·형태 다양성)을 고려한
> **현실 적응형 Document Classification System**을 구축했다.

---
