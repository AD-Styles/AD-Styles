# 안녕하세요, '원리를 파고들어 실전에 적용하는' AI 엔지니어 AD입니다. 👋

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAD-Styles&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

저는 단순히 만들어진 라이브러리를 사용하는 것을 넘어, **"모델 내부의 핵심 로직과 수학적 원리를 이해해야만 실무의 엣지 케이스(Edge Case)를 해결할 수 있다"**고 믿습니다. 밑바닥부터 직접 코딩하며 다진 '기초 체력'을 바탕으로, 실제 도로 위 위험 요소를 탐지하는 고도화된 SOTA 모델 최적화까지 아우르는 파이프라인을 구축하고 있습니다.

### 🛠️ Tech Stack
- **AI / Deep Learning:** `PyTorch`, `Torchvision`, `YOLOv5`, `LangChain`
- **Machine Learning & Data:** `NumPy`, `Pandas`, `Matplotlib`, `Scikit-learn`
- **Deployment & System (Focus):** `NVIDIA Triton Inference Server`, `TensorRT`
- **Language & Environment:** `Python 3.10+`, `Git`, `Linux`

---

## 🚀 Featured Projects (포트폴리오 핵심 트랙)
저의 컴퓨터 비전(Vision) 학습 여정은 '기초 원리 파악'에서 시작하여 '실전 최적화'로 이어집니다.

### 1️⃣ [Object Detection from Scratch](https://github.com/AD-Styles/object-detection-fundamentals) 
> **"라이브러리 없이 직접 짜보는 객체 탐지의 핵심 로직"**
- **역할:** 개인 프로젝트 (100%)
- **핵심 내용:** - 픽셀 좌표계를 활용한 `IoU` 연산 및 `NMS` 후처리 로직 직접 구현.
  - ResNet18 Backbone 기반의 Classification/Regression 동시 예측 멀티태스크 신경망 설계.
  - 분모 0 에러 방지(Epsilon) 및 입력 차원 방어 코드를 적용한 실무적 엔지니어링 경험.
- **배운 점:** "컴퓨터 비전 모델이 픽셀 데이터를 어떻게 해석하고 오차를 줄여나가는가"에 대한 근본적 통찰 확보.

### 2️⃣ [YOLOv5 Real-world Pothole Detector](https://github.com/AD-Styles/yolov5-pothole-detector) 
> **"기초 원리를 토대로 구축한 실전 도로 파손(포트홀) 탐지 시스템"**
- **역할:** 개인 프로젝트 (100%)
- **핵심 내용:**
  - Roboflow 데이터셋 기반 YOLOv5s 단일 클래스(`nc=1`) 커스텀 학습 및 평가 파이프라인 구축.
  - 과적합 없는 안정적 Loss 수렴 및 mAP50 0.85 이상의 탐지 성능 확보.
  - MP4 주행 영상 대상 프레임 단위 실시간 추론(Inference) 테스트 완료.
- **배운 점:** 데이터의 물리적 특성(경계가 모호한 포트홀)이 평가지표에 미치는 영향을 분석하고, 모델의 목적성에 맞는 최적화 수행.

---

## 🔭 Currently Working On (현재 진행 중인 고민과 도전)
현재는 모델 개발을 넘어, 더 넓은 AI 생태계와 시스템 효율화를 공부하고 있습니다.

* **NVIDIA Ecosystem & Deployment:** 모델의 실시간 처리량(Throughput)과 지연 시간(Latency) 최적화를 위해 **NVIDIA Triton Inference Server** 및 동적 배치(Dynamic Batching) 기술을 깊이 있게 연구 중입니다. (NCA-GENL 자격 준비 중)
* **Reinforcement Learning (강화학습):** CNN, Frame Stacking, Experience Replay 기법을 결합하여 `CarRacing-v2` 환경에서 스스로 주행하는 에이전트를 개발하고 있습니다.
* **LLM & RAG Application:** LangChain과 Gemini API를 활용한 맞춤형 문서 기반 RAG 챗봇을 설계 및 배포하며 자연어 처리(NLP) 파이프라인으로 시야를 확장하고 있습니다.

---

## 📬 Contact
- **Email:** `910ehdbs@gmail.com`
- **LinkedIn:** [linkedin.com/in/AD-Styles](https://linkedin.com)
