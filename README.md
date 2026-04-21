# 안녕하세요, '원리를 파고들어 실전에 적용하는' AI 엔지니어 AD입니다. 👋

![Profile Views](https://komarev.com/ghpvc/?username=AD-Styles&color=blue)

저는 단순히 만들어진 라이브러리를 사용하는 것을 넘어, **"모델 내부의 핵심 로직과 수학적 원리를 이해해야만 실무의 엣지 케이스(Edge Case)를 해결할 수 있다"**고 믿습니다. 밑바닥부터 직접 코딩하며 다진 '기초 체력'을 바탕으로, 실제 도로 위 위험 요소를 탐지하는 고도화된 SOTA 모델 최적화까지 아우르는 파이프라인을 구축하고 있습니다.

### 🛠️ Tech Stack
- **AI / Deep Learning:** `PyTorch`, `Torchvision`, `YOLOv5`, `LangChain`
- **Machine Learning & Data:** `NumPy`, `Pandas`, `Matplotlib`, `Scikit-learn`
- **Deployment & System (Focus):** `NVIDIA Triton Inference Server`, `TensorRT`
- **Language & Environment:** `Python 3.10+`, `Git`, `Linux`

---

## 🚀 Featured Projects (포트폴리오 핵심 트랙)
저의 AI 개발 여정은 '기초 원리 탐구'에서 시작하여 '실무형 SOTA 모델 최적화', 그리고 '실제 서비스 환경 배포'로 확장되고 있습니다.

**1️⃣ Computer Vision: 기초 원리부터 실전 응용까지**
* **[`object-detection-fundamentals`](https://github.com/AD-Styles/object-detection-fundamentals)**: 블랙박스 라이브러리에 의존하지 않고 IoU, NMS 및 멀티태스크 신경망(ResNet18 기반)의 핵심 로직을 픽셀 좌표계 단위부터 직접 구현하며 모델의 기하학적 동작 원리를 체득했습니다.
* **[`yolov5-pothole-detector`](https://github.com/AD-Styles/yolov5-pothole-detector)**: 기초 원리를 토대로 실제 도로 영상 속 포트홀을 탐지하는 파이프라인을 구축했습니다. 단일 클래스(`nc=1`) 커스텀 학습을 통해 과적합 없는 수렴과 mAP50 0.85 이상의 실시간 탐지 성능을 확보했습니다.
* **[`pytorch-image-classification`](https://github.com/AD-Styles/pytorch-image-classification)**: PyTorch 기반의 체계적인 이미지 분류 모델 학습, 평가, 검증 파이프라인을 설계했습니다.

**2️⃣ Natural Language Processing & LLM: 지능형 시스템 설계 및 최적화**
* **[`nlp-portfolio (Conversational RAG Chatbot)`](https://github.com/AD-Styles/nlp-portfolio)**: **LangChain**과 **Gemini 2.0 Flash**를 활용하여 복잡한 PDF 문서를 이해하고 답변하는 RAG 시스템을 구축했습니다.
  - **Core Logic:** Recursive Character Text Splitter를 통한 청크 최적화 및 Vector DB 연동.
  - **Prompt Engineering:** Few-shot 및 Chain-of-Thought 기법을 적용하여 답변의 정밀도와 논리적 일관성 확보.
  - **Deployment:** Hugging Face Spaces를 통한 모델 배포 및 Streamlit 기반 인터페이스 구현.
* **[`nlp-triton-deployment (LLM Serving)`](https://github.com/AD-Styles/nlp-triton-deployment)**: 대규모 언어 모델의 실무 서빙을 위해 **NVIDIA Triton Inference Server**를 도입했습니다.
  - **Optimization:** Dynamic Batching 및 TensorRT 최적화를 통해 높은 처리량(Throughput) 확보.
  - **Efficiency:** 모델의 동적 축(Dynamic Axis) 지원 설정을 통해 다양한 입력 길이에 유연하게 대응하는 파이프라인 구성.

**3️⃣ Reinforcement Learning: 자율 에이전트 설계**
* **[`carracing-rl`](https://github.com/AD-Styles/carracing-rl)**: `CarRacing-v2` 시뮬레이션 환경에서 CNN 기반의 특징 추출, Frame Stacking, 그리고 Experience Replay 기법을 결합하여 스스로 최적의 주행 경로를 학습하는 강화학습 에이전트를 개발했습니다.

---

## 🔭 Currently Working On (현재 진행 중인 고민과 도전)
단순한 AI 모델링을 넘어, 시스템의 성능 최적화와 사용자 경험(UX)을 통합하는 실무형 엔지니어링에 집중하고 있습니다.

* **NVIDIA LLM Ecosystem 고도화**: **NVIDIA NeMo**를 활용한 모델 미세 조정(Fine-tuning)과 **NVIDIA NIM(Inference Microservices)** 기반의 클라우드 네이티브 추론 환경 구축을 깊이 있게 탐구하고 있습니다. (NCA-GENL 자격 취득 준비 중)
* **Advanced RAG 및 에이전틱 워크플로우**: 단순 검색을 넘어, 도구 사용(Tool-use)과 자기 비판(Self-criticism) 로직이 포함된 에이전트 기반의 고도화된 NLP 시스템 설계를 목표로 합니다.
* **AI UX/UI 통합 설계**: 추론 결과를 사용자에게 가장 효율적으로 전달하기 위해 Modern Wood 스타일의 인터페이스를 직접 기획 및 구현하며, 기술과 사용자를 잇는 엔지니어링을 지향합니다.

---

## 📬 Contact
- **Email:** `910ehdbs@gmail.com`
- **LinkedIn:** [linkedin.com/in/AD-Styles](https://linkedin.com)
