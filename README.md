# 안녕하세요, '원리를 파고들어 실전에 적용하는' AI 엔지니어 김도윤 입니다. 👋

![Profile Views](https://komarev.com/ghpvc/?username=AD-Styles&color=blue)

저는 단순히 만들어진 라이브러리를 사용하는 것을 넘어, **"모델 내부의 핵심 로직과 수학적 원리를 이해해야만 실무의 엣지 케이스(Edge Case)를 해결할 수 있다"** 고 믿습니다. 밑바닥부터 직접 코딩하며 다진 '기초 체력'을 바탕으로, 실제 도로 위 위험 요소를 탐지하는 고도화된 SOTA 모델 최적화까지 아우르는 파이프라인을 구축하고 있습니다.

### 🛠️ Tech Stack & Skills
- **LLM & NLP:** `LangChain`, `Gemini 2.0 Flash API`, `Hugging Face Transformers`, `BERT`
- **Computer Vision:** `PyTorch`, `Torchvision`, `YOLOv5`, `GANs`
- **Data Engineering & Acceleration:** `NVIDIA RAPIDS (cuDF)`, `Dask`, `CUDA (Numba)`
- **MLOps & Deployment:** `NVIDIA Triton Inference Server`, `TensorRT`, `Hugging Face Spaces`

---

**1️⃣ Natural Language Processing & LLM: 지능형 시스템 설계 및 최적화**
* **[`nlp-portfolio (Conversational RAG Chatbot)`](https://github.com/AD-Styles/nlp-portfolio)**: **LangChain**과 **Gemini 2.0 Flash**를 활용하여 복잡한 PDF 문서를 이해하고 답변하는 RAG 시스템을 구축했습니다.
  - **Data Pipeline:** PyPDFLoader를 활용한 문서 추출 및 Recursive Character Text Splitter 기반의 청크(Chunk) 최적화.
  - **Vector Storage:** 생성된 임베딩 벡터를 ChromaDB(또는 FAISS)에 저장하여 시맨틱 검색(Semantic Search) 효율 극대화.
  - **Deployment:** Hugging Face Spaces를 통한 모델 배포 및 Streamlit 기반 웹 인터페이스 구현.
* **[`nlp-triton-deployment (LLM Serving)`](https://github.com/AD-Styles/nlp-triton-deployment)**: 대규모 언어 모델의 실무 서빙을 위해 **NVIDIA Triton Inference Server**를 도입했습니다.
  - **Dynamic Axis:** 모델 설정 파일(`config.pbtxt`)에서 `dims: [-1]`을 명시하여 다양한 길이의 시퀀스 입력을 유연하게 처리하는 동적 축 지원 환경 구축.
  - **Optimization:** Dynamic Batching 기능 활성화 및 TensorRT 엔진 변환을 통해 모델의 추론 지연 시간(Latency) 최소화 및 처리량(Throughput) 극대화.

**2️⃣ Computer Vision: 기초 원리부터 실전 데이터 처리까지**
* **[`object-detection-fundamentals`](https://github.com/AD-Styles/object-detection-fundamentals)**: 블랙박스 라이브러리 없이 IoU, NMS 및 멀티태스크 신경망의 핵심 로직을 픽셀 좌표계 단위부터 직접 구현하며 모델의 기하학적 동작 원리를 체득했습니다.
* **[`yolov5-pothole-detector`](https://github.com/AD-Styles/yolov5-pothole-detector)**: YOLOv5를 활용하여 실시간 도로 영상에서 포트홀을 탐지하는 파이프라인을 구축했습니다. 단일 클래스(`nc=1`) 커스텀 학습으로 mAP50 0.85 이상의 탐지 성능을 확보했습니다.

**3️⃣ Reinforcement Learning: 자율 에이전트 설계**
* **[`carracing-rl`](https://github.com/AD-Styles/carracing-rl)**: `CarRacing-v2` 시뮬레이션 환경에서 CNN 기반의 특징 추출, Frame Stacking, Experience Replay 기법을 결합하여 스스로 최적의 주행 경로를 학습하는 강화학습 에이전트를 개발했습니다.

---

## 🔭 Currently Working On (현재 진행 중인 고민과 도전)
단순한 모델링을 넘어, 데이터 처리 가속화와 LLM 생태계의 효율적 운영을 연구하고 있습니다.

* **NVIDIA LLM Ecosystem 고도화**: **NVIDIA NeMo**를 활용한 모델 미세 조정과 **NVIDIA NIM(Inference Microservices)** 기반의 클라우드 네이티브 추론 환경 구축을 깊이 있게 탐구하며 **NCA-GENL(Generative AI LLMs)** 자격증 취득을 준비 중입니다.
* **GPU 기반 데이터 파이프라인 가속**: 대규모 데이터 전처리 병목을 해소하기 위해 `Dask` 분산 처리와 `NVIDIA RAPIDS (cuDF)`를 활용한 GPU 메모리 기반 데이터프레임 연산을 최적화하고 있습니다.
* **AI UX/UI 통합 설계**: 추론 결과를 사용자에게 가장 효율적으로 전달하기 위해 'Modern Wood' 스타일의 웹 챗봇 인터페이스를 기획 및 구현(`HTML/CSS/JS`)하며 풀스택 관점의 시야를 넓히고 있습니다.

---

## 📬 Contact
- **Email:** `910ehdbs@gmail.com`
- **LinkedIn:** [linkedin.com/in/AD-Styles](https://linkedin.com)
