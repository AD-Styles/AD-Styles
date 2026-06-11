# 안녕하세요. <br/>원리를 파고들어 실전에 적용하는 'AI 테크 인재' 김도윤 입니다! 👋...........

![Profile Views](https://komarev.com/ghpvc/?username=AD-Styles&color=blue)

저는 단순히 만들어진 라이브러리를 사용하는 것을 넘어, **"모델 내부의 핵심 로직과 수학적 원리를 이해해야만 실무의 엣지 케이스(Edge Case)를 해결할 수 있다"** 고 믿습니다. 논문을 직접 구현하며 다진 '기초 체력' 위에, **밑바닥(From Scratch) → 전이학습/파인튜닝 → SOTA 모델 최적화 및 서빙(Triton/TensorRT)** 까지 이어지는 엔드투엔드 AI 파이프라인을 설계하고 있습니다.

> **핵심 3대 키워드** — From Scratch (원리 검증) · Production & Edge (Triton · RAPIDS · Jetson) · Multi-Modal (Vision · Language · Voice · Sensor)

---

### 🛠️ Tech Stack & Skills

- **Foundations (논문 직접 구현):** `PyTorch`, `NumPy` only — Transformer, ResNet, GPT, VAE, GAN, Diffusion, CLIP, **Mini-LLaVA (VLM)**
- **LLM & NLP:** `LangChain`, `LangGraph`, `Gemini 2.0 Flash`, `OpenAI API`, `Qwen2.5`, `EXAONE-3.5`, `Hugging Face Transformers`, `KoGPT2`, `KLUE-BERT`, `Sentence-Transformers`
- **LLM Fine-Tuning & Optimization:** `Unsloth`, `QLoRA (4-bit NF4)`, `GGUF`, `PEFT`, `Pydantic-Structured Output`
- **Computer Vision:** `PyTorch`, `Torchvision`, `YOLOv5`, `YOLOv8`, `U-Net`, `FCN`, `SMP (segmentation_models_pytorch)`, `ResNet`, `GANs`, `OpenCV`
- **RAG & Vector DB:** `ChromaDB`, `FAISS`, `RecursiveCharacterTextSplitter`, `LCEL`, `Multi-Collection Routing`
- **Reinforcement Learning:** `Q-Learning`, `DQN`, `Experience Replay`, `Frame Stacking`, `Gymnasium`
- **Data Engineering & Acceleration:** `NVIDIA RAPIDS (cuDF / cuGraph)`, `Dask`, `CUDA (Numba)`, `Parquet`, `NetworkX`
- **MLOps & Deployment:** `NVIDIA Triton Inference Server`, `TensorRT`, `ONNX`, `llama.cpp`, `Dynamic Batching`, `Hugging Face Spaces`, `Gradio`, `Streamlit`
- **Edge AI & Full-Stack:** `NVIDIA Jetson`, `ROS 2`, `FastAPI`, `Flutter`

---

## 🚀 대표 프로젝트 (Representative Projects)

### 1️⃣ Foundations — 논문 / 핵심 알고리즘 From Scratch

> **"라이브러리 한 줄로 끝나는 코드는 누구나 작성할 수 있다."** — 그래서 저는 모델의 핵심 메커니즘을 PyTorch / NumPy 만으로 직접 구현하며 "왜 이 구조인가"를 설명할 수 있는 깊이를 쌓았습니다.

* **⭐ [`vlm-from-scratch-v4`](https://github.com/AD-Styles/vlm-from-scratch-v4)** — **v3 가 정량 입증한 "vision encoder 크기 ≠ VLM 능력, LLM 이 진짜 병목" 결론을 정면 돌파한 4세대 반복.** v3 가 추론 wrapper 5종으로 *우회* 했던 0.5B LLM 한계를, v4 는 언어 모델 자체를 **Qwen2.5-1.5B-Instruct 로 교체 (0.5B → 1.5B, 3배)** 해 정공법으로 해결. **CLIP-ViT-B/32 (frozen) + 학습형 2-layer MLP projector (768→1536, GELU) + 4-bit NF4 QLoRA** 구조를 double quantization 으로 압축(fp32 6 GB → 0.9 GB)해 **단일 8 GB consumer GPU** 에서 2-stage 학습 (projector alignment loss 5.0→1.98 / instruction tuning loss 3.65→1.01). Qwen2.5 내장 `<|image_pad|>` 토큰을 재사용해 신규 토큰 없이 **v3 의 1 GB adapter bloat 를 원천 차단**, custom `_merge` 함수로 임베딩 splice 직접 구현. **VQAv2 36.7% → 56.8%, POPE 50.0% → 71.8% (yes-F1 0.735 · prediction bias 0.568 · refusal rate 0.000)**, first-token entropy 기반 **OOD 탐지 ROC AUC 0.971** 와 OOD abstention layer 로 환각 경고. 검증 게이트를 배포 *이전* 으로 이동, Stage 2 는 VQAv2·LocalizedNarratives·A-OKVQA·KoLLaVA(한국어) 46K 혼합 데이터.
* **[`vlm-from-scratch-v3`](https://github.com/AD-Styles/vlm-from-scratch-v3)** — **v2 의 3가지 한계 (한국어 catastrophic forgetting · OOD 환각 · 1GB adapter) 를 모두 해결한 차세대 반복.** **추론 wrapper 5종 (CLIP yes/no 게이팅 · CLIP 색상 분류 · 출력 후처리 · m2m100 한↔영 번역 · OOD 감지 layer)** 으로 0.5B LLM 한계를 우회. **POPE 50% → 53.33% (untuned) / 70% (tuned), LoRA adapter 1045 MB → 8.28 MB (−99.21%, greedy 출력 bit-단위 동일)**. ViT-L/14 ablation 실패에서 *"vision encoder 크기 ≠ VLM 능력 — LLM 이 진짜 병목"* 정량 입증. HF Spaces Live Demo + gradio_client API + Playwright Chromium 자동화로 **3중 외부 검증**.
* **[`vlm-from-scratch (Mini-LLaVA v1→v2)`](https://github.com/AD-Styles/vlm-from-scratch)** — v3 의 출발점이 된 baseline. CLIP-from-scratch + GPT-from-scratch 의 빌딩 블록을 실무 스케일로 조립한 멀티모달 LLM. HuggingFace `LlavaForConditionalGeneration` 미사용, **`<image>` 토큰 splice / projector / LoRA adapter 통합** 직접 구현. **v1 (Stage 1 alignment) 한계 정량 분석 → v2 (Stage 2 LoRA + 균형 instruction 데이터)** 로 개선하는 두 차례 반복 사이클 전체 기록. 영문 강아지 VQA 5문항 4/5 정확, 한국어에서 **catastrophic forgetting 정량 입증**, 피카츄(OOD)에서 **체계적 오류 패턴 분석** 까지 모델 한계의 솔직한 해부.
* **⭐ [`transformer-from-scratch`](https://github.com/AD-Styles/transformer-from-scratch)** — 『Attention Is All You Need』 논문 재현. `nn.Transformer` / HuggingFace 미사용, **Scaled Dot-Product Attention · Multi-Head · Sinusoidal Positional Encoding · Encoder-Decoder** 를 텐서 연산만으로 구현. 토이 태스크에서 **Val Acc 98.4%**, Attention Heatmap의 anti-diagonal 패턴으로 학습 원리 검증.
* **[`gpt-from-scratch`](https://github.com/AD-Styles/gpt-from-scratch)** — nanoGPT 영감의 **Decoder-only Transformer (10.79M params)**. Q/K/V 분할까지 손작성, 자체 Char-level Tokenizer (vocab 65), Tiny Shakespeare 학습 후 **Greedy / Temperature / Top-k 샘플링 비교**.
* **[`resnet-from-scratch`](https://github.com/AD-Styles/resnet-from-scratch)** — Skip Connection이 **Degradation Problem을 정말 해결하는가?** 를 검증. 동일 깊이·파라미터의 Plain-20 vs ResNet-20 비교 실험에서 `out + shortcut(x)` 단 한 줄의 효과를 **+2.19%p (89.16% vs 86.97%)** 로 정량화.
* **[`diffusion-models-from-scratch`](https://github.com/AD-Styles/diffusion-models-from-scratch)** — **U-Net + Sinusoidal Time Embedding + Classifier-Free Guidance** 를 한 파일에. Forward/Reverse 확산 과정과 Bernoulli Context Mask까지 직접 구현.
* **[`vae-from-scratch`](https://github.com/AD-Styles/vae-from-scratch)** — **Reparameterization Trick · ELBO (Reconstruction + β·KL)** 직접 구현. 16차원 Latent Space에서 Latent Traversal과 Interpolation으로 **차원별 의미 분석**.
* **[`gan-from-scratch`](https://github.com/AD-Styles/gan-from-scratch)** — **DCGAN (ConvTranspose Generator + Conv Discriminator)** 으로 Minimax 게임 직접 작성. Mode Collapse 검증, D Accuracy 0.5 균형 모니터링, **VAE vs GAN (Blurry vs Sharp)** 동일 조건 비교.
* **[`clip-from-scratch`](https://github.com/AD-Styles/clip-from-scratch)** — 이전 From-Scratch 부품(ResNet-20 + Transformer)을 **직접 조립한 Multi-Modal 시스템**. Symmetric InfoNCE Loss + Learnable Temperature τ로 학습, **학습 양식 64.4% vs Zero-shot 64.6%** — 임베딩 공간 정렬(Alignment) 성공.
* **[`object-detection-fundamentals`](https://github.com/AD-Styles/object-detection-fundamentals)** — **IoU · NMS · 멀티태스크(좌표 회귀 + 분류) Loss 밸런싱** 을 NumPy로 직접 구현. ResNet18 Backbone 위에 픽셀 좌표계부터 손으로.
* **[`image-segmentation-from-scratch`](https://github.com/AD-Styles/image-segmentation-from-scratch)** — **3단계 심화 학습 (FCN8s 논문 재현 → U-Net 직접 설계 → SMP 실무)**. 단순 코드 재현이 아닌 "왜 이 구조인가"를 설명할 수 있는 깊이.

---

### 2️⃣ NLP & LLM — 지능형 시스템 설계 / 파인튜닝 / 서빙

#### LLM 응용 & RAG 시스템
* **[`langchain-rag-enterprise-chatbot`](https://github.com/AD-Styles/langchain-rag-enterprise-chatbot)** — **5종 사내 문서(TXT/JSON/JSONL/CSV/PDF)** 를 멀티 컬렉션 ChromaDB로 분리 저장하여 자연어 라우팅 기반 RAG 챗봇 구축. **JSON은 통째로 보존, 텍스트는 청크 분할** 등 형식별 전략 차별화.
* **⭐ [`langchain-production-chatbot`](https://github.com/AD-Styles/langchain-production-chatbot)** — **LangGraph + Pydantic Structured Output** 으로 분류·메모리·라우팅을 한 호출에. **InMemorySaver thread 격리 + SummarizationMiddleware (4000 토큰 임계치)** 로 토큰 폭주 방지.
* **[`langchain-agent-tool-integration`](https://github.com/AD-Styles/langchain-agent-tool-integration)** — LLM이 외부 시스템(시간 조회 / 웹 스크래핑 / SQL)에 직접 접근하는 **3종 Tool 통합 Agent**. 시스템 프롬프트로 **읽기 전용 SQL 강제** 등 안전 설계.
* **⭐ [`rag-chatbot`](https://github.com/AD-Styles/rag-chatbot)** — LangChain + Chroma + **Gemini 2.0 Flash** 로 대화 메모리 유지 RAG 챗봇 구축, Gradio UI로 Hugging Face Spaces 배포까지 엔드투엔드 완성.
* **[`nlp-portfolio`](https://github.com/AD-Styles/nlp-portfolio)** — **수학적 기초(NumPy Self-Attention) → 분류(BERT/Logistic Regression) → 의미 검색(Multilingual-MiniLM)** 3단계 일관 파이프라인. Error Analysis 중심의 Data-Centric 접근.

#### LLM 파인튜닝 & 효율화
* **[`unsloth-qlora-finetuning`](https://github.com/AD-Styles/unsloth-qlora-finetuning)** — **Llama-3 8B를 T4 16GB 한 장으로** 파인튜닝. 4-bit NF4 양자화 + LoRA로 **전체 파라미터의 0.08% 미만만 학습**, **VRAM 60% 절감 / 훈련 속도 2.6배** 가속.
* **⭐ [`kogpt2-korean-finetuning`](https://github.com/AD-Styles/kogpt2-korean-finetuning)** — KoGPT2를 NSMC 영화 리뷰 도메인으로 파인튜닝. **PreTrainedTokenizerFast 인덱스 밀림 문제를 근본 디버깅**, Top-P · Temperature · Repetition Penalty 디코딩 전략 비교.
* **[`nlp-bert-finetuning`](https://github.com/AD-Styles/nlp-bert-finetuning)** — Self-Attention → BERT → 전이 학습([CLS] Token Pooling)까지 단계별 학습.

#### LLM Serving (Production)
* **[`nlp-triton-deployment`](https://github.com/AD-Styles/nlp-triton-deployment)** — BERT를 **PyTorch → ONNX → NVIDIA Triton Inference Server** 로 배포. `config.pbtxt`의 `dims:[-1]` 동적 축 지원 + **Dynamic Batching + 동적 패딩** 으로 **지연 시간 45ms → 12.5ms (▼72%) / 처리량 22 → 145 TPS (▲6.6배)** 달성.

#### NLP 기초 다지기
* **[`nlp-foundations`](https://github.com/AD-Styles/nlp-foundations)** — Attention 메커니즘을 NumPy로 직접 구현 + KLUE-BERT 파인튜닝 (NSMC **정확도 50% → 89%**) + Masked LM의 사회적 편향성 분석.
* **[`nlp-preprocessing-foundation`](https://github.com/AD-Styles/nlp-preprocessing-foundation)** — 텍스트 정제·정규화·토큰화·TF-IDF·RNN/LSTM 까지 OOP 모듈화.
* **[`nlp-text-classification`](https://github.com/AD-Styles/nlp-text-classification)** — 영화 리뷰(이진) + 국민청원(17개 다중 분류) 파이프라인. 데이터 불균형·반어법 한계 분석.
* **[`nlp-semantic-search`](https://github.com/AD-Styles/nlp-semantic-search)** — Sentence-Transformers + Cosine Similarity 의미 검색 시스템.

---

### 3️⃣ Computer Vision — 기초 원리부터 실시간 탐지까지

* **[`yolov5-pothole-detector`](https://github.com/AD-Styles/yolov5-pothole-detector)** — `object-detection-fundamentals` 의 원리 위에 **YOLOv5s 단일 클래스(`nc=1`) 커스텀 학습** 으로 실제 도로 영상에서 포트홀 탐지. **mAP@0.5 0.85+** 의 실무 서비스 수준 파이프라인 완성.
* **[`multimodal-ai-sensor-fusion`](https://github.com/AD-Styles/multimodal-ai-sensor-fusion)** — RGB + LiDAR 센서 결합. **Early / Late / Intermediate 3가지 융합 아키텍처 비교** + **CLIP 스타일 NT-Xent 대조학습**. 단일 모델 92.7% → **융합 모델 100%** 인식률, 유사도 행렬 95% 감소로 정렬(Alignment) 입증.
* **[`pytorch-image-classification`](https://github.com/AD-Styles/pytorch-image-classification)** — 텐서 조작 → MLP/CNN/VGG 직접 설계 → ResNet 전이학습까지 이미지 분류 전 과정. OOM 문제 해결로 하드웨어 파이프라인 엔지니어링 역량 확보.
* **⭐ [`resnet-transfer-learning-cifar10`](https://github.com/AD-Styles/resnet-transfer-learning-cifar10)** — ResNet50 vs ResNet101 효율성 비교, **Stage A/B 전이학습 전략 분석**, ImageNet Pretrained Weights의 Skip Connection 효과 실험적 검증.

---

### 4️⃣ Reinforcement Learning — 자율 에이전트 설계

* **⭐ [`car-racing-dqn`](https://github.com/AD-Styles/car-racing-dqn)** — `CarRacing-v2` 에서 **96x96 픽셀 → CNN-DQN 직접 학습**. **Frame Stacking (4 frames) · Experience Replay · Target Network** 로 학습 안정화, Hugging Face 배포 완료.
* **[`rl-optimization-benchmark`](https://github.com/AD-Styles/rl-optimization-benchmark)** — 동일한 Q-Learning 알고리즘으로 **Taxi → CliffWalking → Blackjack 도장깨기**. 500개 상태부터 튜플 상태까지 자료구조 확장으로 범용 설계 능력 입증.
* **[`rl-q-learning`](https://github.com/AD-Styles/rl-q-learning)** — FrozenLake에서 **희소 보상의 한계를 극복하는 밀집 보상(Dense Reward) 설계** + Epsilon-Greedy + 에이전트-환경 분리 OOP 아키텍처.

---

### 5️⃣ Data Engineering & Domain Application — GPU 가속 데이터 파이프라인 / 도메인 특화 분석

* **[`rapids-dask-pipeline`](https://github.com/AD-Styles/rapids-dask-pipeline)** — 대규모 데이터 전처리 병목 해소를 위해 **NVIDIA RAPIDS cuDF + Dask** 결합. **CSV → Parquet, CPU → GPU** 로 저장 포맷과 연산 엔진을 함께 최적화, MapReduce DAG 시각화로 Lazy Evaluation 효율성 입증.
* **[`road-network-graph-analytics`](https://github.com/AD-Styles/road-network-graph-analytics)** — **NVIDIA cuGraph** 로 영국 도로망 (1,225 노드 / 2,622 엣지) 그래프 분석. Dijkstra SSSP + 5종 중심성 (Betweenness · Eigenvector · PageRank · Katz · Degree) 비교로 실무 인사이트 도출.
* **[`skhynix-stock-analysis`](https://github.com/AD-Styles/skhynix-stock-analysis)** — 반도체 도메인 지식을 결합한 SK하이닉스 주가 예측. **Bidirectional LSTM + 기술적 지표 (MA/RSI/Bollinger/MACD) + Huber Loss**, 데이터 분포 변화에 대한 모델 한계의 객관적 인정.

---

### 6️⃣ Team Project — 팀 협업으로 완성한 온디바이스 멀티모달 AI 시스템

* **⭐ [`MIND-CARE-Conversational-ChatBot`](https://github.com/AD-Styles/MIND-CARE-Conversational-ChatBot)** — **독거노인 케어용 온디바이스 HRI 시스템 '마음돌봄'.** 클라우드 의존 없이 **NVIDIA Jetson AGX Xavier 한 대** 에서 음성·언어·비전 추론을 전부 로컬 실행하는 프로덕션급 시스템. **ROS 2 기반 4개 느슨한 결합 서브시스템** (음성대화 STT→LLM→TTS · 비전 · 응급 상태머신 · WebSocket API 게이트웨이) 으로 모듈화. LLM 은 **EXAONE-3.5-7.8B-Instruct 를 GGUF 양자화 (Q3_K_M 속도 / Q4_K_M 품질) + llama.cpp 부분 GPU offload**, 의료 지식은 **ChromaDB + 서울아산병원 질환 백과** 기반 RAG 로 사실 근거 제공. 낙상 감지는 **YOLOv8n-pose (TensorRT FP16)** 골격 추출 후 **frame-level + 시간적 확인 2-stage 검증** 으로 앉기·숙이기 오탐을 억제 — **Recall 0.77 / Precision 0.68 (URFDD)**. '경보 전 질문(query-before-alert)' 로직 · 진단/투약 금지 프롬프트 등 **false negative 방지를 우선한 안전 설계**, GPIO·FCM·SMS 다중 채널 보호자 알림과 Flutter 모바일 앱까지 엔드투엔드 완성.

---

## 🔭 Currently Working On (현재 진행 중인 고민과 도전)

단순한 모델링을 넘어, **데이터 처리 가속화 · LLM 생태계의 효율적 운영 · 풀스택 사용자 경험** 을 동시에 연구하고 있습니다.

* **NVIDIA LLM Ecosystem 고도화** — **NeMo** 기반 모델 미세 조정과 **NIM (Inference Microservices)** 기반 클라우드 네이티브 추론 환경 구축을 탐구하며 **NCA-GENL (Generative AI LLMs)** 자격증 취득 준비 중.
* **GPU 기반 데이터 파이프라인 가속** — `Dask` 분산 처리 + `RAPIDS cuDF` GPU 메모리 데이터프레임 연산을 결합하여 대규모 ETL의 병목 해소 연구.
* **Multi-Modal AI 심화** — `clip-from-scratch` → `multimodal-ai-sensor-fusion` → `vlm-from-scratch (v1→v2)` → `vlm-from-scratch-v3` → **`vlm-from-scratch-v4`** 로 이어지는 **이미지 · 텍스트 · 센서 데이터의 공통 임베딩 공간 정렬** 학습 궤적. v4 에서 **언어 모델 업그레이드 (Qwen2.5-1.5B, 0.5B→1.5B) · VQAv2 56.8% / POPE 71.8% · first-token entropy OOD 탐지 (ROC AUC 0.971)** 를 달성하며 v3 가 입증한 'LLM 병목' 가설을 정공법으로 해소. 다음 목표 (v5): LLM 추가 확장 (Qwen2.5-3B) + ViT-L/14 재시도 + vLLM/Triton 프로덕션 서빙 통합.

---

## 📬 Contact

- **Email:** `910ehdbs@gmail.com`
- **LinkedIn:** [linkedin.com/in/AD-Styles](https://linkedin.com)
- **GitHub:** [github.com/AD-Styles](https://github.com/AD-Styles)
- **Hugging Face:** [huggingface.co/AD-Styles](https://huggingface.co/AD-Styles)
