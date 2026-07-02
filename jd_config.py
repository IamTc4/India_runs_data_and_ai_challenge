"""
jd_config.py — Job Description configuration and keyword expansions.

This is the FIXED JD for the Redrob AI/ML Engineer challenge.
Hardcoded synonym expansion replaces LLM-based expansion to keep
the pipeline CPU-only and network-free during ranking.
"""

# ---------------------------------------------------------------------------
# JOB DESCRIPTION TEXT (used for embedding in Stage 3 & 4)
# ---------------------------------------------------------------------------
JD_TEXT = """
We are looking for a strong AI/ML Engineer to join our applied AI team.

Requirements:
- 3–10 years of hands-on experience in machine learning and deep learning
- Proficiency in Python with frameworks like PyTorch, TensorFlow, or JAX
- Hands-on experience building and fine-tuning Large Language Models (LLMs)
- Experience with NLP pipelines: tokenization, embeddings, transformers, BERT, GPT
- Familiarity with MLOps: experiment tracking (MLflow, W&B), model deployment,
  feature stores, CI/CD for ML
- Experience with vector databases (Pinecone, FAISS, Weaviate, Chroma) and RAG systems
- Strong understanding of model evaluation, A/B testing, and statistical analysis
- Experience with cloud platforms (AWS SageMaker, GCP Vertex AI, Azure ML)
- Computer vision experience (CNNs, YOLO, OpenCV) is a plus
- Strong GitHub presence with open-source contributions or personal ML projects

Preferred Qualifications:
- B.Tech / M.Tech / PhD in Computer Science, Mathematics, or related STEM field
- Published research or technical blog posts in ML/AI
- Experience with distributed training (DeepSpeed, Horovod, Ray)
- Knowledge of model quantization, ONNX, and edge deployment

Work Mode: Hybrid / Remote
Location: India preferred
"""

# ---------------------------------------------------------------------------
# Stage 2: Hardcoded keyword expansions (replaces LLM query expansion)
# These were manually crafted from the JD above — zero latency, zero network
# ---------------------------------------------------------------------------
EXPANDED_KEYWORDS = [
    # Core ML/AI
    "machine learning", "ml", "deep learning", "dl", "artificial intelligence", "ai",
    "neural network", "neural networks", "model training", "model inference",

    # Frameworks
    "pytorch", "tensorflow", "keras", "jax", "scikit-learn", "sklearn",
    "hugging face", "huggingface",

    # LLM / GenAI
    "llm", "large language model", "gpt", "chatgpt", "llama", "mistral",
    "generative ai", "gen ai", "prompt engineering", "langchain",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft", "instruction tuning",
    "rlhf", "reinforcement learning from human feedback",

    # NLP
    "nlp", "natural language processing", "transformer", "transformers",
    "bert", "gpt-2", "gpt-4", "sentence transformers", "embeddings",
    "text classification", "sentiment analysis", "ner", "named entity recognition",
    "question answering", "summarization", "tokenization",

    # Computer Vision
    "computer vision", "cv", "cnn", "convolutional neural network",
    "image classification", "object detection", "yolo", "opencv",
    "image segmentation", "vision transformer", "vit",

    # MLOps
    "mlops", "mlflow", "wandb", "weights and biases", "experiment tracking",
    "model deployment", "model serving", "feature store", "kubeflow",
    "model monitoring", "data versioning", "dvc",

    # Vector / RAG
    "vector database", "vector db", "rag", "retrieval augmented generation",
    "pinecone", "weaviate", "chroma", "faiss", "embeddings",

    # Cloud ML
    "sagemaker", "vertex ai", "azure ml", "databricks",

    # Data Engineering (supporting)
    "python", "pandas", "numpy", "spark", "pyspark", "sql", "airflow",
    "data pipeline", "etl", "data engineering",

    # Infrastructure
    "docker", "kubernetes", "k8s", "ci/cd", "git", "github",
    "distributed training", "deepspeed", "horovod", "ray",

    # Stats & Math
    "statistics", "mathematics", "linear algebra", "calculus",
    "a/b testing", "hypothesis testing", "bayesian", "probability",
]

# ---------------------------------------------------------------------------
# Stage 1: Hard filtering configuration
# ---------------------------------------------------------------------------
FILTER_CONFIG = {
    # Experience bounds (years)
    "min_experience_years": 1.0,
    "max_experience_years": 40.0,

    # Candidates inactive for more than this many days get deprioritized
    "max_inactive_days": 730,  # 2 years — hard drop beyond this

    # Titles to completely exclude (irrelevant to AI/ML role)
    "blacklisted_titles": {
        "content writer", "content creator", "copywriter", "blogger",
        "sales executive", "sales representative", "sales associate",
        "account executive", "business development",
        "hr manager", "hr generalist", "human resources",
        "recruiter", "talent acquisition",
        "graphic designer", "ui designer", "ux designer",
        "accountant", "finance manager", "ca", "chartered accountant",
        "legal counsel", "lawyer", "attorney",
        "mechanical engineer", "civil engineer", "structural engineer",
        "customer support", "customer success", "customer service",
        "operations manager",  # unless in tech context — caught by BM25 later
        "marketing manager", "digital marketer", "seo specialist",
        "social media manager",
    },

    # BM25 stage hard cap
    "bm25_top_k": 5000,

    # Stage 3 light semantic hard cap
    "stage3_top_k": 1000,

    # Stage 4 deep semantic hard cap
    "stage4_top_k": 200,

    # Final output
    "final_top_k": 100,
}

# ---------------------------------------------------------------------------
# Stage 3: Light model config
# ---------------------------------------------------------------------------
LIGHT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ~80MB, very fast on CPU, good for initial semantic filtering

# ---------------------------------------------------------------------------
# Stage 4: Deep model config
# ---------------------------------------------------------------------------
DEEP_MODEL_NAME = "BAAI/bge-large-en-v1.5"
# ~1.3GB, 1024 dims, top-ranked on MTEB, great for precise matching
