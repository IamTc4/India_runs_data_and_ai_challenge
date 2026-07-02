"""
ai_skills.py — Curated AI/ML skills taxonomy for the Redrob ranking challenge.

Tiers reflect how directly a skill maps to the AI/ML Engineer job description.
Tier 1 = core must-have  (weight 3.0)
Tier 2 = strong signal   (weight 2.0)
Tier 3 = supporting      (weight 1.0)
"""

# ---------------------------------------------------------------------------
# Tier 1 — Core AI/ML must-have skills
# ---------------------------------------------------------------------------
TIER1_SKILLS = {
    # Languages & Frameworks
    "python", "pytorch", "tensorflow", "keras",
    # ML/DL Concepts
    "machine learning", "deep learning", "neural network", "neural networks",
    "reinforcement learning", "supervised learning", "unsupervised learning",
    # LLM / GenAI
    "llm", "llms", "large language model", "large language models",
    "generative ai", "gen ai", "genai", "gpt", "chatgpt", "openai",
    "langchain", "llama", "mistral", "gemini", "claude", "prompt engineering",
    # NLP
    "nlp", "natural language processing", "transformers", "bert", "gpt-4",
    "text classification", "sentiment analysis", "named entity recognition",
    # Computer Vision
    "computer vision", "opencv", "image classification", "object detection",
    "yolo", "cnn", "convolutional neural network",
    # MLOps
    "mlops", "model deployment", "model serving", "ml pipeline",
    "feature store", "experiment tracking", "mlflow", "wandb",
}

# ---------------------------------------------------------------------------
# Tier 2 — Strong AI/ML supporting skills
# ---------------------------------------------------------------------------
TIER2_SKILLS = {
    # Libraries
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "hugging face", "huggingface", "sentence transformers",
    "spacy", "nltk", "gensim",
    # Data / Vector
    "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "vector database", "vector db", "pinecone", "weaviate", "faiss", "chroma",
    "rag", "retrieval augmented generation", "embeddings",
    # Cloud / Infra ML
    "sagemaker", "vertex ai", "azure ml", "databricks", "kubeflow",
    "docker", "kubernetes", "ray", "dask",
    # Fine-tuning / Training
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "model fine-tuning", "transfer learning",
    # Data Engineering (adjacent)
    "spark", "pyspark", "apache spark", "airflow", "apache airflow",
    "dbt", "sql", "data pipeline", "etl",
    # Stats & Math
    "statistics", "linear algebra", "calculus", "probability",
    "a/b testing", "hypothesis testing", "bayesian",
    # APIs / Deployment
    "fastapi", "flask", "rest api", "grpc", "onnx",
}

# ---------------------------------------------------------------------------
# Tier 3 — Broader tech skills with some relevance
# ---------------------------------------------------------------------------
TIER3_SKILLS = {
    "java", "scala", "r", "julia", "c++", "go", "rust",
    "cloud", "aws", "gcp", "azure", "s3", "bigquery", "redshift",
    "git", "github", "ci/cd", "linux", "bash",
    "postgresql", "mysql", "mongodb", "redis",
    "snowflake", "hadoop", "kafka",
    "excel", "power bi", "tableau", "looker",
}

# ---------------------------------------------------------------------------
# Tier weights
# ---------------------------------------------------------------------------
TIER_WEIGHTS = {
    1: 3.0,
    2: 2.0,
    3: 1.0,
}

# ---------------------------------------------------------------------------
# Job description keywords — used in title/summary matching
# ---------------------------------------------------------------------------
JD_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "mlops", "deep learning", "research scientist",
    "ai researcher", "applied scientist", "llm engineer",
    "computer vision", "ml researcher", "ml platform", "genai",
]

JD_SUMMARY_KEYWORDS = [
    "machine learning", "deep learning", "neural", "pytorch", "tensorflow",
    "llm", "nlp", "computer vision", "mlops", "model training", "model deployment",
    "generative ai", "transformers", "fine-tuning", "embeddings",
    "feature engineering", "data science", "artificial intelligence",
]

# Relevant industries for an AI company
AI_INDUSTRIES = {
    "artificial intelligence", "machine learning", "data science",
    "technology", "software", "it services", "saas", "cloud computing",
    "research", "edtech", "fintech", "healthtech", "deeptech",
    "internet", "e-commerce", "startup",
}

# Preferred company sizes (larger companies typically have better ML teams)
COMPANY_SIZE_SCORE = {
    "1-10": 0.5,
    "11-50": 0.6,
    "51-200": 0.7,
    "201-500": 0.75,
    "501-1000": 0.8,
    "1001-5000": 0.9,
    "5001-10000": 0.95,
    "10001+": 1.0,
}

# Education field relevance
RELEVANT_FIELDS = {
    "computer science", "cs", "software engineering", "information technology",
    "electrical engineering", "electronics", "mathematics", "statistics",
    "data science", "machine learning", "artificial intelligence",
    "computational mathematics", "applied mathematics", "physics",
    "information systems",
}

# Relevant degrees
RELEVANT_DEGREES = {
    "b.tech", "b.e.", "b.sc", "bsc", "b.s.", "bs",
    "m.tech", "m.e.", "m.sc", "msc", "m.s.", "ms",
    "ph.d", "phd", "doctorate",
    "b.tech.", "m.tech.", "bachelor", "master",
}
