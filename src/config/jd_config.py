"""
jd_config.py — Job Description text and hardcoded keyword expansions.
Replace JD_TEXT with any new job description to rank for a different role.
"""

# Full JD used for semantic embedding in Stages 3 & 4
JD_TEXT = """
We are looking for a strong AI/ML Engineer to join our applied AI team.

Requirements:
- 3-10 years of hands-on experience in machine learning and deep learning
- Proficiency in Python with frameworks like PyTorch, TensorFlow, or JAX
- Hands-on experience building and fine-tuning Large Language Models (LLMs)
- Experience with NLP pipelines: tokenization, embeddings, transformers, BERT, GPT
- MLOps experience: experiment tracking (MLflow, W&B), model deployment, feature stores
- Experience with vector databases (Pinecone, FAISS, Weaviate) and RAG systems
- Strong model evaluation skills, A/B testing, and statistical analysis
- Experience with cloud ML platforms (AWS SageMaker, GCP Vertex AI, Azure ML)
- Strong GitHub presence with open-source contributions
- Semantic search, information retrieval, ranking, and recommendation systems experience

Preferred:
- B.Tech / M.Tech / PhD in Computer Science, Mathematics, or STEM
- Published research or technical blog posts in ML/AI
- Experience with distributed training (DeepSpeed, Horovod, Ray)
"""

# Expanded keyword set for BM25 Stage 2 (replaces LLM query expansion)
BM25_KEYWORDS = [
    # Core ML/AI
    "machine learning", "deep learning", "artificial intelligence", "ai", "ml",
    # LLM / GenAI
    "llm", "large language model", "gpt", "llama", "mistral",
    "generative ai", "prompt engineering", "langchain", "fine-tuning", "lora", "rag",
    # NLP
    "nlp", "natural language processing", "transformers", "bert", "embeddings",
    "sentence transformers", "text classification", "named entity recognition",
    # Search & Retrieval
    "semantic search", "information retrieval", "ranking", "recommendation",
    "bm25", "vector database", "faiss", "pinecone", "weaviate", "qdrant",
    "hybrid search", "reranking", "ndcg", "mrr",
    # Frameworks
    "pytorch", "tensorflow", "keras", "scikit-learn", "hugging face",
    "xgboost", "lightgbm",
    # MLOps
    "mlops", "mlflow", "wandb", "model deployment", "model serving",
    "kubeflow", "sagemaker", "vertex ai",
    # Roles
    "ml engineer", "ai engineer", "data scientist", "applied scientist",
    "research scientist", "nlp engineer", "search engineer",
    # Infrastructure
    "python", "docker", "kubernetes", "distributed training",
]
