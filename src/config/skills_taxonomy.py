"""
skills_taxonomy.py — AI/ML skills taxonomy with tier weights.

Tier 1 = Core must-have    (weight 3.0)
Tier 2 = Strong signal     (weight 2.0)
Tier 3 = Supporting        (weight 1.0)
"""

TIER1_SKILLS = frozenset({
    "python", "pytorch", "tensorflow", "keras",
    "machine learning", "deep learning", "neural network",
    "llm", "llms", "large language model", "generative ai", "genai",
    "langchain", "llama", "mistral", "gpt", "prompt engineering",
    "nlp", "natural language processing", "transformers", "bert",
    "computer vision", "opencv", "cnn", "yolo",
    "mlops", "model deployment", "model serving", "mlflow", "wandb",
    "embeddings", "embedding", "sentence-transformers",
    "vector database", "faiss", "pinecone", "weaviate", "qdrant",
    "semantic search", "information retrieval", "ranking", "recommendation",
    "fine-tuning", "fine tuning", "lora", "qlora", "rag",
})

TIER2_SKILLS = frozenset({
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "hugging face", "huggingface", "spacy", "nltk",
    "pandas", "numpy", "scipy", "matplotlib",
    "sagemaker", "vertex ai", "azure ml", "databricks", "kubeflow",
    "docker", "kubernetes", "ray", "dask",
    "fastapi", "flask", "rest api", "onnx",
    "statistics", "linear algebra", "bayesian", "a/b testing",
    "spark", "pyspark", "airflow", "dbt", "sql",
    "bm25", "reranking", "re-ranking", "ndcg", "mrr",
})

TIER3_SKILLS = frozenset({
    "java", "scala", "r", "c++", "go",
    "aws", "gcp", "azure", "s3", "bigquery",
    "git", "github", "ci/cd", "linux",
    "postgresql", "mysql", "mongodb", "redis",
    "snowflake", "hadoop", "kafka", "elasticsearch",
})

TIER_WEIGHTS = {1: 3.0, 2: 2.0, 3: 1.0}

# Job description relevance
JD_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "mlops", "deep learning", "research scientist",
    "applied scientist", "llm engineer", "search engineer",
    "ranking engineer", "recommendation engineer", "retrieval engineer",
]

JD_SUMMARY_KEYWORDS = [
    "machine learning", "deep learning", "neural", "pytorch", "tensorflow",
    "llm", "nlp", "mlops", "model training", "model deployment",
    "generative ai", "transformers", "fine-tuning", "embeddings",
    "feature engineering", "data science", "artificial intelligence",
    "vector", "retrieval", "ranking", "recommendation", "search",
]

AI_INDUSTRIES = frozenset({
    "ai/ml", "software", "fintech", "e-commerce", "food delivery",
    "saas", "healthtech", "edtech", "deeptech", "internet",
})

BAD_INDUSTRIES = frozenset({
    "it services", "manufacturing", "transportation",
    "paper products", "conglomerate",
})

PRODUCT_COMPANIES = frozenset({
    "swiggy", "zomato", "flipkart", "meesho", "razorpay", "zepto", "blinkit",
    "cred", "groww", "zerodha", "sharechat", "nykaa", "paytm",
    "ola", "rapido", "amazon", "google", "microsoft", "meta",
    "netflix", "uber", "linkedin", "salesforce", "adobe",
    "atlassian", "freshworks", "zoho", "sarvam", "haptik", "rephrase",
})

CONSULTING_COMPANIES = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "hexaware", "mindtree", "ltimindtree",
    "persistent", "cyient", "niit", "kpit", "zensar",
})

TARGET_CITIES = frozenset({
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurgaon", "gurugram", "chennai", "delhi ncr", "ncr",
})

RELEVANT_FIELDS = frozenset({
    "computer science", "cs", "software engineering", "information technology",
    "electrical engineering", "mathematics", "statistics",
    "data science", "machine learning", "artificial intelligence",
    "applied mathematics", "physics", "computational mathematics",
})
