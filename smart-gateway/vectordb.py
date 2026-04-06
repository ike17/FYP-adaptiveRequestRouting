# smart-gateway/vectordb.py
# In-memory ChromaDB vector store for the retrieval stage of the RAG pipeline.
# Identical knowledge base to the original rag-app — moved here so the gateway
# owns the full request lifecycle without needing a separate retrieval service.

import logging

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorDB:
    """Wraps ChromaDB with a small hardcoded knowledge base for experiments."""

    def __init__(self):
        logger.info("initialising chromadb vector store...")

        # ephemeral = in-memory only, no persistence between restarts
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # chromadb uses all-MiniLM-L6-v2 as the default embedding model
        self.collection = self.client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

        self._initialize_knowledge_base()
        logger.info(f"vector store ready with {self.collection.count()} documents")

    def _initialize_knowledge_base(self):
        documents = [
            # kubernetes concepts
            "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.",
            "A Kubernetes pod is the smallest deployable unit and can contain one or more containers that share storage and network resources.",
            "Kubernetes uses a declarative configuration model where users specify the desired state and the system works to maintain it.",
            "The Kubernetes scheduler assigns pods to nodes based on resource requirements, constraints, and affinity rules.",
            "Kubernetes services provide stable networking and load balancing for pods, abstracting away individual pod IP addresses.",

            # machine learning concepts
            "Machine learning enables systems to learn patterns from data without being explicitly programmed for each scenario.",
            "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
            "Reinforcement learning agents learn optimal policies through trial and error interactions with an environment.",
            "Random Forest is an ensemble learning method that constructs multiple decision trees and outputs their majority vote.",
            "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.",

            # rag and llm concepts
            "Retrieval-Augmented Generation combines information retrieval with language model generation for grounded responses.",
            "Large Language Models like GPT and Gemma use transformer architectures to generate human-like text.",
            "Vector embeddings represent text as dense numerical vectors that capture semantic meaning.",
            "Semantic search uses vector similarity to find documents related by meaning rather than exact keywords.",
            "GPU acceleration significantly improves LLM inference speed through parallel tensor operations.",

            # scheduling concepts
            "Heterogeneous clusters contain nodes with different hardware capabilities like CPUs, GPUs, and TPUs.",
            "Workload scheduling in distributed systems aims to optimize metrics like latency, throughput, and resource utilization.",
            "Piecewise stationary environments exhibit stable behavior for periods followed by abrupt regime changes.",
            "Contextual bandits extend multi-armed bandits by incorporating feature information to guide arm selection.",
            "Online learning algorithms adapt their models incrementally as new data arrives without full retraining.",
        ]

        self.collection.add(
            documents=documents,
            ids=[f"doc_{i:03d}" for i in range(len(documents))]
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Semantic search — the CPU-bound part of the RAG pipeline."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        if results and results.get('documents'):
            return results['documents'][0]
        return []

    def get_stats(self) -> dict:
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
        }
