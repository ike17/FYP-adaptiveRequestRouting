"""
rag-app/vectordb.py
In-memory vector database using ChromaDB for the retrieval stage.
This represents the CPU-bound portion of the RAG pipeline.
"""

import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Lightweight vector store for RAG retrieval.
    Uses ChromaDB with in-memory storage and default embedding function.
    """
    
    def __init__(self):
        """Initialize ChromaDB with ephemeral (in-memory) storage."""
        logger.info("Initializing ChromaDB vector store...")
        
        # Use ephemeral client (in-memory, no persistence)
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create collection with default embedding function
        # ChromaDB will use all-MiniLM-L6-v2 by default
        self.collection = self.client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load sample documents
        self._initialize_knowledge_base()
        logger.info(f"Vector store initialized with {self.collection.count()} documents")
    
    def _initialize_knowledge_base(self):
        """
        Pre-load the vector database with sample documents.
        In production, this would load from external sources.
        """
        documents = [
            # Kubernetes concepts
            "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.",
            "A Kubernetes pod is the smallest deployable unit and can contain one or more containers that share storage and network resources.",
            "Kubernetes uses a declarative configuration model where users specify the desired state and the system works to maintain it.",
            "The Kubernetes scheduler assigns pods to nodes based on resource requirements, constraints, and affinity rules.",
            "Kubernetes services provide stable networking and load balancing for pods, abstracting away individual pod IP addresses.",
            
            # Machine Learning concepts  
            "Machine learning enables systems to learn patterns from data without being explicitly programmed for each scenario.",
            "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
            "Reinforcement learning agents learn optimal policies through trial and error interactions with an environment.",
            "Random Forest is an ensemble learning method that constructs multiple decision trees and outputs their majority vote.",
            "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.",
            
            # RAG and LLM concepts
            "Retrieval-Augmented Generation combines information retrieval with language model generation for grounded responses.",
            "Large Language Models like GPT and Gemma use transformer architectures to generate human-like text.",
            "Vector embeddings represent text as dense numerical vectors that capture semantic meaning.",
            "Semantic search uses vector similarity to find documents related by meaning rather than exact keywords.",
            "GPU acceleration significantly improves LLM inference speed through parallel tensor operations.",
            
            # Scheduling concepts
            "Heterogeneous clusters contain nodes with different hardware capabilities like CPUs, GPUs, and TPUs.",
            "Workload scheduling in distributed systems aims to optimize metrics like latency, throughput, and resource utilization.",
            "Piecewise stationary environments exhibit stable behavior for periods followed by abrupt regime changes.",
            "Contextual bandits extend multi-armed bandits by incorporating feature information to guide arm selection.",
            "Online learning algorithms adapt their models incrementally as new data arrives without full retraining.",
        ]
        
        # Add documents with auto-generated IDs
        ids = [f"doc_{i:03d}" for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            ids=ids
        )
    
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Perform semantic similarity search.
        
        This is the main CPU-bound operation in the retrieval stage.
        The embedding computation and vector search happen here.
        
        Args:
            query: The user's question or search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant document strings
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Extract documents from nested structure
        if results and results.get('documents'):
            return results['documents'][0]
        return []
    
    def get_stats(self) -> dict:
        """Return database statistics."""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }
