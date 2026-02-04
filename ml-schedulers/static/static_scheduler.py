#!/usr/bin/env python3
"""
ml-schedulers/static/static_scheduler.py

Static ML Scheduler - External Kubernetes scheduler using Random Forest.

This scheduler:
1. Watches for pods with schedulerName: static-scheduler
2. Extracts features from pod specification
3. Uses trained Random Forest to predict optimal node
4. Binds pod to predicted node via Kubernetes API

The scheduler runs as a pod in the cluster with appropriate RBAC permissions.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths (mounted from ConfigMap or baked into image)
MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/static_scheduler_model.pkl')
FEATURE_NAMES_PATH = os.getenv('FEATURE_NAMES_PATH', '/app/model/feature_names.pkl')

# Scheduler configuration
SCHEDULER_NAME = os.getenv('SCHEDULER_NAME', 'static-scheduler')
NAMESPACE = os.getenv('WATCH_NAMESPACE', 'default')

# Node mapping
NODE_MAPPING = {
    0: os.getenv('CPU_NODE_NAME', 'minikube'),
    1: os.getenv('GPU_NODE_NAME', 'minikube-m02')
}

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/app/logs/scheduling_decisions.jsonl')

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('static-scheduler')


class StaticMLScheduler:
    """
    Kubernetes scheduler using offline-trained Random Forest model.
    
    This scheduler demonstrates supervised learning for pod placement.
    It predicts the optimal node based on pod features extracted from
    the pod specification.
    """
    
    def __init__(self, model_path: str, feature_names_path: str):
        """
        Initialize scheduler with trained model.
        
        Args:
            model_path: Path to trained Random Forest model (.pkl)
            feature_names_path: Path to feature names list (.pkl)
        """
        logger.info("Initializing Static ML Scheduler...")
        
        # Load Kubernetes configuration
        self._init_kubernetes()
        
        # Load ML model
        self._load_model(model_path, feature_names_path)
        
        # Initialize metrics
        self.scheduling_count = 0
        self.node_0_count = 0
        self.node_1_count = 0
        
        # Node load cache
        self._node_loads_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 5  # seconds
        
        logger.info("Static ML Scheduler initialized successfully")
    
    def _init_kubernetes(self):
        """Initialize Kubernetes client."""
        try:
            # Try in-cluster config first (when running as pod)
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            # Fall back to local kubeconfig (for development)
            try:
                config.load_kube_config()
                logger.info("Loaded local Kubernetes configuration")
            except config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise
        
        self.v1 = client.CoreV1Api()
    
    def _load_model(self, model_path: str, feature_names_path: str):
        """Load trained model and feature names."""
        model_path = Path(model_path)
        feature_names_path = Path(feature_names_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not feature_names_path.exists():
            raise FileNotFoundError(f"Feature names not found: {feature_names_path}")
        
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)
        
        logger.info(f"Loaded model from: {model_path}")
        logger.info(f"Model expects {len(self.feature_names)} features: {self.feature_names}")
    
    def get_node_loads(self) -> dict:
        """
        Fetch current node utilization metrics.
        
        In a production system, this would query Prometheus or Metrics Server.
        Here we approximate load by counting running pods per node.
        
        Returns:
            Dict with 'cpu_node_load' and 'gpu_node_load' in [0, 1]
        """
        current_time = time.time()
        
        # Return cached values if still valid
        if current_time - self._cache_timestamp < self._cache_ttl:
            return self._node_loads_cache
        
        try:
            pods = self.v1.list_pod_for_all_namespaces(watch=False)
            
            node_pod_counts = {node: 0 for node in NODE_MAPPING.values()}
            
            for pod in pods.items:
                node_name = pod.spec.node_name
                if node_name in node_pod_counts and pod.status.phase == "Running":
                    node_pod_counts[node_name] += 1
            
            # Normalize to [0, 1] assuming max ~10 pods per node
            MAX_PODS = 10
            self._node_loads_cache = {
                'cpu_node_load': min(node_pod_counts[NODE_MAPPING[0]] / MAX_PODS, 1.0),
                'gpu_node_load': min(node_pod_counts[NODE_MAPPING[1]] / MAX_PODS, 1.0)
            }
            self._cache_timestamp = current_time
            
            logger.debug(f"Node loads: {self._node_loads_cache}")
            
        except Exception as e:
            logger.warning(f"Failed to fetch node metrics: {e}")
            # Return default values on error
            self._node_loads_cache = {
                'cpu_node_load': 0.5,
                'gpu_node_load': 0.5
            }
        
        return self._node_loads_cache
    
    def extract_features(self, pod) -> tuple[np.ndarray, dict]:
        """
        Extract feature vector from pod specification.
        
        Maps pod attributes to the feature format expected by the model.
        
        Args:
            pod: Kubernetes pod object
            
        Returns:
            Tuple of (feature_vector as numpy array, feature_dict for logging)
        """
        pod_name = pod.metadata.name.lower()
        labels = pod.metadata.labels or {}
        containers = pod.spec.containers or []
        
        # ================================================================
        # FEATURE 1: task_type (0=retrieval/CPU, 1=generation/GPU)
        # ================================================================
        # Infer from pod name or labels
        if any(x in pod_name for x in ['generation', 'ollama', 'llm', 'gpu']):
            task_type = 1
        elif any(x in pod_name for x in ['rag', 'retrieval', 'api', 'cpu']):
            task_type = 0
        elif labels.get('workload-type') == 'generation':
            task_type = 1
        elif labels.get('workload-type') == 'retrieval':
            task_type = 0
        else:
            # Default to retrieval (safer for unknown pods)
            task_type = 0
        
        # ================================================================
        # FEATURE 2: input_size_mb (simulated from labels or default)
        # ================================================================
        input_size_mb = float(labels.get('input-size-mb', 50))
        
        # ================================================================
        # FEATURE 3: model_size_b (extract from labels or infer)
        # ================================================================
        if task_type == 1:
            # For generation tasks, try to get from labels
            model_size_b = float(labels.get('model-size-b', 2.0))
        else:
            model_size_b = 0.0
        
        # ================================================================
        # FEATURE 4 & 5: Node loads
        # ================================================================
        loads = self.get_node_loads()
        cpu_node_load = loads['cpu_node_load']
        gpu_node_load = loads['gpu_node_load']
        
        # ================================================================
        # FEATURE 6: memory_required_gb
        # ================================================================
        memory_required_gb = 1.0  # Default
        if containers and containers[0].resources and containers[0].resources.requests:
            mem_request = containers[0].resources.requests.get('memory', '1Gi')
            memory_required_gb = self._parse_memory(mem_request)
        
        # ================================================================
        # FEATURE 7: is_batch_request
        # ================================================================
        is_batch = 1 if labels.get('batch', 'false').lower() == 'true' else 0
        
        # Build feature dictionary
        feature_dict = {
            'task_type': task_type,
            'input_size_mb': input_size_mb,
            'model_size_b': model_size_b,
            'cpu_node_load': cpu_node_load,
            'gpu_node_load': gpu_node_load,
            'memory_required_gb': memory_required_gb,
            'is_batch_request': is_batch
        }
        
        # Create feature vector in correct order
        feature_vector = np.array([feature_dict[name] for name in self.feature_names])
        
        return feature_vector, feature_dict
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse Kubernetes memory string to GB."""
        memory_str = str(memory_str)
        if 'Gi' in memory_str:
            return float(memory_str.replace('Gi', ''))
        elif 'Mi' in memory_str:
            return float(memory_str.replace('Mi', '')) / 1024
        elif 'Ki' in memory_str:
            return float(memory_str.replace('Ki', '')) / (1024 * 1024)
        else:
            # Assume bytes
            try:
                return float(memory_str) / (1024 ** 3)
            except ValueError:
                return 1.0
    
    def schedule_pod(self, pod) -> bool:
        """
        Schedule a pod using ML prediction.
        
        Args:
            pod: Kubernetes pod object
            
        Returns:
            True if successfully scheduled, False otherwise
        """
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SCHEDULING: {namespace}/{pod_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Extract features
            feature_vector, feature_dict = self.extract_features(pod)
            logger.info(f"Features: {feature_dict}")
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            probabilities = self.model.predict_proba([feature_vector])[0]
            confidence = max(probabilities)
            
            target_node = NODE_MAPPING[prediction]
            
            logger.info(f"Prediction: Node {prediction} ({target_node})")
            logger.info(f"Confidence: {confidence:.2%}")
            logger.info(f"Probabilities: CPU={probabilities[0]:.2%}, GPU={probabilities[1]:.2%}")
            
            # Create binding
            binding = client.V1Binding(
                api_version="v1",
                kind="Binding",
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    namespace=namespace
                ),
                target=client.V1ObjectReference(
                    api_version="v1",
                    kind="Node",
                    name=target_node
                )
            )
            
            # Execute binding
            self.v1.create_namespaced_binding(
                namespace=namespace,
                body=binding,
                _preload_content=False
            )
            
            logger.info(f"✓ Successfully bound {pod_name} to {target_node}")
            
            # Update metrics
            self.scheduling_count += 1
            if prediction == 0:
                self.node_0_count += 1
            else:
                self.node_1_count += 1
            
            # Log decision for analysis
            self._log_decision(pod_name, namespace, target_node, 
                             feature_dict, prediction, confidence)
            
            return True
            
        except ApiException as e:
            if e.status == 409:
                logger.warning(f"Pod {pod_name} already bound (conflict)")
            else:
                logger.error(f"API error scheduling {pod_name}: {e.status} {e.reason}")
            return False
        except Exception as e:
            logger.error(f"Error scheduling {pod_name}: {e}", exc_info=True)
            return False
    
    def _log_decision(self, pod_name: str, namespace: str, target_node: str,
                      features: dict, prediction: int, confidence: float):
        """Log scheduling decision to JSONL file."""
        try:
            log_dir = Path(LOG_FILE).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'pod_name': pod_name,
                'namespace': namespace,
                'target_node': target_node,
                'prediction': prediction,
                'confidence': confidence,
                'features': features,
                'scheduler': 'static-ml'
            }
            
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")
    
    def run(self):
        """
        Main scheduler loop.
        
        Watches for pending pods with our scheduler name and schedules them.
        """
        logger.info("\n" + "=" * 60)
        logger.info("STATIC ML SCHEDULER STARTED")
        logger.info(f"Scheduler name: {SCHEDULER_NAME}")
        logger.info(f"Watching namespace: {NAMESPACE}")
        logger.info(f"Node mapping: {NODE_MAPPING}")
        logger.info("=" * 60 + "\n")
        
        w = watch.Watch()
        
        while True:
            try:
                logger.info("Watching for pending pods...")
                
                for event in w.stream(
                    self.v1.list_namespaced_pod,
                    namespace=NAMESPACE,
                    timeout_seconds=0  # Infinite watch, reconnect on timeout
                ):
                    event_type = event['type']
                    pod = event['object']
                    
                    # Only process pods that:
                    # 1. Are newly ADDED
                    # 2. Are in Pending phase
                    # 3. Have our scheduler name
                    # 4. Don't already have a node assigned
                    if (event_type == "ADDED" and
                        pod.status.phase == "Pending" and
                        pod.spec.scheduler_name == SCHEDULER_NAME and
                        not pod.spec.node_name):
                        
                        self.schedule_pod(pod)
                        
                        # Log current stats
                        logger.info(f"Stats: Total={self.scheduling_count}, "
                                  f"CPU={self.node_0_count}, GPU={self.node_1_count}")
                        
            except ApiException as e:
                logger.error(f"API error in watch stream: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Watch stream error: {e}", exc_info=True)
                logger.info("Reconnecting in 5 seconds...")
                time.sleep(5)


def main():
    """Entry point."""
    print("=" * 60)
    print("STATIC ML SCHEDULER")
    print("Random Forest-based Kubernetes Pod Scheduler")
    print("=" * 60)
    
    try:
        scheduler = StaticMLScheduler(MODEL_PATH, FEATURE_NAMES_PATH)
        scheduler.run()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure to train the model first:")
        print("  python generate_dataset.py")
        print("  python train_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
