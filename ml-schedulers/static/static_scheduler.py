#!/usr/bin/env python3
# static_scheduler.py
# watches for pending pods and places them using a trained random forest

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

MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/static_scheduler_model.pkl')
FEATURE_NAMES_PATH = os.getenv('FEATURE_NAMES_PATH', '/app/model/feature_names.pkl')

SCHEDULER_NAME = os.getenv('SCHEDULER_NAME', 'static-scheduler')
NAMESPACE = os.getenv('WATCH_NAMESPACE', 'default')

# which physical node each class maps to (0=cpu, 1=gpu)
NODE_MAPPING = {
    0: os.getenv('CPU_NODE_NAME', 'k3d-fyp-server-0'),
    1: os.getenv('GPU_NODE_NAME', 'k3d-fyp-agent-0')
}

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/app/logs/scheduling_decisions.jsonl')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('static-scheduler')


class StaticMLScheduler:
    """kubernetes scheduler backed by a trained random forest"""

    def __init__(self, model_path: str, feature_names_path: str):
        self._init_kubernetes()
        self._load_model(model_path, feature_names_path)

        self.scheduling_count = 0
        self.node_0_count = 0
        self.node_1_count = 0

        # cache node loads so we're not hitting the api on every single pod
        self._node_loads_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 5

    def _init_kubernetes(self):
        try:
            # works when running inside the cluster
            config.load_incluster_config()
            logger.info("loaded in-cluster config")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("loaded local kubeconfig")
            except config.ConfigException as e:
                logger.error(f"failed to load kubernetes config: {e}")
                raise

        self.v1 = client.CoreV1Api()

    def _load_model(self, model_path: str, feature_names_path: str):
        model_path = Path(model_path)
        feature_names_path = Path(feature_names_path)

        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not feature_names_path.exists():
            raise FileNotFoundError(f"feature names not found: {feature_names_path}")

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)

        logger.info(f"loaded model from {model_path}")
        logger.info(f"expects {len(self.feature_names)} features: {self.feature_names}")

    def get_node_loads(self) -> dict:
        # approximate load by counting running pods per node
        # ideally would query prometheus but this is good enough for experiments
        current_time = time.time()

        if current_time - self._cache_timestamp < self._cache_ttl:
            return self._node_loads_cache

        try:
            pods = self.v1.list_pod_for_all_namespaces(watch=False)

            node_pod_counts = {node: 0 for node in NODE_MAPPING.values()}

            for pod in pods.items:
                node_name = pod.spec.node_name
                if node_name in node_pod_counts and pod.status.phase == "Running":
                    node_pod_counts[node_name] += 1

            MAX_PODS = 10
            self._node_loads_cache = {
                'cpu_node_load': min(node_pod_counts[NODE_MAPPING[0]] / MAX_PODS, 1.0),
                'gpu_node_load': min(node_pod_counts[NODE_MAPPING[1]] / MAX_PODS, 1.0)
            }
            self._cache_timestamp = current_time

        except Exception as e:
            logger.warning(f"failed to fetch node metrics, using defaults: {e}")
            self._node_loads_cache = {'cpu_node_load': 0.5, 'gpu_node_load': 0.5}

        return self._node_loads_cache

    def extract_features(self, pod) -> tuple[np.ndarray, dict]:
        """pull the 7 features out of the pod spec"""
        pod_name = pod.metadata.name.lower()
        labels = pod.metadata.labels or {}
        containers = pod.spec.containers or []

        # feature 1: task type - 0 for retrieval (cpu), 1 for generation (gpu)
        if any(x in pod_name for x in ['generation', 'ollama', 'llm', 'gpu']):
            task_type = 1
        elif any(x in pod_name for x in ['rag', 'retrieval', 'api', 'cpu']):
            task_type = 0
        elif labels.get('workload-type') == 'generation':
            task_type = 1
        elif labels.get('workload-type') == 'retrieval':
            task_type = 0
        else:
            task_type = 0  # default to retrieval if unknown

        # feature 2: input size
        input_size_mb = float(labels.get('input-size-mb', 50))

        # feature 3: model size (only relevant for generation tasks)
        model_size_b = float(labels.get('model-size-b', 2.0)) if task_type == 1 else 0.0

        # features 4 & 5: current load on each node
        loads = self.get_node_loads()
        cpu_node_load = loads['cpu_node_load']
        gpu_node_load = loads['gpu_node_load']

        # feature 6: memory request converted to GB
        memory_required_gb = 1.0
        if containers and containers[0].resources and containers[0].resources.requests:
            mem_request = containers[0].resources.requests.get('memory', '1Gi')
            memory_required_gb = self._parse_memory(mem_request)

        # feature 7: batch vs single
        is_batch = 1 if labels.get('batch', 'false').lower() == 'true' else 0

        feature_dict = {
            'task_type': task_type,
            'input_size_mb': input_size_mb,
            'model_size_b': model_size_b,
            'cpu_node_load': cpu_node_load,
            'gpu_node_load': gpu_node_load,
            'memory_required_gb': memory_required_gb,
            'is_batch_request': is_batch
        }

        # order must match what the model was trained on
        feature_vector = np.array([feature_dict[name] for name in self.feature_names])

        return feature_vector, feature_dict

    def _parse_memory(self, memory_str: str) -> float:
        """convert k8s memory string (e.g. '4Gi') to a float in GB"""
        memory_str = str(memory_str)
        if 'Gi' in memory_str:
            return float(memory_str.replace('Gi', ''))
        elif 'Mi' in memory_str:
            return float(memory_str.replace('Mi', '')) / 1024
        elif 'Ki' in memory_str:
            return float(memory_str.replace('Ki', '')) / (1024 * 1024)
        else:
            try:
                return float(memory_str) / (1024 ** 3)
            except ValueError:
                return 1.0

    def schedule_pod(self, pod) -> bool:
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace

        logger.info(f"scheduling {namespace}/{pod_name}")

        try:
            feature_vector, feature_dict = self.extract_features(pod)
            logger.info(f"features: {feature_dict}")

            prediction = self.model.predict([feature_vector])[0]
            probabilities = self.model.predict_proba([feature_vector])[0]
            confidence = max(probabilities)

            target_node = NODE_MAPPING[prediction]

            logger.info(f"prediction: node {prediction} ({target_node}), confidence: {confidence:.2%}")
            logger.info(f"class probabilities: cpu={probabilities[0]:.2%}, gpu={probabilities[1]:.2%}")

            binding = client.V1Binding(
                api_version="v1",
                kind="Binding",
                metadata=client.V1ObjectMeta(name=pod_name, namespace=namespace),
                target=client.V1ObjectReference(
                    api_version="v1",
                    kind="Node",
                    name=target_node
                )
            )

            self.v1.create_namespaced_binding(
                namespace=namespace,
                body=binding,
                _preload_content=False
            )

            logger.info(f"bound {pod_name} to {target_node}")

            self.scheduling_count += 1
            if prediction == 0:
                self.node_0_count += 1
            else:
                self.node_1_count += 1

            self._log_decision(pod_name, namespace, target_node, feature_dict, prediction, confidence)

            return True

        except ApiException as e:
            if e.status == 409:
                logger.warning(f"pod {pod_name} already bound")
            else:
                logger.error(f"api error scheduling {pod_name}: {e.status} {e.reason}")
            return False
        except Exception as e:
            logger.error(f"error scheduling {pod_name}: {e}", exc_info=True)
            return False

    def _log_decision(self, pod_name, namespace, target_node, features, prediction, confidence):
        """append scheduling decision to jsonl for later analysis"""
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
            logger.warning(f"failed to log decision: {e}")

    def run(self):
        logger.info(f"static ml scheduler started")
        logger.info(f"watching namespace: {NAMESPACE}, node mapping: {NODE_MAPPING}")

        w = watch.Watch()

        while True:
            try:
                logger.info("watching for pending pods...")

                for event in w.stream(
                    self.v1.list_namespaced_pod,
                    namespace=NAMESPACE,
                    timeout_seconds=0
                ):
                    event_type = event['type']
                    pod = event['object']

                    if (event_type == "ADDED" and
                        pod.status.phase == "Pending" and
                        pod.spec.scheduler_name == SCHEDULER_NAME and
                        not pod.spec.node_name):

                        self.schedule_pod(pod)
                        logger.info(f"totals: scheduled={self.scheduling_count}, cpu={self.node_0_count}, gpu={self.node_1_count}")

            except ApiException as e:
                logger.error(f"api error in watch stream: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"watch stream error: {e}", exc_info=True)
                logger.info("reconnecting in 5 seconds...")
                time.sleep(5)


def main():
    logger.info("starting static ml scheduler")

    try:
        scheduler = StaticMLScheduler(MODEL_PATH, FEATURE_NAMES_PATH)
        scheduler.run()
    except FileNotFoundError as e:
        print(f"error: {e}")
        print("train the model first: python generate_dataset.py && python train_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
