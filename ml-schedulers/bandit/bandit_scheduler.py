#!/usr/bin/env python3
# bandit_scheduler.py
# adaptive kubernetes scheduler using thompson sampling with regime detection
# rewards are simulated - see adaptive_bandit_scheduler.py for the real-metrics version

import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

SCHEDULER_NAME = os.getenv('SCHEDULER_NAME', 'k3d-scheduler')
NAMESPACE = os.getenv('WATCH_NAMESPACE', 'default')

NODE_MAPPING = {
    0: os.getenv('CPU_NODE_NAME', 'k3d-fyp-server-0'),
    1: os.getenv('GPU_NODE_NAME', 'k3d-fyp-agent-0')
}

# thompson sampling priors
INITIAL_ALPHA = 1.0
INITIAL_BETA = 1.0

# regime detection
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '20'))
RESET_THRESHOLD = float(os.getenv('RESET_THRESHOLD', '0.4'))

# shock simulation - injects a performance degradation at a specific query
SHOCK_ENABLED = os.getenv('SHOCK_ENABLED', 'true').lower() == 'true'
SHOCK_QUERY = int(os.getenv('SHOCK_QUERY', '50'))
SIMULATE_REWARDS = os.getenv('SIMULATE_REWARDS', 'true').lower() == 'true'

RAG_APP_METRICS_URL = os.getenv('RAG_APP_METRICS_URL', 'http://rag-app-service:8000/metrics')

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/app/logs/bandit_decisions.jsonl')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('bandit-scheduler')


class ThompsonSamplingBandit:
    """
    beta-bernoulli bandit with a sliding window for detecting regime changes.

    each arm (node) has a beta(alpha, beta) posterior. we sample from each
    and pick the arm with the highest sample. when rewards drop significantly
    below the baseline we assume something changed and reset to uniform priors.
    """

    def __init__(self, n_arms: int = 2, window_size: int = 20,
                 reset_threshold: float = 0.4):
        self.n_arms = n_arms
        self.window_size = window_size
        self.reset_threshold = reset_threshold

        self.alpha = np.array([INITIAL_ALPHA] * n_arms)
        self.beta = np.array([INITIAL_BETA] * n_arms)

        self.reward_window = deque(maxlen=window_size)
        self.baseline_reward: Optional[float] = None
        self.last_baseline_update = 0

        self.total_pulls = 0
        self.arm_pulls = np.zeros(n_arms, dtype=int)
        self.total_reward = 0.0
        self.arm_rewards = np.zeros(n_arms)
        self.reset_count = 0

        logger.info(f"bandit initialised: {n_arms} arms, window={window_size}, threshold={reset_threshold}")

    def select_arm(self) -> int:
        # sample from each arm's posterior and pick the best
        samples = np.array([
            np.random.beta(self.alpha[a], self.beta[a])
            for a in range(self.n_arms)
        ])

        selected_arm = np.argmax(samples)
        logger.debug(f"thompson samples: {samples}, selected: {selected_arm}")

        return int(selected_arm)

    def update(self, arm: int, reward: float) -> dict:
        reward = np.clip(reward, 0.0, 1.0)

        # fractional update: treat reward as expected successes
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)

        self.total_pulls += 1
        self.arm_pulls[arm] += 1
        self.total_reward += reward
        self.arm_rewards[arm] += reward

        self.reward_window.append(reward)

        regime_change = self._check_regime_change()

        return {
            'arm': arm,
            'reward': reward,
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'regime_change_detected': regime_change
        }

    def _check_regime_change(self) -> bool:
        if len(self.reward_window) < self.window_size:
            return False

        current_avg = np.mean(self.reward_window)

        # set baseline from the first full window
        if self.baseline_reward is None:
            self.baseline_reward = current_avg
            self.last_baseline_update = self.total_pulls
            logger.info(f"baseline reward established: {self.baseline_reward:.4f}")
            return False

        # if recent average drops below threshold * baseline, something changed
        if current_avg < self.baseline_reward * self.reset_threshold:
            logger.warning(f"regime change detected! current avg: {current_avg:.4f}, baseline: {self.baseline_reward:.4f}")
            self._reset()
            return True

        # slowly drift the baseline to account for gradual changes
        if self.total_pulls - self.last_baseline_update > self.window_size * 2:
            self.baseline_reward = 0.9 * self.baseline_reward + 0.1 * current_avg
            self.last_baseline_update = self.total_pulls

        return False

    def _reset(self):
        """reset posteriors to uniform so the bandit re-explores"""
        self.alpha = np.array([INITIAL_ALPHA] * self.n_arms)
        self.beta = np.array([INITIAL_BETA] * self.n_arms)
        self.reward_window.clear()
        self.baseline_reward = None
        self.reset_count += 1
        logger.info("posteriors reset, re-exploring")

    def get_statistics(self) -> dict:
        return {
            'total_pulls': self.total_pulls,
            'arm_pulls': self.arm_pulls.tolist(),
            'avg_reward': self.total_reward / max(self.total_pulls, 1),
            'arm_avg_rewards': [
                self.arm_rewards[a] / max(self.arm_pulls[a], 1)
                for a in range(self.n_arms)
            ],
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'reset_count': self.reset_count,
            'current_baseline': self.baseline_reward
        }

    def get_expected_rewards(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)


class BanditScheduler:
    """main scheduler class - wraps the bandit and handles k8s pod binding"""

    def __init__(self):
        logger.info("initialising bandit scheduler...")

        self._init_kubernetes()

        self.bandit = ThompsonSamplingBandit(
            n_arms=2,
            window_size=WINDOW_SIZE,
            reset_threshold=RESET_THRESHOLD
        )

        self.query_counter = 0
        self.decisions = []

        logger.info(f"shock simulation: {'enabled at query ' + str(SHOCK_QUERY) if SHOCK_ENABLED else 'disabled'}")

    def _init_kubernetes(self):
        try:
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

    def simulate_reward(self, arm: int, task_type: int) -> tuple[float, float]:
        """
        simulate latency and reward based on which node the pod was placed on.

        before shock query:  gpu node is fast for generation (0.4-0.7s)
        after shock query:   gpu node degrades (13-17s), simulating thermal throttling
        this is what forces the bandit to adapt and discover the new optimum.
        """
        MAX_LATENCY = 20.0

        is_shock_active = SHOCK_ENABLED and self.query_counter >= SHOCK_QUERY

        if task_type == 1:  # generation task
            if is_shock_active:
                # gpu degraded - cpu is now the better option
                latency = np.random.uniform(13.0, 17.0) if arm == 1 else np.random.uniform(8.0, 12.0)
            else:
                # normal: gpu is fast, cpu is slow for generation
                latency = np.random.uniform(0.4, 0.7) if arm == 1 else np.random.uniform(8.0, 12.0)
        else:  # retrieval task - similar on both nodes
            latency = np.random.uniform(0.1, 0.3) if arm == 0 else np.random.uniform(0.15, 0.35)

        reward = np.clip(1.0 - (latency / MAX_LATENCY), 0.01, 0.99)
        return reward, latency * 1000

    def get_real_reward(self) -> Optional[float]:
        """fetch reward from real rag-app metrics - not used in simulate mode"""
        try:
            response = requests.get(RAG_APP_METRICS_URL, timeout=5)
            response.raise_for_status()
            metrics = response.json()

            if metrics.get('history'):
                latest = metrics['history'][-1]
                latency_ms = latest.get('total_time_ms', 1000)
                return np.clip(1.0 - (latency_ms / 20000), 0.01, 0.99)

            return None

        except Exception as e:
            logger.warning(f"failed to fetch real metrics: {e}")
            return None

    def extract_context(self, pod) -> tuple[int, dict]:
        pod_name = pod.metadata.name.lower()
        labels = pod.metadata.labels or {}

        if any(x in pod_name for x in ['generation', 'ollama', 'llm', 'gpu']):
            task_type = 1
        elif labels.get('workload-type') == 'generation':
            task_type = 1
        else:
            task_type = 0

        context = {
            'pod_name': pod.metadata.name,
            'task_type': task_type,
            'task_type_label': 'generation' if task_type == 1 else 'retrieval',
            'query_number': self.query_counter
        }

        return task_type, context

    def schedule_pod(self, pod) -> bool:
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace

        self.query_counter += 1

        logger.info(f"scheduling query #{self.query_counter}: {namespace}/{pod_name}")

        if SHOCK_ENABLED and self.query_counter == SHOCK_QUERY:
            logger.warning(f"shock event at query {SHOCK_QUERY}")

        try:
            task_type, context = self.extract_context(pod)
            logger.info(f"context: {context}")

            arm = self.bandit.select_arm()
            target_node = NODE_MAPPING[arm]

            expected_rewards = self.bandit.get_expected_rewards()
            logger.info(f"expected rewards: cpu={expected_rewards[0]:.4f}, gpu={expected_rewards[1]:.4f}")
            logger.info(f"selected: {target_node} (arm {arm})")

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

            if SIMULATE_REWARDS:
                reward, latency_ms = self.simulate_reward(arm, task_type)
                logger.info(f"simulated latency: {latency_ms:.1f}ms, reward: {reward:.4f}")
            else:
                time.sleep(0.5)
                reward = self.get_real_reward()
                if reward is None:
                    reward = 0.5
                latency_ms = None
                logger.info(f"real reward: {reward:.4f}")

            update_result = self.bandit.update(arm, reward)

            if update_result['regime_change_detected']:
                logger.warning("regime change handled - posteriors reset")

            stats = self.bandit.get_statistics()
            logger.info(f"bandit stats: pulls={stats['total_pulls']}, avg_reward={stats['avg_reward']:.4f}, resets={stats['reset_count']}")

            decision = {
                'timestamp': datetime.utcnow().isoformat(),
                'query_number': self.query_counter,
                'pod_name': pod_name,
                'task_type': task_type,
                'arm': arm,
                'target_node': target_node,
                'reward': reward,
                'latency_ms': latency_ms,
                'shock_active': SHOCK_ENABLED and self.query_counter >= SHOCK_QUERY,
                'regime_change': update_result['regime_change_detected'],
                'bandit_stats': stats
            }
            self.decisions.append(decision)
            self._log_decision(decision)

            return True

        except ApiException as e:
            if e.status == 409:
                logger.warning(f"pod {pod_name} already bound")
            else:
                logger.error(f"api error: {e.status} {e.reason}")
            return False
        except Exception as e:
            logger.error(f"error scheduling {pod_name}: {e}", exc_info=True)
            return False

    def _log_decision(self, decision: dict):
        try:
            log_dir = Path(LOG_FILE).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(decision) + '\n')
        except Exception as e:
            logger.warning(f"failed to log decision: {e}")

    def run(self):
        logger.info(f"bandit scheduler started")
        logger.info(f"watching namespace: {NAMESPACE}, node mapping: {NODE_MAPPING}")
        logger.info(f"window size: {WINDOW_SIZE}, reset threshold: {RESET_THRESHOLD}")
        if SHOCK_ENABLED:
            logger.info(f"shock enabled at query {SHOCK_QUERY}")

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

            except ApiException as e:
                logger.error(f"api error: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"watch error: {e}", exc_info=True)
                logger.info("reconnecting in 5 seconds...")
                time.sleep(5)


def main():
    logger.info("starting bandit scheduler")

    try:
        scheduler = BanditScheduler()
        scheduler.run()
    except Exception as e:
        print(f"fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
