#!/usr/bin/env python3
# adaptive_bandit_scheduler.py
# extends the basic bandit scheduler with real metrics and active pod eviction
#
# the key difference from bandit_scheduler.py is that this one:
# - polls the rag-app /metrics endpoint to get real latency instead of simulating it
# - evicts running pods when a regime change is detected so they get rescheduled
#   with the updated bandit beliefs
# - runs two loops concurrently: one for scheduling, one for monitoring

import json
import logging
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

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
NODE_TO_ARM = {v: k for k, v in NODE_MAPPING.items()}

INITIAL_ALPHA = float(os.getenv('INITIAL_ALPHA', '1.0'))
INITIAL_BETA = float(os.getenv('INITIAL_BETA', '1.0'))

WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '20'))
RESET_THRESHOLD = float(os.getenv('RESET_THRESHOLD', '0.4'))

METRICS_POLL_INTERVAL = float(os.getenv('METRICS_POLL_INTERVAL', '2.0'))
RAG_APP_METRICS_URL = os.getenv('RAG_APP_METRICS_URL', 'http://rag-app-service:8000/metrics')

# when running on cpu-only (no gpu), set SIMULATE_REWARDS=true so the shock still works
SIMULATE_REWARDS = os.getenv('SIMULATE_REWARDS', 'false').lower() == 'true'
SHOCK_ENABLED = os.getenv('SHOCK_ENABLED', 'true').lower() == 'true'
SHOCK_QUERY = int(os.getenv('SHOCK_QUERY', '50'))

EVICTION_ENABLED = os.getenv('EVICTION_ENABLED', 'true').lower() == 'true'
EVICTION_COOLDOWN = float(os.getenv('EVICTION_COOLDOWN', '30.0'))
MIN_QUERIES_BEFORE_EVICTION = int(os.getenv('MIN_QUERIES_BEFORE_EVICTION', '10'))
EVICTION_TARGET_LABELS = os.getenv('EVICTION_TARGET_LABELS', 'app=generation')

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/app/logs/adaptive_bandit.jsonl')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('adaptive-bandit-scheduler')


class ThompsonSamplingBandit:
    """same beta-bernoulli bandit as the basic version, just fed with real metrics"""

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

        self.last_arm: Optional[int] = None

        logger.info(f"bandit initialised: {n_arms} arms, window={window_size}, threshold={reset_threshold}")

    def select_arm(self) -> int:
        samples = np.array([
            np.random.beta(self.alpha[a], self.beta[a])
            for a in range(self.n_arms)
        ])

        selected = int(np.argmax(samples))
        self.last_arm = selected
        logger.debug(f"thompson samples: {samples}, selected arm {selected}")
        return selected

    def update(self, arm: int, reward: float) -> Dict:
        reward = np.clip(reward, 0.0, 1.0)

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
            'regime_change_detected': regime_change,
            'window_avg': float(np.mean(self.reward_window)) if self.reward_window else 0.0,
            'baseline': self.baseline_reward
        }

    def _check_regime_change(self) -> bool:
        if len(self.reward_window) < self.window_size:
            return False

        current_avg = np.mean(self.reward_window)

        if self.baseline_reward is None:
            self.baseline_reward = current_avg
            self.last_baseline_update = self.total_pulls
            logger.info(f"baseline established: {self.baseline_reward:.4f}")
            return False

        threshold = self.baseline_reward * self.reset_threshold
        if current_avg < threshold:
            logger.warning(f"regime change detected - current avg: {current_avg:.4f}, baseline: {self.baseline_reward:.4f}, threshold: {threshold:.4f}")
            return True

        # slowly drift the baseline to track gradual improvement
        if self.total_pulls - self.last_baseline_update > self.window_size * 2:
            self.baseline_reward = 0.9 * self.baseline_reward + 0.1 * current_avg
            self.last_baseline_update = self.total_pulls

        return False

    def reset(self):
        """reset to uniform priors after regime change"""
        logger.info("resetting bandit to uniform priors")
        self.alpha = np.array([INITIAL_ALPHA] * self.n_arms)
        self.beta = np.array([INITIAL_BETA] * self.n_arms)
        self.reward_window.clear()
        self.baseline_reward = None
        self.reset_count += 1

    def get_expected_rewards(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def get_statistics(self) -> Dict:
        return {
            'total_pulls': self.total_pulls,
            'arm_pulls': self.arm_pulls.tolist(),
            'avg_reward': self.total_reward / max(self.total_pulls, 1),
            'expected_rewards': self.get_expected_rewards().tolist(),
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'reset_count': self.reset_count,
            'baseline': self.baseline_reward,
            'window_size': len(self.reward_window)
        }


class MetricsCollector:
    """polls the rag-app /metrics endpoint and converts latency to reward"""

    def __init__(self, metrics_url: str):
        self.metrics_url = metrics_url
        self.last_query_id = 0
        self.query_count = 0

    def get_latest_metrics(self) -> Optional[Dict]:
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"failed to fetch metrics: {e}")
            return None

    def get_new_queries(self) -> List[Dict]:
        """return only queries we haven't seen yet"""
        metrics = self.get_latest_metrics()
        if not metrics or 'history' not in metrics:
            return []

        history = metrics['history']
        new_queries = []

        for entry in history:
            query_id = entry.get('query_id', 0)
            if query_id > self.last_query_id:
                new_queries.append(entry)
                self.last_query_id = query_id
                self.query_count += 1

        return new_queries

    def latency_to_reward(self, latency_ms: float, max_latency_ms: float = 20000) -> float:
        reward = 1.0 - (latency_ms / max_latency_ms)
        return float(np.clip(reward, 0.01, 0.99))


class PodEvictor:
    """deletes running pods so the deployment controller recreates them at better nodes"""

    def __init__(self, v1_client: client.CoreV1Api, namespace: str):
        self.v1 = v1_client
        self.namespace = namespace
        self.last_eviction_time = 0
        self.eviction_count = 0

    def can_evict(self) -> bool:
        return (time.time() - self.last_eviction_time) >= EVICTION_COOLDOWN

    def evict_pods(self, label_selector: str) -> List[str]:
        if not self.can_evict():
            remaining = EVICTION_COOLDOWN - (time.time() - self.last_eviction_time)
            logger.info(f"eviction on cooldown, {remaining:.1f}s remaining")
            return []

        evicted = []

        try:
            pods = self.v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector
            )

            for pod in pods.items:
                if pod.status.phase == "Running":
                    pod_name = pod.metadata.name
                    node_name = pod.spec.node_name

                    logger.warning(f"evicting pod {pod_name} from {node_name}")

                    self.v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=self.namespace,
                        grace_period_seconds=5
                    )

                    evicted.append(pod_name)
                    self.eviction_count += 1

            if evicted:
                self.last_eviction_time = time.time()
                logger.info(f"evicted {len(evicted)} pods: {evicted}")

        except ApiException as e:
            logger.error(f"error evicting pods: {e}")

        return evicted


class AdaptiveBanditScheduler:
    """
    combines thompson sampling scheduling with real-time metrics monitoring.
    when a regime change is detected it resets the bandit AND evicts the
    affected pods so they get rescheduled with the updated beliefs.
    """

    def __init__(self):
        logger.info("initialising adaptive bandit scheduler...")

        self._init_kubernetes()

        self.bandit = ThompsonSamplingBandit(
            n_arms=2,
            window_size=WINDOW_SIZE,
            reset_threshold=RESET_THRESHOLD
        )
        self.metrics = MetricsCollector(RAG_APP_METRICS_URL)
        self.evictor = PodEvictor(self.v1, NAMESPACE)

        self.scheduled_pods: Dict[str, int] = {}
        self.query_counter = 0
        self.decisions = []
        self.running = True

        logger.info(f"eviction enabled: {EVICTION_ENABLED}")
        logger.info(f"simulate rewards: {SIMULATE_REWARDS}")
        if SIMULATE_REWARDS and SHOCK_ENABLED:
            logger.info(f"shock enabled at query {SHOCK_QUERY}")
        logger.info(f"metrics url: {RAG_APP_METRICS_URL}")

    def _init_kubernetes(self):
        try:
            config.load_incluster_config()
            logger.info("loaded in-cluster config")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("loaded local kubeconfig")
            except config.ConfigException as e:
                logger.error(f"failed to load config: {e}")
                raise

        self.v1 = client.CoreV1Api()

    def get_current_arm(self) -> Optional[int]:
        """figure out which node is currently running the generation service"""
        try:
            pods = self.v1.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=EVICTION_TARGET_LABELS
            )

            for pod in pods.items:
                if pod.status.phase == "Running" and pod.spec.node_name:
                    return NODE_TO_ARM.get(pod.spec.node_name)

            return None

        except ApiException as e:
            logger.error(f"error getting current arm: {e}")
            return None

    def schedule_pod(self, pod) -> bool:
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace

        logger.info(f"scheduling {namespace}/{pod_name}")

        try:
            arm = self.bandit.select_arm()
            target_node = NODE_MAPPING[arm]

            expected = self.bandit.get_expected_rewards()
            logger.info(f"expected rewards: cpu={expected[0]:.3f}, gpu={expected[1]:.3f}")
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

            self.scheduled_pods[pod_name] = arm
            logger.info(f"bound {pod_name} to {target_node}")

            self._log_decision({
                'event': 'schedule',
                'pod_name': pod_name,
                'arm': arm,
                'node': target_node,
                'expected_rewards': expected.tolist(),
                'stats': self.bandit.get_statistics()
            })

            return True

        except ApiException as e:
            if e.status == 409:
                logger.warning(f"pod {pod_name} already bound")
            else:
                logger.error(f"binding failed: {e}")
            return False

    def simulate_reward(self, arm: int, task_type: int = 1) -> tuple:
        """simulate reward with shock injection - mirrors bandit_scheduler.py"""
        MAX_LATENCY = 20.0

        is_shock_active = SHOCK_ENABLED and self.query_counter >= SHOCK_QUERY

        if task_type == 1:
            if is_shock_active:
                latency = np.random.uniform(13.0, 17.0) if arm == 1 else np.random.uniform(8.0, 12.0)
            else:
                latency = np.random.uniform(0.4, 0.7) if arm == 1 else np.random.uniform(8.0, 12.0)
        else:
            latency = np.random.uniform(0.1, 0.3) if arm == 0 else np.random.uniform(0.15, 0.35)

        reward = np.clip(1.0 - (latency / MAX_LATENCY), 0.01, 0.99)
        return reward, latency * 1000

    def process_metrics(self):
        """update bandit with either real or simulated rewards, return true if regime changed"""
        if SIMULATE_REWARDS:
            return self._process_simulated_metrics()

        new_queries = self.metrics.get_new_queries()
        if not new_queries:
            return False

        current_arm = self.get_current_arm()
        if current_arm is None:
            logger.warning("cannot determine current arm, skipping update")
            return False

        regime_change = False

        for query in new_queries:
            latency_ms = query.get('total_time_ms', 1000)
            reward = self.metrics.latency_to_reward(latency_ms)

            result = self.bandit.update(current_arm, reward)

            logger.info(f"query {query.get('query_id')}: latency={latency_ms:.0f}ms, reward={reward:.3f}, arm={current_arm} ({NODE_MAPPING[current_arm]})")

            if result['regime_change_detected']:
                regime_change = True

                self._log_decision({
                    'event': 'regime_change',
                    'query_id': query.get('query_id'),
                    'latency_ms': latency_ms,
                    'reward': reward,
                    'window_avg': result['window_avg'],
                    'baseline': result['baseline'],
                    'stats': self.bandit.get_statistics()
                })

        return regime_change

    def _process_simulated_metrics(self) -> bool:
        """simulated version of process_metrics - used when SIMULATE_REWARDS=true"""
        self.query_counter += 1

        if SHOCK_ENABLED and self.query_counter == SHOCK_QUERY:
            logger.warning(f"shock event at query {SHOCK_QUERY}")

        selected_arm = self.bandit.select_arm()
        reward, latency_ms = self.simulate_reward(selected_arm, task_type=1)

        shock_active = SHOCK_ENABLED and self.query_counter >= SHOCK_QUERY
        logger.info(f"[simulated] query #{self.query_counter}: latency={latency_ms:.0f}ms, reward={reward:.3f}, arm={selected_arm} ({NODE_MAPPING[selected_arm]}){' [shock]' if shock_active else ''}")

        result = self.bandit.update(selected_arm, reward)
        regime_change = result['regime_change_detected']

        if regime_change:
            logger.warning(f"regime change at query #{self.query_counter}, window avg: {result['window_avg']:.3f}, baseline: {result['baseline']:.3f}")
            self._log_decision({
                'event': 'regime_change',
                'query_number': self.query_counter,
                'latency_ms': latency_ms,
                'reward': reward,
                'shock_active': shock_active,
                'window_avg': result['window_avg'],
                'baseline': result['baseline'],
                'stats': self.bandit.get_statistics()
            })

        if self.query_counter % 10 == 0:
            stats = self.bandit.get_statistics()
            logger.info(f"stats q#{self.query_counter}: pulls={stats['total_pulls']}, avg_reward={stats['avg_reward']:.3f}, resets={stats['reset_count']}")

        return regime_change

    def handle_regime_change(self):
        """reset bandit and optionally evict pods so they get rescheduled"""
        logger.warning("handling regime change")

        self.bandit.reset()

        if not EVICTION_ENABLED:
            logger.info("eviction disabled, skipping")
            return

        query_count = self.query_counter if SIMULATE_REWARDS else self.metrics.query_count
        if query_count < MIN_QUERIES_BEFORE_EVICTION:
            logger.info(f"only {query_count} queries so far, waiting for {MIN_QUERIES_BEFORE_EVICTION} before evicting")
            return

        evicted = self.evictor.evict_pods(EVICTION_TARGET_LABELS)

        if evicted:
            self._log_decision({
                'event': 'eviction',
                'evicted_pods': evicted,
                'reason': 'regime_change',
                'stats': self.bandit.get_statistics()
            })
            logger.info("evicted pods will be rescheduled with updated beliefs")
        else:
            logger.warning("no pods evicted (cooldown active or no running pods)")

    def _log_decision(self, data: Dict):
        data['timestamp'] = datetime.utcnow().isoformat()
        self.decisions.append(data)

        try:
            log_dir = Path(LOG_FILE).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.warning(f"failed to write log: {e}")

    def monitoring_loop(self):
        """background thread - polls metrics and handles regime changes"""
        logger.info("metrics monitoring loop started")

        while self.running:
            try:
                regime_change = self.process_metrics()

                if regime_change:
                    self.handle_regime_change()

            except Exception as e:
                logger.error(f"monitoring error: {e}")

            time.sleep(METRICS_POLL_INTERVAL)

        logger.info("monitoring loop stopped")

    def scheduling_loop(self):
        """main thread - watches for pending pods and schedules them"""
        logger.info("scheduling loop started")

        w = watch.Watch()

        while self.running:
            try:
                for event in w.stream(
                    self.v1.list_namespaced_pod,
                    namespace=NAMESPACE,
                    timeout_seconds=60
                ):
                    if not self.running:
                        break

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
                logger.error(f"watch error: {e}")
                time.sleep(5)

        logger.info("scheduling loop stopped")

    def run(self):
        logger.info("adaptive bandit scheduler started")
        logger.info(f"watching namespace: {NAMESPACE}, node mapping: {NODE_MAPPING}")
        logger.info(f"eviction enabled: {EVICTION_ENABLED}, eviction target: {EVICTION_TARGET_LABELS}")
        logger.info(f"simulate rewards: {SIMULATE_REWARDS}")
        if SIMULATE_REWARDS and SHOCK_ENABLED:
            logger.info(f"shock enabled at query {SHOCK_QUERY}")

        monitor_thread = threading.Thread(
            target=self.monitoring_loop,
            daemon=True,
            name="metrics-monitor"
        )
        monitor_thread.start()

        try:
            self.scheduling_loop()
        except KeyboardInterrupt:
            logger.info("shutting down...")
            self.running = False

        monitor_thread.join(timeout=5)
        logger.info("scheduler stopped")


def main():
    logger.info("starting adaptive bandit scheduler")

    try:
        scheduler = AdaptiveBanditScheduler()
        scheduler.run()
    except Exception as e:
        print(f"fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
