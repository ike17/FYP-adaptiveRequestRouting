#!/usr/bin/env python3
"""
ml-schedulers/bandit/bandit_scheduler.py

Adaptive Bandit Scheduler using Thompson Sampling with Regime Detection.

This scheduler demonstrates online learning for Kubernetes pod scheduling.
It adapts to changing cluster conditions without retraining.

Key Features:
    - Thompson Sampling for exploration-exploitation balance
    - Sliding window regime detection for piecewise-stationary environments
    - Automatic reset when performance degradation is detected
    - Simulated rewards with shock injection for evaluation

Algorithm: Detection-Augmented Bandit (DAB)
    - Reference: Besson et al., 2022 (Piecewise-Stationary Bandits)

IMPORTANT: This is a PROTOTYPE. Rewards are simulated, not measured from real latency.
In production, you would query the /metrics endpoint of rag-app for actual latency.
"""

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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Scheduler configuration
SCHEDULER_NAME = os.getenv('SCHEDULER_NAME', 'bandit-scheduler')
NAMESPACE = os.getenv('WATCH_NAMESPACE', 'default')

# Node mapping
NODE_MAPPING = {
    0: os.getenv('CPU_NODE_NAME', 'minikube'),
    1: os.getenv('GPU_NODE_NAME', 'minikube-m02')
}

# Thompson Sampling parameters
INITIAL_ALPHA = 1.0  # Prior successes
INITIAL_BETA = 1.0   # Prior failures

# Regime detection parameters
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '20'))
RESET_THRESHOLD = float(os.getenv('RESET_THRESHOLD', '0.4'))  # Reset if reward drops to 40% of baseline

# Shock simulation parameters (for evaluation)
SHOCK_ENABLED = os.getenv('SHOCK_ENABLED', 'true').lower() == 'true'
SHOCK_QUERY = int(os.getenv('SHOCK_QUERY', '50'))  # Shock occurs at this query
SIMULATE_REWARDS = os.getenv('SIMULATE_REWARDS', 'true').lower() == 'true'

# Real metrics endpoint (when not simulating)
RAG_APP_METRICS_URL = os.getenv('RAG_APP_METRICS_URL', 'http://rag-app-service:8000/metrics')

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/app/logs/bandit_decisions.jsonl')

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('bandit-scheduler')


class ThompsonSamplingBandit:
    """
    Thompson Sampling Multi-Armed Bandit with Regime Detection.
    
    This implements a Beta-Bernoulli bandit where:
    - Each arm maintains a Beta(α, β) posterior distribution
    - We sample from each posterior and select the arm with highest sample
    - We detect regime changes via sliding window reward monitoring
    - On regime change, we reset to uniform priors to re-explore
    
    This is the core algorithm for adaptive scheduling.
    """
    
    def __init__(self, n_arms: int = 2, window_size: int = 20, 
                 reset_threshold: float = 0.4):
        """
        Initialize Thompson Sampling bandit.
        
        Args:
            n_arms: Number of arms (nodes to choose from)
            window_size: Sliding window size for regime detection
            reset_threshold: Fraction of baseline reward below which we reset
        """
        self.n_arms = n_arms
        self.window_size = window_size
        self.reset_threshold = reset_threshold
        
        # Beta distribution parameters for each arm
        # Beta(α, β) where:
        # - α = 1 + sum of rewards (successes)
        # - β = 1 + sum of (1 - reward) (failures)
        self.alpha = np.array([INITIAL_ALPHA] * n_arms)
        self.beta = np.array([INITIAL_BETA] * n_arms)
        
        # Sliding window for regime detection
        self.reward_window = deque(maxlen=window_size)
        self.baseline_reward: Optional[float] = None
        self.last_baseline_update = 0
        
        # Statistics
        self.total_pulls = 0
        self.arm_pulls = np.zeros(n_arms, dtype=int)
        self.total_reward = 0.0
        self.arm_rewards = np.zeros(n_arms)
        self.reset_count = 0
        
        logger.info(f"Initialized Thompson Sampling with {n_arms} arms")
        logger.info(f"Window size: {window_size}, Reset threshold: {reset_threshold}")
    
    def select_arm(self) -> int:
        """
        Select an arm using Thompson Sampling.
        
        Algorithm:
        1. For each arm, sample θ ~ Beta(α, β)
        2. Select arm with highest θ
        
        Returns:
            Index of selected arm
        """
        # Sample from Beta posterior for each arm
        samples = np.array([
            np.random.beta(self.alpha[a], self.beta[a])
            for a in range(self.n_arms)
        ])
        
        selected_arm = np.argmax(samples)
        
        logger.debug(f"Thompson samples: {samples}, Selected arm: {selected_arm}")
        
        return int(selected_arm)
    
    def update(self, arm: int, reward: float) -> dict:
        """
        Update posterior after observing reward and check for regime change.
        
        The reward is treated as a probability of success:
        - We update α += reward (expected successes)
        - We update β += (1 - reward) (expected failures)
        
        Args:
            arm: Selected arm
            reward: Observed reward in [0, 1]
        
        Returns:
            Dict with update info including whether regime change was detected
        """
        # Clip reward to [0, 1]
        reward = np.clip(reward, 0.0, 1.0)
        
        # Update Beta parameters
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
        
        # Update statistics
        self.total_pulls += 1
        self.arm_pulls[arm] += 1
        self.total_reward += reward
        self.arm_rewards[arm] += reward
        
        # Track reward for regime detection
        self.reward_window.append(reward)
        
        # Check for regime change
        regime_change = self._check_regime_change()
        
        return {
            'arm': arm,
            'reward': reward,
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'regime_change_detected': regime_change
        }
    
    def _check_regime_change(self) -> bool:
        """
        Detect regime change using sliding window analysis.
        
        If average reward in recent window drops significantly below
        the established baseline, we declare a regime change and reset.
        
        Returns:
            True if regime change was detected and reset performed
        """
        if len(self.reward_window) < self.window_size:
            return False
        
        current_avg = np.mean(self.reward_window)
        
        # Establish baseline from first full window
        if self.baseline_reward is None:
            self.baseline_reward = current_avg
            self.last_baseline_update = self.total_pulls
            logger.info(f"Baseline reward established: {self.baseline_reward:.4f}")
            return False
        
        # Check for significant drop
        if current_avg < self.baseline_reward * self.reset_threshold:
            logger.warning(f"\n{'!'*60}")
            logger.warning("REGIME CHANGE DETECTED!")
            logger.warning(f"Current avg reward: {current_avg:.4f}")
            logger.warning(f"Baseline reward: {self.baseline_reward:.4f}")
            logger.warning(f"Threshold: {self.baseline_reward * self.reset_threshold:.4f}")
            logger.warning("Resetting posteriors to uniform priors...")
            logger.warning(f"{'!'*60}\n")
            
            self._reset()
            return True
        
        # Slowly update baseline if stable (adaptive baseline)
        if self.total_pulls - self.last_baseline_update > self.window_size * 2:
            # Weighted update: 90% old baseline, 10% current
            self.baseline_reward = 0.9 * self.baseline_reward + 0.1 * current_avg
            self.last_baseline_update = self.total_pulls
        
        return False
    
    def _reset(self):
        """Reset to uniform priors after regime change."""
        self.alpha = np.array([INITIAL_ALPHA] * self.n_arms)
        self.beta = np.array([INITIAL_BETA] * self.n_arms)
        self.reward_window.clear()
        self.baseline_reward = None
        self.reset_count += 1
        
        logger.info("Bandit reset complete. Ready to explore new regime.")
    
    def get_statistics(self) -> dict:
        """Return current bandit statistics."""
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
        """Get expected reward (posterior mean) for each arm."""
        return self.alpha / (self.alpha + self.beta)


class BanditScheduler:
    """
    Kubernetes scheduler using Thompson Sampling bandit.
    
    This scheduler learns optimal node placement online by:
    1. Selecting nodes using Thompson Sampling (exploration-exploitation)
    2. Observing rewards (inverse latency) from actual performance
    3. Updating posteriors based on observed rewards
    4. Detecting regime changes and adapting
    """
    
    def __init__(self):
        """Initialize Kubernetes client and bandit algorithm."""
        logger.info("Initializing Bandit Scheduler...")
        
        # Initialize Kubernetes
        self._init_kubernetes()
        
        # Initialize bandit algorithm
        self.bandit = ThompsonSamplingBandit(
            n_arms=2,
            window_size=WINDOW_SIZE,
            reset_threshold=RESET_THRESHOLD
        )
        
        # Query counter for shock simulation
        self.query_counter = 0
        
        # Scheduling decisions log
        self.decisions = []
        
        logger.info("Bandit Scheduler initialized successfully")
        logger.info(f"Shock simulation: {'ENABLED' if SHOCK_ENABLED else 'DISABLED'}")
        if SHOCK_ENABLED:
            logger.info(f"Shock will occur at query {SHOCK_QUERY}")
    
    def _init_kubernetes(self):
        """Initialize Kubernetes client."""
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("Loaded local Kubernetes configuration")
            except config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise
        
        self.v1 = client.CoreV1Api()
    
    def simulate_reward(self, arm: int, task_type: int) -> tuple[float, float]:
        """
        Simulate reward based on placement quality.
        
        This is the PROTOTYPE reward function that simulates latency.
        It includes SHOCK INJECTION at query SHOCK_QUERY.
        
        In production, replace this with real metrics from rag-app.
        
        Reward = 1 / (latency / max_latency)
        This normalizes reward to approximately [0, 1]
        
        Args:
            arm: Selected arm (0=CPU, 1=GPU)
            task_type: Type of task (0=retrieval, 1=generation)
        
        Returns:
            Tuple of (reward, simulated_latency_ms)
        """
        # Maximum expected latency for normalization
        MAX_LATENCY = 20.0  # seconds
        
        is_shock_active = SHOCK_ENABLED and self.query_counter >= SHOCK_QUERY
        
        if task_type == 1:  # Generation task
            if is_shock_active:
                # SHOCK SCENARIO: GPU is degraded
                if arm == 1:  # GPU node
                    # GPU is now SLOW (simulates thermal throttling, noisy neighbor, etc.)
                    latency = np.random.uniform(13.0, 17.0)
                else:  # CPU node
                    # CPU remains at baseline (now relatively better!)
                    latency = np.random.uniform(8.0, 12.0)
            else:
                # NORMAL SCENARIO
                if arm == 1:  # GPU node (correct placement)
                    latency = np.random.uniform(0.4, 0.7)  # Fast
                else:  # CPU node (suboptimal)
                    latency = np.random.uniform(8.0, 12.0)  # Slow
        else:  # Retrieval task
            # Retrieval performance is similar on both nodes
            if arm == 0:  # CPU node (correct placement)
                latency = np.random.uniform(0.1, 0.3)
            else:  # GPU node (works but wastes GPU)
                latency = np.random.uniform(0.15, 0.35)
        
        # Calculate reward (inverse latency, normalized)
        reward = 1.0 - (latency / MAX_LATENCY)
        reward = np.clip(reward, 0.01, 0.99)  # Avoid extreme values
        
        latency_ms = latency * 1000
        
        return reward, latency_ms
    
    def get_real_reward(self) -> Optional[float]:
        """
        Fetch real reward from rag-app metrics endpoint.
        
        NOT USED IN PROTOTYPE - for production implementation.
        
        Returns:
            Reward based on actual latency, or None if unavailable
        """
        try:
            response = requests.get(RAG_APP_METRICS_URL, timeout=5)
            response.raise_for_status()
            metrics = response.json()
            
            if metrics.get('history'):
                # Get most recent latency
                latest = metrics['history'][-1]
                latency_ms = latest.get('total_time_ms', 1000)
                
                # Convert to reward (inverse, normalized)
                MAX_LATENCY_MS = 20000
                reward = 1.0 - (latency_ms / MAX_LATENCY_MS)
                return np.clip(reward, 0.01, 0.99)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch real metrics: {e}")
            return None
    
    def extract_context(self, pod) -> tuple[int, dict]:
        """
        Extract context from pod for decision making.
        
        Args:
            pod: Kubernetes pod object
            
        Returns:
            Tuple of (task_type, context_dict)
        """
        pod_name = pod.metadata.name.lower()
        labels = pod.metadata.labels or {}
        
        # Determine task type
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
        """
        Schedule a pod using Thompson Sampling.
        
        Args:
            pod: Kubernetes pod object
            
        Returns:
            True if successfully scheduled
        """
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace
        
        self.query_counter += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SCHEDULING [Query #{self.query_counter}]: {namespace}/{pod_name}")
        if SHOCK_ENABLED and self.query_counter == SHOCK_QUERY:
            logger.warning("⚡ SHOCK EVENT OCCURRING NOW ⚡")
        logger.info(f"{'='*60}")
        
        try:
            # Extract context
            task_type, context = self.extract_context(pod)
            logger.info(f"Context: {context}")
            
            # Bandit selects arm
            arm = self.bandit.select_arm()
            target_node = NODE_MAPPING[arm]
            
            expected_rewards = self.bandit.get_expected_rewards()
            logger.info(f"Expected rewards: CPU={expected_rewards[0]:.4f}, GPU={expected_rewards[1]:.4f}")
            logger.info(f"Selected: {target_node} (arm {arm})")
            
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
            
            logger.info(f"✓ Bound {pod_name} to {target_node}")
            
            # Get reward (simulated or real)
            if SIMULATE_REWARDS:
                reward, latency_ms = self.simulate_reward(arm, task_type)
                logger.info(f"Simulated latency: {latency_ms:.1f}ms, Reward: {reward:.4f}")
            else:
                # Wait briefly for pod to run, then get real metrics
                time.sleep(0.5)
                reward = self.get_real_reward()
                if reward is None:
                    reward = 0.5  # Neutral reward if metrics unavailable
                latency_ms = None
                logger.info(f"Real reward: {reward:.4f}")
            
            # Update bandit
            update_result = self.bandit.update(arm, reward)
            
            if update_result['regime_change_detected']:
                logger.warning("Regime change handled - posteriors reset")
            
            # Log statistics
            stats = self.bandit.get_statistics()
            logger.info(f"Bandit stats: pulls={stats['total_pulls']}, "
                       f"avg_reward={stats['avg_reward']:.4f}, "
                       f"resets={stats['reset_count']}")
            
            # Record decision
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
                logger.warning(f"Pod {pod_name} already bound")
            else:
                logger.error(f"API error: {e.status} {e.reason}")
            return False
        except Exception as e:
            logger.error(f"Error scheduling {pod_name}: {e}", exc_info=True)
            return False
    
    def _log_decision(self, decision: dict):
        """Log decision to JSONL file."""
        try:
            log_dir = Path(LOG_FILE).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(decision) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")
    
    def run(self):
        """Main scheduler loop."""
        logger.info("\n" + "=" * 60)
        logger.info("ADAPTIVE BANDIT SCHEDULER STARTED")
        logger.info(f"Algorithm: Thompson Sampling with Regime Detection")
        logger.info(f"Scheduler name: {SCHEDULER_NAME}")
        logger.info(f"Watching namespace: {NAMESPACE}")
        logger.info(f"Node mapping: {NODE_MAPPING}")
        logger.info(f"Window size: {WINDOW_SIZE}, Reset threshold: {RESET_THRESHOLD}")
        if SHOCK_ENABLED:
            logger.info(f"⚡ SHOCK MODE: Enabled at query {SHOCK_QUERY}")
        logger.info("=" * 60 + "\n")
        
        w = watch.Watch()
        
        while True:
            try:
                logger.info("Watching for pending pods...")
                
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
                logger.error(f"API error: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Watch error: {e}", exc_info=True)
                logger.info("Reconnecting in 5 seconds...")
                time.sleep(5)


def main():
    """Entry point."""
    print("=" * 60)
    print("ADAPTIVE BANDIT SCHEDULER")
    print("Thompson Sampling with Regime Detection")
    print("=" * 60)
    
    try:
        scheduler = BanditScheduler()
        scheduler.run()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
