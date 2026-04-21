# Troubleshooting

Common issues for the L7 Smart Gateway on bare-metal k3s.

## 1) Pods stuck in `Pending`

```bash
kubectl describe pod <pod-name>
```

Check the Events section. Common causes:
- Missing node labels. Fix with `./scripts/setup_cluster.sh` or apply labels manually.
- GPU resource not registered. Re-apply the NVIDIA device plugin from `scripts/setup_cluster.sh`.
- `DiskPressure` on the CPU node. Free disk space, then allow the node to schedule again.

## 2) `ErrImageNeverPull`

k3s nodes do not share the host Docker daemon. After rebuilding images, re-import them:

```bash
GPU_NODE=<gpu-ip> ./scripts/build_all.sh
```

## 3) Static mode fails at startup

`static` requires the trained Random Forest model at `/app/model/gateway_model.pkl`.

If the gateway crashes in static mode:

```bash
cd ml-training
python generate_dataset.py
python train_model.py
cd ..
GPU_NODE=<gpu-ip> ./scripts/build_all.sh
```

The submission branch also ships a tracked copy at `smart-gateway/model/gateway_model.pkl`
so a clean checkout remains runnable.

## 4) Gateway is healthy but `/ready` returns `503`

`/health` only checks the gateway process. `/ready` verifies that both Ollama backends
are reachable and is intended for diagnostics.

Check:

```bash
kubectl get pods -o wide
kubectl logs deployment/smart-gateway
curl http://<gateway-host>:32367/ready
```

Common causes:
- Ollama pod restarting while the model is still loading.
- Pod eviction caused by low disk space.
- Incorrect service names or a backend deployment not running.

## 5) First query is very slow

Expected after a fresh rollout. Ollama loads `gemma:2b` lazily, and the CPU/GPU pods
need warming before latency stabilises.

Manual checks:

```bash
kubectl exec <ollama-pod> -- ollama list
```

`run_all.sh` already handles model wait plus CPU/GPU warmup before each experiment.

## 6) Gateway metrics are empty

The metrics endpoint only reports data after at least one request.

Clear metrics between runs:

```bash
curl -X DELETE http://<gateway-host>:32367/metrics
```

## 7) Port-forward drops during benchmarking

Use the NodePort directly when possible:

```bash
python3 benchmarks/run_experiments.py --config bandit_plain --url http://<cpu-node-ip>:32367
```

If you must use a port-forward, the experiment runner can restart it automatically.

## 8) `DiskPressure` on the CPU node

```bash
kubectl describe node <cpu-node> | grep DiskPressure
```

If the condition is `True`, free space on the node:

```bash
sudo journalctl --vacuum-size=100M
docker system prune -af
sudo k3s crictl rmi --prune
```

Then restart affected workloads:

```bash
kubectl rollout restart deployment/ollama-cpu deployment/smart-gateway
```

## 9) Clean reset between experiments

`run_all.sh` already rotates through all six modes with fresh rollouts. For manual resets:

```bash
kubectl set env deployment/smart-gateway ROUTING_MODE=bandit_plain
kubectl rollout restart deployment/ollama-gpu deployment/ollama-cpu deployment/smart-gateway
```

## 10) Re-run the final aggregated analysis

If `results_final/` needs regeneration, run the multi-run analysis over the collected results
directory rather than editing the CSV or JSON files by hand:

```bash
python3 benchmarks/analyze_multiple.py --input <results-dir> --output results_final
```
