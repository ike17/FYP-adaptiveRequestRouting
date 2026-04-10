# Troubleshooting

Common issues for the L7 Smart Gateway on bare-metal k3s.

## 1) Pods stuck in Pending

```bash
kubectl describe pod <pod-name>
```

Check Events section. Common causes:
- Missing node labels. Fix: `./scripts/setup_cluster.sh` or manually:
  ```bash
  kubectl label node <cpu-node> node_type=cpu_standard --overwrite
  kubectl label node <gpu-node> node_type=gpu_accelerated --overwrite
  ```
- GPU resource not registered. Fix: reapply NVIDIA device plugin (see setup_cluster.sh).
- DiskPressure on CPU node. Fix: free disk space, then `kubectl uncordon <node>`.

## 2) ErrImageNeverPull

k3s nodes don't share the host Docker daemon. After any `docker build`, reimport:

```bash
GPU_NODE=<gpu-ip> ./scripts/build_all.sh
```

## 3) First query very slow (60-100s)

Expected. The postStart lifecycle hook pulls gemma:2b on first pod startup. `run_all.sh` handles this via `wait_for_model` + `warm_gpu_vram`. If running manually:

```bash
# Check if model is loaded
kubectl exec <ollama-pod> -- ollama list
```

## 4) Gateway returns 503 on /ready

One or both Ollama nodes unreachable. Check:

```bash
kubectl get pods -o wide
kubectl logs deployment/smart-gateway
```

Common causes:
- Ollama pod restarting (model pull in progress). Wait.
- Pod evicted due to DiskPressure. Fix: free disk, restart deployment.
- Wrong service name. Verify: `kubectl get svc` should show `ollama-gpu-service` and `ollama-cpu-service`.

## 5) Adaptive mode not reacting to shock

Check bandit stats during experiment:

```bash
curl $GATEWAY_URL/bandit/stats
```

Look for:
- `reset_count > 0` — regime change was detected
- `recovery_remaining` — non-zero means circuit-breaker is active
- If `reset_count == 0` after shock: detection didn't fire. Possible causes: window not filled (too few GPU queries during detection window), baseline drifted low.

## 6) Metrics endpoint returns empty

Send at least one query before checking. Clear between runs:

```bash
curl -X DELETE $GATEWAY_URL/metrics
```

## 7) Port-forward drops during benchmark

Use `--port-forward` flag with run_experiments.py (auto-restarts dead tunnels):

```bash
python benchmarks/run_experiments.py --config bandit --port-forward
```

Or use the NodePort directly (no port-forward needed):

```bash
python benchmarks/run_experiments.py --config bandit --url http://<cpu-node-ip>:32367
```

## 8) DiskPressure on CPU node (ProDesk)

```bash
kubectl describe node <cpu-node> | grep DiskPressure
```

If True: pods get evicted. Fix:

```bash
# SSH to CPU node
ssh prox@192.168.0.20
# Free space
sudo journalctl --vacuum-size=100M
docker system prune -af
sudo k3s crictl rmi --prune
```

After freeing space, k3s should automatically clear the condition. If pods don't reschedule:

```bash
kubectl rollout restart deployment/ollama-cpu deployment/smart-gateway
```

## 9) Clean reset between experiments

`run_all.sh` does this automatically. For manual reset:

```bash
kubectl set env deployment/smart-gateway ROUTING_MODE=bandit
kubectl rollout restart deployment/ollama-gpu deployment/ollama-cpu deployment/smart-gateway
```

To delete everything:

```bash
kubectl delete -f k8s/03-gateway-app.yaml
kubectl delete -f k8s/02-ollama-cpu.yaml
kubectl delete -f k8s/01-ollama-gpu.yaml
```

## 10) run_multiple.sh fails repeatedly

Check `results/multi_*/failure_log.txt` for per-attempt details. Common causes:
- DiskPressure (preflight check catches this, waits 60s and retries)
- Ollama pod crash loop (check `kubectl describe pod`)
- Network timeout (gateway URL wrong — set `GATEWAY_URL` env var)
