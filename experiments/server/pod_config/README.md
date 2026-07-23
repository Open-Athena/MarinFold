# CoreWeave pod config

These manifests deploy the minimal MarinFold inference API to the CoreWeave
cluster. The current deployment clones this public repo at startup and runs the
server from `experiments/server`; replace the image/command with a built image
once the service grows beyond health checks.

## Deploy

```bash
export KUBECONFIG=~/.kube/CWKubeconfig_marin-gpu_US-EAST-02A
kubectl config use-context marin-gpu_US-EAST-02A

kubectl create secret generic marinfold-inference-server-token \
  --from-literal=token="$(openssl rand -hex 32)" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f experiments/server/pod_config/deployment.yaml
kubectl apply -f experiments/server/pod_config/service.yaml
kubectl rollout status deployment/marinfold-inference-server

kubectl port-forward svc/marinfold-inference-server 8080:8080
curl http://127.0.0.1:8080/healthz
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8080/v1/auth-check
```

## Cleanup

```bash
kubectl delete -f experiments/server/pod_config/service.yaml
kubectl delete -f experiments/server/pod_config/deployment.yaml
```
