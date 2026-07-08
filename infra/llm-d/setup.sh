#!/usr/bin/env sh
set -e

echo "Deploying Gateway API CRDs (experimental)..."
kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.1/experimental-install.yaml

echo "Adding llm-d helm repository..."
# We use the oci registry directly for GAIE/llm-d

echo "Deploying Inference Extension CRDs..."
kubectl apply -k "github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=v1.4.0"
helm upgrade --install gaie oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool --version v1.4.0 --namespace llm-d --create-namespace --set inferencePool.modelServers.matchLabels.app=vllm-sim -f infra/llm-d/values.yaml

echo "Deploying kgateway (Gateway)..."
helm upgrade --install kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds --namespace kgateway-system --create-namespace
helm upgrade --install kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --namespace kgateway-system --create-namespace --set inferenceExtension.enabled=true



echo "Deploying InferencePools (Simulators)..."
kubectl apply -f infra/llm-d/inference-pool.yaml

echo "llm-d stack deployment complete!"
