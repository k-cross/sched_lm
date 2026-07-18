#!/usr/bin/env sh
set -e

echo "Deploying Gateway API CRDs (experimental)..."
kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.1/experimental-install.yaml

echo "Adding llm-d helm repository..."
# We use the oci registry directly for GAIE/llm-d

echo "Deploying Inference Extension CRDs..."
# v1.5.0 to match the GIE version pinned by llm-d-router v0.9.2 (src/gateway-plugin).
kubectl apply -k "github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=v1.5.0"
helm upgrade --install gaie oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool --version v1.5.0 --namespace llm-d --create-namespace -f infra/llm-d/values.yaml

echo "Deploying kgateway (Gateway)..."
# Pinned. Version notes: kgateway v2.2 removed InferencePool backendRef support from
# the Envoy data plane (moved to agentgateway), so the EPP is attached with a
# hand-rolled GatewayExtension/TrafficPolicy in inference-pool.yaml instead; the
# v2.1.x envoy-wrapper images are unusable anyway on this ARM64 cluster (they ship
# x86_64 envoy binaries that crash under Rosetta). inferenceExtension.enabled no
# longer exists in the 2.3.x chart.
KGTW_VERSION=v2.3.6
helm upgrade --install kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds --version $KGTW_VERSION --namespace kgateway-system --create-namespace
helm upgrade --install kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --version $KGTW_VERSION --namespace kgateway-system --create-namespace



echo "Deploying InferencePools (Simulators)..."
kubectl apply -f infra/llm-d/inference-pool.yaml

echo "llm-d stack deployment complete!"
