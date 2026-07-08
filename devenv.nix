{ pkgs, lib, config, ... }:

{
  # ── Packages: Kubernetes & monitoring CLI tools ──────────────────
  packages = with pkgs; [
    k3d
    kubectl
    kubernetes-helm
    jq
    curl
  ];

  # ── Python with uv ──────────────────────────────────────────────
  languages.python = {
    enable = true;
    version = "3.12";
    uv = {
      enable = true;
      sync.enable = true;  # auto-runs `uv sync` on shell entry
    };
  };

  # ── Environment Variables ───────────────────────────────────────
  env = {
    PROJECT_NAME = "llm-d-emulation-bench";
    KUBECONFIG = "${config.devenv.root}/.k3d/kubeconfig.yaml";
    K3D_CLUSTER_NAME = "llm-d-bench";
  };

  # ── Convenience Scripts ─────────────────────────────────────────
  scripts."cluster-create".exec = ''
    echo "Creating k3d cluster..."
    if ! k3d cluster list | grep -q $K3D_CLUSTER_NAME; then
      k3d cluster create --config infra/k3d-config.yaml
    else
      echo "Cluster $K3D_CLUSTER_NAME already exists."
    fi
    mkdir -p $(dirname $KUBECONFIG)
    k3d kubeconfig get $K3D_CLUSTER_NAME > $KUBECONFIG
    echo "Cluster '$K3D_CLUSTER_NAME' is ready!"
  '';

  scripts."cluster-delete".exec = ''
    k3d cluster delete $K3D_CLUSTER_NAME
    rm -f $KUBECONFIG
  '';

  scripts."cluster-status".exec = ''
    echo "=== Cluster ==="
    k3d cluster list
    echo ""
    echo "=== Nodes ==="
    kubectl get nodes 2>/dev/null || echo "No cluster running"
    echo ""
    echo "=== Pods (all namespaces) ==="
    kubectl get pods -A 2>/dev/null || true
  '';

  scripts."deploy-monitoring".exec = ''
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    helm repo update
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
      --namespace monitoring \
      --create-namespace \
      -f infra/monitoring/prometheus-values.yaml \
      --wait
    echo "Monitoring stack deployed!"
  '';
  
  scripts."deploy-llmd".exec = ''
    sh infra/llm-d/setup.sh
  '';

  scripts."lint".exec = ''
    uv run ruff check src/
  '';

  scripts."format".exec = ''
    uv run ruff format src/
  '';

  # ── Shell Hook ──────────────────────────────────────────────────
  enterShell = ''
    echo ""
    echo "🚀 $PROJECT_NAME development environment"
    echo "   Python: $(python --version)"
    echo "   uv:     $(uv --version)"
    echo "   k3d:    $(k3d --version 2>&1 | head -1)"
    echo "   kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client 2>&1 | head -1)"
    echo "   helm:   $(helm version --short)"
    echo ""
    echo "Available commands:"
    echo "  cluster-create    - Create the k3d cluster"
    echo "  cluster-delete    - Delete the k3d cluster"
    echo "  cluster-status    - Show cluster status"
    echo "  deploy-monitoring - Deploy Prometheus + Grafana via Helm"
    echo "  deploy-llmd       - Deploy the llm-d stack and simulators"
    echo "  lint              - Run ruff linter"
    echo "  format            - Run ruff formatter"
    echo ""
  '';
}
