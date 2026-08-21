#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -a
source "${ROOT_DIR}/config.env"
set +a

echo "=========================================================="
echo "Step 03: Executing Multi-Node JAX CPU Workload on GKE"
echo "=========================================================="

CPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${CPU_IMAGE_NAME}:${IMAGE_TAG}"

echo "Rendering JobSet manifest..."
sed "s|LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/CPU_IMAGE_NAME:IMAGE_TAG|${CPU_FULL_IMAGE}|g" \
    "${ROOT_DIR}/manifests/jobset-cpu.yaml" > "${ROOT_DIR}/manifests/jobset-cpu-rendered.yaml"

echo "Deploying JobSet to GKE..."
kubectl apply -f "${ROOT_DIR}/manifests/jobset-cpu-rendered.yaml"

echo "Monitoring Pod initialization..."
for i in {1..6}; do
    kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-cpu-job -o custom-columns=POD_NAME:.metadata.name,POD_IP:.status.podIP,NODE:.spec.nodeName,STATUS:.status.phase
    sleep 5
done

echo "Streaming logs from CPU multi-node workers..."
kubectl logs -l jobset.x-k8s.io/jobset-name=jax-cpu-job --all-containers --tail=100
