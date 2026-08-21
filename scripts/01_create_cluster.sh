#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -a
source "${ROOT_DIR}/config.env"
set +a

echo "=========================================================="
echo "Step 01: Provisioning GKE Standard Cluster & JobSet Operator"
echo "=========================================================="

echo "Creating GKE Cluster ${CLUSTER_NAME}..."
gcloud container clusters create "${CLUSTER_NAME}" \
    --zone="${ZONE}" \
    --release-channel=regular \
    --workload-pool="${PROJECT_ID}.svc.id.goog" \
    --enable-ip-alias \
    --network="${NETWORK_NAME}" \
    --subnetwork="${SUBNET_NAME}" \
    --cluster-secondary-range-name=pods-range \
    --services-secondary-range-name=services-range \
    --enable-shielded-nodes \
    --shielded-secure-boot \
    --shielded-integrity-monitoring \
    --num-nodes=2 \
    --machine-type=e2-standard-4 || true

echo "Fetching cluster credentials..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone="${ZONE}"

echo "Installing Kubernetes JobSet Controller (${JOBSET_VERSION})..."
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml"

echo "Cluster & Operator Setup Complete!"
kubectl get nodes -o wide
