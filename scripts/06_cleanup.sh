#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -a
source "${ROOT_DIR}/config.env"
set +a

echo "=========================================================="
echo "Step 06: Tearing Down All Deployed GCP Resources"
echo "=========================================================="

echo "Deleting JobSets..."
kubectl delete jobset jax-cpu-job jax-scale-job jax-gpu-job jax-tpu-job --ignore-not-found || true

echo "Deleting GKE Cluster ${CLUSTER_NAME}..."
gcloud container clusters delete "${CLUSTER_NAME}" --zone="${ZONE}" --quiet || true

echo "Deleting Artifact Registry Repository ${ARTIFACT_REGISTRY_REPO}..."
gcloud artifacts repositories delete "${ARTIFACT_REGISTRY_REPO}" --location="${REGION}" --quiet || true

echo "Deleting Custom Subnet & VPC Network..."
gcloud compute networks subnets delete "${SUBNET_NAME}" --region="${REGION}" --quiet || true
gcloud compute networks delete "${NETWORK_NAME}" --quiet || true

echo "Cleanup Complete!"
