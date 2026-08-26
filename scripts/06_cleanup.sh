#!/usr/bin/env bash
# ==============================================================================
# Module 06: Resource Teardown and Cost Management
# ==============================================================================
# Learning Objectives & Teardown Steps:
# 1. Delete active Kubernetes JobSets (jax-cpu-job, jax-scale-job, jax-gpu-job, jax-tpu-job).
# 2. Scale down or delete accelerator node pools (GPU and TPU).
# 3. Delete Artifact Registry container images & repository.
# 4. Delete GKE Cluster.
# 5. Delete Custom VPC Subnet and Network.
# 6. Final verification to ensure zero lingering compute billing.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------------------------
# 1. Load Central Configuration Variables
# ------------------------------------------------------------------------------
if [ -f "${ROOT_DIR}/config.env" ]; then
    set -a
    source "${ROOT_DIR}/config.env"
    set +a
else
    echo "❌ Error: config.env not found at ${ROOT_DIR}/config.env!"
    exit 1
fi

echo "=============================================================================="
echo "📘 MODULE 06: Resource Teardown and Cost Management"
echo "=============================================================================="
echo "Project ID:      ${PROJECT_ID}"
echo "Cluster Name:    ${CLUSTER_NAME}"
echo "Zone:            ${ZONE}"
echo "Repository:      ${ARTIFACT_REGISTRY_REPO}"
echo "Network:         ${NETWORK_NAME}"
echo "Subnet:          ${SUBNET_NAME}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. Delete Kubernetes JobSets
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 1/5] Deleting Kubernetes JobSets..."
kubectl delete jobset jax-cpu-job jax-scale-job jax-gpu-job jax-tpu-job --ignore-not-found || true

# ------------------------------------------------------------------------------
# 3. Delete GKE Cluster
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 2/5] Deleting GKE Cluster '${CLUSTER_NAME}' in project '${PROJECT_ID}'..."
if gcloud container clusters describe "${CLUSTER_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" >/dev/null 2>&1; then
    gcloud container clusters delete "${CLUSTER_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet
else
    echo "GKE Cluster '${CLUSTER_NAME}' does not exist or already deleted in '${PROJECT_ID}'."
fi

# ------------------------------------------------------------------------------
# 4. Delete Artifact Registry Repository
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 3/5] Deleting Artifact Registry Repository '${ARTIFACT_REGISTRY_REPO}' in project '${PROJECT_ID}'..."
if gcloud artifacts repositories describe "${ARTIFACT_REGISTRY_REPO}" --project="${PROJECT_ID}" --location="${REGION}" >/dev/null 2>&1; then
    gcloud artifacts repositories delete "${ARTIFACT_REGISTRY_REPO}" --project="${PROJECT_ID}" --location="${REGION}" --quiet
else
    echo "Artifact Registry repository '${ARTIFACT_REGISTRY_REPO}' does not exist or already deleted in '${PROJECT_ID}'."
fi

# ------------------------------------------------------------------------------
# 5. Delete Cloud NAT, Cloud Router, Custom Subnet & VPC Network
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 4/5] Deleting Cloud NAT, Cloud Router, Subnet '${SUBNET_NAME}' and VPC Network '${NETWORK_NAME}' in '${PROJECT_ID}'..."
if gcloud compute routers nats describe "${NETWORK_NAME}-nat" --project="${PROJECT_ID}" --router="${NETWORK_NAME}-router" --region="${REGION}" >/dev/null 2>&1; then
    gcloud compute routers nats delete "${NETWORK_NAME}-nat" --project="${PROJECT_ID}" --router="${NETWORK_NAME}-router" --region="${REGION}" --quiet || true
fi

if gcloud compute routers describe "${NETWORK_NAME}-router" --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
    gcloud compute routers delete "${NETWORK_NAME}-router" --project="${PROJECT_ID}" --region="${REGION}" --quiet || true
fi

if gcloud compute networks subnets describe "${SUBNET_NAME}" --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
    gcloud compute networks subnets delete "${SUBNET_NAME}" --project="${PROJECT_ID}" --region="${REGION}" --quiet || true
fi

if gcloud compute networks describe "${NETWORK_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud compute networks delete "${NETWORK_NAME}" --project="${PROJECT_ID}" --quiet || true
fi

# ------------------------------------------------------------------------------
# 6. Final GCP Verification
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 5/5] Performing Final GCP Verification Check for '${PROJECT_ID}'..."
echo "Active GKE clusters in ${ZONE}:"
gcloud container clusters list --project="${PROJECT_ID}" --zone="${ZONE}"

echo ""
echo "=============================================================================="
echo "✅ Module 06 Teardown & Cleanup Complete!"
echo "=============================================================================="
