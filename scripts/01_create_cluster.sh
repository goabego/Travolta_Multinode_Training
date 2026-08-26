#!/usr/bin/env bash
# ==============================================================================
# Module 01: GKE Standard Cluster & Accelerator Node Pools Setup
# ==============================================================================
# Learning Objectives:
# 1. Provision a GKE Standard Cluster with Workload Identity, Shielded VM, and IP-Aliasing.
# 2. Authenticate kubectl against the newly provisioned GKE cluster.
# 3. Install the Kubernetes JobSet Controller (jobset.x-k8s.io).
# 4. Optional: Provision GPU (NVIDIA L4) and/or TPU (v5e 2x4 slice) Node Pools.
# 5. Verify cluster readiness, node distribution, and JobSet CRDs.
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
echo "📘 MODULE 01: GKE Standard Cluster & Accelerator Node Pools Setup"
echo "=============================================================================="
echo "Cluster Name:    ${CLUSTER_NAME}"
echo "Zone:            ${ZONE}"
echo "VPC Network:     ${NETWORK_NAME}"
echo "Subnetwork:      ${SUBNET_NAME}"
echo "JobSet Version:  ${JOBSET_VERSION}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. Provision GKE Base Standard Cluster with Shielded VM & Private Nodes
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 1/5] Creating GKE Cluster '${CLUSTER_NAME}' with Workload Identity & Private Nodes..."

# If cluster exists in ERROR state, delete it first
if gcloud container clusters describe "${CLUSTER_NAME}" --zone="${ZONE}" --format="value(status)" 2>/dev/null | grep -q "ERROR"; then
    echo "⚠️ Existing cluster is in ERROR state. Recreating..."
    gcloud container clusters delete "${CLUSTER_NAME}" --zone="${ZONE}" --quiet
fi

if ! gcloud container clusters describe "${CLUSTER_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
    gcloud container clusters create "${CLUSTER_NAME}" \
        --zone="${ZONE}" \
        --release-channel=regular \
        --workload-pool="${PROJECT_ID}.svc.id.goog" \
        --enable-ip-alias \
        --enable-private-nodes \
        --master-ipv4-cidr="172.16.0.0/28" \
        --no-enable-master-authorized-networks \
        --network="${NETWORK_NAME}" \
        --subnetwork="${SUBNET_NAME}" \
        --cluster-secondary-range-name=pods-range \
        --services-secondary-range-name=services-range \
        --num-nodes="${CPU_NODE_COUNT:-2}" \
        --machine-type=e2-standard-4
else
    echo "GKE Cluster '${CLUSTER_NAME}' already exists."
fi

# ------------------------------------------------------------------------------
# 3. Fetch Cluster Credentials for kubectl
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 2/5] Fetching cluster credentials for kubectl..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone="${ZONE}"

echo ""
echo "▶ [Step 3/5] Installing Kubernetes JobSet Controller (${JOBSET_VERSION})..."
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml"

echo "Waiting for JobSet Controller Manager pod to be Ready..."
kubectl wait --for=condition=Ready pod -l control-plane=controller-manager -n jobset-system --timeout=90s || true

# ------------------------------------------------------------------------------
# 5. Optional Accelerator Node Pools (GPU / TPU)
# ------------------------------------------------------------------------------
WITH_GPU=false
WITH_TPU=false

for arg in "$@"; do
    case $arg in
        --with-gpu) WITH_GPU=true ;;
        --with-tpu) WITH_TPU=true ;;
        --all) WITH_GPU=true; WITH_TPU=true ;;
    esac
done

if [ "$WITH_GPU" = true ]; then
    echo ""
    echo "▶ [Optional] Provisioning GPU Node Pool (${GPU_NODE_POOL_NAME})..."
    if ! gcloud container node-pools describe "${GPU_NODE_POOL_NAME}" --cluster="${CLUSTER_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
        gcloud container node-pools create "${GPU_NODE_POOL_NAME}" \
            --cluster="${CLUSTER_NAME}" \
            --zone="${ZONE}" \
            --machine-type="${GPU_MACHINE_TYPE}" \
            --accelerator=type="${GPU_TYPE}",count="${GPU_COUNT_PER_NODE}" \
            --num-nodes="${GPU_NODE_COUNT}" || echo "Warning: GPU pool creation failed (check quota)."
        
        kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/ubuntu/daemonset-unified.yaml || true
    fi
fi

if [ "$WITH_TPU" = true ]; then
    echo ""
    echo "▶ [Optional] Provisioning TPU Node Pool (${TPU_NODE_POOL_NAME})..."
    if ! gcloud container node-pools describe "${TPU_NODE_POOL_NAME}" --cluster="${CLUSTER_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
        gcloud container node-pools create "${TPU_NODE_POOL_NAME}" \
            --cluster="${CLUSTER_NAME}" \
            --zone="${ZONE}" \
            --node-locations="${ZONE}" \
            --machine-type="${TPU_MACHINE_TYPE}" \
            --tpu-topology="${TPU_TOPOLOGY}" \
            --num-nodes="${TPU_NODE_COUNT}" || echo "Warning: TPU pool creation failed (check quota)."
    fi
fi

# ------------------------------------------------------------------------------
# 6. Verify Cluster Readiness & JobSet CRD Status
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 4/5] Verifying GKE Nodes..."
kubectl get nodes -o wide

echo ""
echo "▶ [Step 5/5] Verifying JobSet CRD Registration..."
kubectl get crds | grep jobsets

echo ""
echo "=============================================================================="
echo "✅ Module 01 GKE Cluster & Operator Setup Complete!"
echo "=============================================================================="
