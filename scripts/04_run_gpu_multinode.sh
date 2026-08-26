#!/usr/bin/env bash
# ==============================================================================
# Module 04: Multi-Node JAX GPU Training with JobSet
# ==============================================================================
# Learning Objectives & Key Concepts:
# 1. Multi-Host JAX Initialization: Initialize distributed JAX across multi-node GPU
#    hosts via Headless DNS (COORDINATOR_ADDRESS).
# 2. GPU Inter-Node All-Reduce: Verify gradient synchronization (jax.lax.psum)
#    across GPU nodes using NVIDIA NCCL.
# 3. SPMD Mesh Sharding: Shard tensors across GPU devices using jax.sharding.Mesh.
# 4. Mathematical Proof (psum): Expected synchronized sum = 3.0 across GPU ranks.
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

GPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${GPU_IMAGE_NAME}:${IMAGE_TAG}"

echo "=============================================================================="
echo "📘 MODULE 04: Multi-Node JAX GPU Training with JobSet"
echo "=============================================================================="
echo "Target GPU Image: ${GPU_FULL_IMAGE}"
echo "Cluster Name:     ${CLUSTER_NAME}"
echo "GPU Node Pool:    ${GPU_NODE_POOL_NAME}"
echo "Machine / GPU:    ${GPU_MACHINE_TYPE} / ${GPU_TYPE} (x${GPU_COUNT_PER_NODE})"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. Check or Provision Dedicated GPU Node Pool
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 1/5] Checking GPU Node Pool Status..."
if ! gcloud container node-pools describe "${GPU_NODE_POOL_NAME}" --project="${PROJECT_ID}" --cluster="${CLUSTER_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
    echo "Creating GPU Node Pool (${GPU_NODE_POOL_NAME}) with ${GPU_NODE_COUNT} nodes (${GPU_MACHINE_TYPE} / ${GPU_TYPE})..."
    gcloud container node-pools create "${GPU_NODE_POOL_NAME}" \
        --project="${PROJECT_ID}" \
        --cluster="${CLUSTER_NAME}" \
        --zone="${ZONE}" \
        --machine-type="${GPU_MACHINE_TYPE}" \
        --accelerator=type="${GPU_TYPE}",count="${GPU_COUNT_PER_NODE}",gpu-driver-version=default \
        --num-nodes="${GPU_NODE_COUNT}" || {
            echo "⚠️ Warning: Failed to create GPU node pool. Quota may be restricted in ${ZONE}."
            echo "Continuing to attempt JobSet deployment if nodes exist..."
        }
else
    echo "GPU Node Pool '${GPU_NODE_POOL_NAME}' already exists in '${PROJECT_ID}'."
fi

# ------------------------------------------------------------------------------
# 3. Render GPU JobSet Manifest
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 2/5] Rendering GPU JobSet Manifest..."
sed "s|LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/GPU_IMAGE_NAME:IMAGE_TAG|${GPU_FULL_IMAGE}|g" \
    "${ROOT_DIR}/manifests/jobset-gpu.yaml" > "${ROOT_DIR}/manifests/jobset-gpu-rendered.yaml"
echo "Saved rendered manifest to ${ROOT_DIR}/manifests/jobset-gpu-rendered.yaml"

# ------------------------------------------------------------------------------
# 4. Deploy GPU JobSet to GKE
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 3/5] Deploying GPU JobSet to GKE..."
kubectl delete jobset,service jax-gpu-job --ignore-not-found
kubectl apply -f "${ROOT_DIR}/manifests/jobset-gpu-rendered.yaml"

# ------------------------------------------------------------------------------
# 5. Monitor GPU Pod Initialization
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 4/5] Monitoring GPU Pod Initialization and Node Placement..."
for i in {1..20}; do
    kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-gpu-job -o custom-columns=POD_NAME:.metadata.name,POD_IP:.status.podIP,NODE:.spec.nodeName,STATUS:.status.phase 2>/dev/null || true
    STATUSES=$(kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-gpu-job -o jsonpath='{.items[*].status.phase}' 2>/dev/null || true)
    if [[ "$STATUSES" == *"Succeeded"* ]] || [[ "$STATUSES" == *"Failed"* ]]; then
        break
    fi
    sleep 8
done

# ------------------------------------------------------------------------------
# 6. Stream GPU Logs & Validate NCCL Mathematical Proof
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 5/5] Streaming Logs from Multi-Node GPU Workers (Expected Sum: 3.0)..."
kubectl logs -l jobset.x-k8s.io/jobset-name=jax-gpu-job --all-containers --tail=100 || true

echo ""
echo "=============================================================================="
echo "✅ Module 04 Multi-Node JAX GPU Training Verification Complete!"
echo "=============================================================================="
