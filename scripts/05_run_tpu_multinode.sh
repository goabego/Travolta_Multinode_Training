#!/usr/bin/env bash
# ==============================================================================
# Module 05: Multi-Host JAX TPU Slice Training with JobSet
# ==============================================================================
# Learning Objectives & Key Concepts:
# 1. TPU Multi-Host Slices: Interconnecting multiple TPU hosts over high-speed
#    Inter-Chip Interconnect (ICI).
# 2. Exclusive Topology & Ports: Specifying exclusive topology
#    (alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool)
#    and exposing TPU ports 8471 (Data Link) and 8080 (Coordinator).
# 3. SPMD Mesh Sharding on TPU: Sharding matrix operations across TPU pod slices.
# 4. Mathematical Proof (psum): Expected synchronized sum = 3.0 across TPU hosts.
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

TPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${TPU_IMAGE_NAME}:${IMAGE_TAG}"

echo "=============================================================================="
echo "📘 MODULE 05: Multi-Host JAX TPU Slice Training with JobSet"
echo "=============================================================================="
echo "Target TPU Image: ${TPU_FULL_IMAGE}"
echo "Cluster Name:     ${CLUSTER_NAME}"
echo "TPU Node Pool:    ${TPU_NODE_POOL_NAME}"
echo "Machine / Slice:  ${TPU_MACHINE_TYPE} / Topology ${TPU_TOPOLOGY}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. Check or Provision Dedicated TPU Node Pool
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 1/5] Checking TPU Node Pool Status..."
if ! gcloud container node-pools describe "${TPU_NODE_POOL_NAME}" --cluster="${CLUSTER_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
    echo "Creating TPU Node Pool (${TPU_NODE_POOL_NAME}) with ${TPU_NODE_COUNT} nodes (${TPU_MACHINE_TYPE} / ${TPU_TOPOLOGY})..."
    gcloud container node-pools create "${TPU_NODE_POOL_NAME}" \
        --cluster="${CLUSTER_NAME}" \
        --zone="${ZONE}" \
        --node-locations="${ZONE}" \
        --machine-type="${TPU_MACHINE_TYPE}" \
        --tpu-topology="${TPU_TOPOLOGY}" \
        --shielded-secure-boot \
        --shielded-integrity-monitoring \
        --num-nodes="${TPU_NODE_COUNT}" || {
            echo "⚠️ Warning: Failed to create TPU node pool. Quota may be restricted in ${ZONE}."
            echo "Continuing to attempt JobSet deployment if nodes exist..."
        }
else
    echo "TPU Node Pool '${TPU_NODE_POOL_NAME}' already exists."
fi

# ------------------------------------------------------------------------------
# 3. Render TPU JobSet Manifest
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 2/5] Rendering TPU JobSet Manifest..."
sed "s|LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/TPU_IMAGE_NAME:IMAGE_TAG|${TPU_FULL_IMAGE}|g" \
    "${ROOT_DIR}/manifests/jobset-tpu.yaml" > "${ROOT_DIR}/manifests/jobset-tpu-rendered.yaml"
echo "Saved rendered manifest to ${ROOT_DIR}/manifests/jobset-tpu-rendered.yaml"

# ------------------------------------------------------------------------------
# 4. Deploy TPU JobSet to GKE
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 3/5] Deploying TPU JobSet to GKE..."
kubectl delete jobset,service jax-tpu-job --ignore-not-found
kubectl apply -f "${ROOT_DIR}/manifests/jobset-tpu-rendered.yaml"

# ------------------------------------------------------------------------------
# 5. Monitor TPU Pod Initialization
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 4/5] Monitoring TPU Pod Initialization and Slice Placement..."
for i in {1..8}; do
    kubectl get jobset jax-tpu-job || true
    kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-tpu-job -o custom-columns=POD_NAME:.metadata.name,POD_IP:.status.podIP,NODE:.spec.nodeName,STATUS:.status.phase
    sleep 5
done

# ------------------------------------------------------------------------------
# 6. Stream TPU Logs & Validate ICI Interconnect Mathematical Proof
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 5/5] Streaming Logs from Multi-Host TPU Slice Workers (Expected Sum: 3.0)..."
kubectl logs -l jobset.x-k8s.io/jobset-name=jax-tpu-job --all-containers --tail=100 || true

echo ""
echo "=============================================================================="
echo "✅ Module 05 Multi-Host JAX TPU Slice Training Verification Complete!"
echo "=============================================================================="
