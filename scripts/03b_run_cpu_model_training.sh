#!/usr/bin/env bash
# ==============================================================================
# Module 03b: Multi-Node JAX CPU Toy Model Training (Zero Quota)
# ==============================================================================
# Learning Objectives & Key Concepts:
# 1. Distributed Model Training on CPUs: Execute a full deep learning training loop
#    (MLP classifier) across multi-node GKE CPUs using Flax + Optax.
# 2. SPMD Data Parallelism: Shard mini-batches across physical nodes while
#    keeping model weights replicated.
# 3. Distributed Backpropagation & Optimizer: Verify cross-node gradient all-reduce
#    synchronization and weight updates in real-time.
# 4. Convergence Verification: Validate that loss decreases monotonically across
#    epochs to prove end-to-end multi-node training correctness.
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

CPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${CPU_IMAGE_NAME}:${IMAGE_TAG}"

echo "=============================================================================="
echo "📘 MODULE 03b: Multi-Node JAX CPU Toy Model Training (Zero Quota)"
echo "=============================================================================="
echo "Project ID:      ${PROJECT_ID}"
echo "Target Image:    ${CPU_FULL_IMAGE}"
echo "Cluster Name:    ${CLUSTER_NAME}"
echo "Zone:            ${ZONE}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. Render CPU Training JobSet Manifest
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 1/4] Rendering CPU Model Training JobSet Manifest..."
sed "s|LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/CPU_IMAGE_NAME:IMAGE_TAG|${CPU_FULL_IMAGE}|g" \
    "${ROOT_DIR}/manifests/jobset-cpu-training.yaml" > "${ROOT_DIR}/manifests/jobset-cpu-training-rendered.yaml"
echo "Saved rendered manifest to ${ROOT_DIR}/manifests/jobset-cpu-training-rendered.yaml"

# ------------------------------------------------------------------------------
# 3. Deploy JobSet to GKE
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 2/4] Deploying CPU Model Training JobSet to GKE..."
kubectl delete jobset,service jax-cpu-train-job --ignore-not-found
kubectl apply -f "${ROOT_DIR}/manifests/jobset-cpu-training-rendered.yaml"

# ------------------------------------------------------------------------------
# 4. Monitor Worker Pods & Verify Pod Anti-Affinity Placement
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 3/4] Monitoring Worker Pods & Node Distribution..."
for i in {1..6}; do
    kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-cpu-train-job -o custom-columns=POD_NAME:.metadata.name,POD_IP:.status.podIP,NODE:.spec.nodeName,STATUS:.status.phase 2>/dev/null || true
    STATUSES=$(kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-cpu-train-job -o jsonpath='{.items[*].status.phase}' 2>/dev/null || true)
    if [[ "$STATUSES" == *"Running"* ]] || [[ "$STATUSES" == *"Succeeded"* ]]; then
        break
    fi
    sleep 4
done

# ------------------------------------------------------------------------------
# 5. Stream Real-Time Training Logs & Convergence Verification
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 4/4] Streaming Multi-Node Model Training Logs (Rank 0 / Epoch Progression)..."
kubectl logs -l jobset.x-k8s.io/jobset-name=jax-cpu-train-job -c jax-cpu-worker -f || true

echo ""
echo "=============================================================================="
echo "✅ Module 03b Multi-Node JAX CPU Toy Model Training Complete!"
echo "=============================================================================="
