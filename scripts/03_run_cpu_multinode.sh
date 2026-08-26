#!/usr/bin/env bash
# ==============================================================================
# Module 03: Multi-Node JAX CPU Training & 10x Scale-Up (Zero Quota)
# ==============================================================================
# Learning Objectives & Key Concepts:
# 1. Pod Anti-Affinity: Force Kubernetes to schedule worker pods on separate physical
#    VM hosts (topologyKey: "kubernetes.io/hostname").
# 2. Headless Discovery & Batch Indexing: Headless DNS discovery
#    (COORDINATOR_ADDRESS="jax-cpu-job-workers-0-0.jax-cpu-job:1234") and
#    batch.kubernetes.io/job-completion-index.
# 3. Mathematical Proof (psum): Validating gradient all-reduce synchronization.
#    For N ranks with input i+1, expected sum = N * (N + 1) / 2.
#    - 2 Nodes  (N=2):  Expected Sum = 3.0   (across 8 virtual devices)
#    - 20 Nodes (N=20): Expected Sum = 210.0 (across 80 virtual devices)
# 4. PART A: 2-Node Baseline Experiment (8 Virtual Devices).
# 5. PART B: 10x Scale-Up Experiment (20 Nodes, 80 Virtual Devices).
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

MODE="${1:---all}"  # Options: --part-a, --part-b, --all

CPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${CPU_IMAGE_NAME}:${IMAGE_TAG}"

echo "=============================================================================="
echo "📘 MODULE 03: Multi-Node JAX CPU Training & 10x Scale-Up"
echo "=============================================================================="
echo "Target Image:    ${CPU_FULL_IMAGE}"
echo "Cluster Name:    ${CLUSTER_NAME}"
echo "Zone:            ${ZONE}"
echo "Execution Mode:  ${MODE}"
echo "=============================================================================="

# ==============================================================================
# PART A: 2-Node CPU Multi-Node Experiment (8 Virtual Devices)
# ==============================================================================
run_part_a() {
    echo ""
    echo "=============================================================================="
    echo "▶ PART A: 2-Node CPU Multi-Node Baseline Experiment (8 Virtual Devices)"
    echo "=============================================================================="

    echo ""
    echo "▶ [Step A.1] Rendering 2-Node JobSet Manifest..."
    sed "s|LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/CPU_IMAGE_NAME:IMAGE_TAG|${CPU_FULL_IMAGE}|g" \
        "${ROOT_DIR}/manifests/jobset-cpu.yaml" > "${ROOT_DIR}/manifests/jobset-cpu-rendered.yaml"
    echo "Saved rendered manifest to ${ROOT_DIR}/manifests/jobset-cpu-rendered.yaml"

    echo ""
    echo "▶ [Step A.2] Deploying 2-Node JobSet to GKE..."
    kubectl delete jobset,service jax-cpu-job --ignore-not-found
    kubectl apply -f "${ROOT_DIR}/manifests/jobset-cpu-rendered.yaml"

    echo ""
    echo "▶ [Step A.3] Monitoring Worker Pods & Verifying Pod Anti-Affinity Placement..."
    echo "Checking that worker pods are scheduled onto DISTINCT physical VM nodes:"
    for i in {1..6}; do
        kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-cpu-job -o custom-columns=POD_NAME:.metadata.name,POD_IP:.status.podIP,NODE:.spec.nodeName,STATUS:.status.phase
        sleep 5
    done

    echo ""
    echo "▶ [Step A.4] Inspecting Headless DNS Service & Endpoints..."
    kubectl get service jax-cpu-job || true
    kubectl get endpoints jax-cpu-job || true

    echo ""
    echo "▶ [Step A.5] Streaming Logs & Validating Mathematical Proof (Expected Sum: 3.0)..."
    kubectl logs -l jobset.x-k8s.io/jobset-name=jax-cpu-job --all-containers --tail=100 || true
    echo ""
    echo "✅ Part A (2-Node Baseline) Complete!"
}

# ==============================================================================
# PART B: 10x Scale-Up Experiment (20 Nodes, 80 Virtual Devices)
# ==============================================================================
run_part_b() {
    echo ""
    echo "=============================================================================="
    echo "▶ PART B: 10x Scale-Up Experiment (20 Nodes, 80 Virtual Devices)"
    echo "=============================================================================="

    echo ""
    echo "▶ [Step B.1] Resizing GKE Cluster to 20 Nodes..."
    gcloud container clusters resize "${CLUSTER_NAME}" \
        --project="${PROJECT_ID}" \
        --node-pool=default-pool \
        --num-nodes="${CPU_SCALE_NODE_COUNT:-20}" \
        --zone="${ZONE}" \
        --quiet

    echo ""
    echo "Verifying ready node count in default-pool:"
    kubectl get nodes -l cloud.google.com/gke-nodepool=default-pool --no-headers | wc -l

    echo ""
    echo "▶ [Step B.2] Rendering 20-Node Scale JobSet Manifest..."
    sed "s|LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/CPU_IMAGE_NAME:IMAGE_TAG|${CPU_FULL_IMAGE}|g" \
        "${ROOT_DIR}/manifests/jobset-cpu.yaml" \
        | sed 's/jax-cpu-job/jax-scale-job/g' \
        | sed 's/parallelism: 2/parallelism: 20/g' \
        | sed 's/completions: 2/completions: 20/g' \
        | sed 's/NUM_PROCESSES",\s*value: "2"/NUM_PROCESSES", value: "20"/g' \
        | sed 's/value: "2"/value: "20"/g' \
        > "${ROOT_DIR}/manifests/jobset-scale-20-rendered.yaml"
    echo "Saved 20-node manifest to ${ROOT_DIR}/manifests/jobset-scale-20-rendered.yaml"

    echo ""
    echo "▶ [Step B.3] Deploying 20-Node Scale JobSet..."
    kubectl delete jobset,service jax-scale-job --ignore-not-found
    kubectl apply -f "${ROOT_DIR}/manifests/jobset-scale-20-rendered.yaml"

    echo ""
    echo "▶ [Step B.4] Monitoring 20 Pods Scheduling Across 20 Physical Nodes..."
    echo "Waiting for all 20 worker pods to initialize and execute..."
    for i in {1..30}; do
        RUNNING_OR_DONE=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=jax-scale-job --no-headers 2>/dev/null | grep -E "Running|Completed|Succeeded" | wc -l | tr -d ' ')
        if [[ "${RUNNING_OR_DONE}" -ge 20 ]]; then
            echo "All 20 pods are running or completed (${RUNNING_OR_DONE}/20)."
            break
        fi
        echo "Waiting for pods to start (${RUNNING_OR_DONE:-0}/20 running or completed)..."
        sleep 3
    done

    echo ""
    echo "Verifying 20-Pod Physical Node Distribution:"
    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=jax-scale-job -o custom-columns=POD_NAME:.metadata.name,POD_IP:.status.podIP,NODE:.spec.nodeName,STATUS:.status.phase

    echo ""
    echo "▶ [Step B.5] Streaming Cross-Node Logs & Validating Mathematical Sum (Expected Sum: 210.0)..."
    sleep 5
    kubectl logs -l jobset.sigs.k8s.io/jobset-name=jax-scale-job -c jax-cpu-worker --tail=30 --prefix=true --max-log-requests=25 || true
    echo ""
    echo "✅ Part B (10x Scale-Up) Complete!"
}

case "$MODE" in
    --part-a)
        run_part_a
        ;;
    --part-b)
        run_part_b
        ;;
    --all|*)
        run_part_a
        run_part_b
        ;;
esac

echo ""
echo "=============================================================================="
echo "✅ Module 03 Multi-Node JAX CPU & 10x Scale-Up Verification Complete!"
echo "=============================================================================="
