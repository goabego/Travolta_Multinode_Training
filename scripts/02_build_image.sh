#!/usr/bin/env bash
# ==============================================================================
# Module 02: Container Build & Artifact Registry Setup
# ==============================================================================
# Learning Objectives:
# 1. Create a Docker Artifact Registry repository in GCP.
# 2. Configure Docker client authentication with Google Artifact Registry.
# 3. Ensure Cloud Build service account permissions for storage and logging.
# 4. Build the primary zero-quota CPU JAX container image (Dockerfile.cpu).
# 5. Build optional GPU (Dockerfile.gpu) and TPU (Dockerfile.tpu) container images.
# 6. Verify built container images in Artifact Registry.
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
echo "📘 MODULE 02: Container Build & Artifact Registry Setup"
echo "=============================================================================="
echo "Project ID:        ${PROJECT_ID}"
echo "Region:            ${REGION}"
echo "Repository:        ${ARTIFACT_REGISTRY_REPO}"
echo "CPU Image:         ${CPU_IMAGE_NAME}:${IMAGE_TAG}"
echo "GPU Image:         ${GPU_IMAGE_NAME}:${IMAGE_TAG}"
echo "TPU Image:         ${TPU_IMAGE_NAME}:${IMAGE_TAG}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. Create Artifact Registry Repository & Configure Authentication
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 1/5] Creating Artifact Registry Repository '${ARTIFACT_REGISTRY_REPO}'..."
if ! gcloud artifacts repositories describe "${ARTIFACT_REGISTRY_REPO}" --location="${REGION}" >/dev/null 2>&1; then
    gcloud artifacts repositories create "${ARTIFACT_REGISTRY_REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="JAX Multi-Node Container Repository"
else
    echo "Artifact Registry repository '${ARTIFACT_REGISTRY_REPO}' already exists."
fi

echo ""
echo "▶ [Step 2/5] Configuring Docker authentication for '${REGION}-docker.pkg.dev'..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ------------------------------------------------------------------------------
# 3. Ensure Service Account Permissions for Cloud Build
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 3/5] Verifying Cloud Build Service Account Permissions..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/storage.objectViewer" --quiet || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/logging.logWriter" --quiet || true

TARGET="${1:-cpu}"
CPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${CPU_IMAGE_NAME}:${IMAGE_TAG}"
GPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${GPU_IMAGE_NAME}:${IMAGE_TAG}"
TPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${TPU_IMAGE_NAME}:${IMAGE_TAG}"

# ------------------------------------------------------------------------------
# 4. Build Container Images via Cloud Build
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 4/6] Building JAX Container Images via Google Cloud Build (Target: ${TARGET})..."
echo "ℹ️ Estimated Build Times:"
echo "   - CPU Image (Dockerfile.cpu): ~1.5 - 2 minutes"
echo "   - GPU Image (Dockerfile.gpu): ~4 - 6 minutes (CUDA 12 + NCCL layers)"
echo "   - TPU Image (Dockerfile.tpu): ~2 - 3 minutes (libtpu packages)"
echo "   - All Images concurrently:   ~6 - 8 minutes total"
echo ""

CLOUDBUILD_CONFIG="/tmp/cloudbuild_jax.yaml"
cat <<EOF > "${CLOUDBUILD_CONFIG}"
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', '\${_IMAGE}', '-f', '\${_DOCKERFILE}', '.']
images:
- '\${_IMAGE}'
EOF

if [ "$TARGET" = "cpu" ] || [ "$TARGET" = "all" ]; then
    echo "🔨 [1/3] Building CPU JAX Image (Estimated: ~2 min): ${CPU_FULL_IMAGE}..."
    gcloud builds submit "${ROOT_DIR}/src/" \
        --config="${CLOUDBUILD_CONFIG}" \
        --substitutions=_IMAGE="${CPU_FULL_IMAGE}",_DOCKERFILE="Dockerfile.cpu"
fi

if [ "$TARGET" = "gpu" ] || [ "$TARGET" = "all" ]; then
    echo "🔨 [2/3] Building GPU JAX Image (Estimated: ~5 min): ${GPU_FULL_IMAGE}..."
    gcloud builds submit "${ROOT_DIR}/src/" \
        --config="${CLOUDBUILD_CONFIG}" \
        --substitutions=_IMAGE="${GPU_FULL_IMAGE}",_DOCKERFILE="Dockerfile.gpu"
fi

if [ "$TARGET" = "tpu" ] || [ "$TARGET" = "all" ]; then
    echo "🔨 [3/3] Building TPU JAX Image (Estimated: ~3 min): ${TPU_FULL_IMAGE}..."
    gcloud builds submit "${ROOT_DIR}/src/" \
        --config="${CLOUDBUILD_CONFIG}" \
        --substitutions=_IMAGE="${TPU_FULL_IMAGE}",_DOCKERFILE="Dockerfile.tpu"
fi

rm -f "${CLOUDBUILD_CONFIG}"

# ------------------------------------------------------------------------------
# 5. Check Recent Cloud Build Execution History
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 5/6] Inspecting Cloud Build Execution Status & Durations..."
echo "To view detailed logs for any build in the future, run: gcloud builds log <BUILD_ID>"
gcloud builds list --limit=3 --format="table(id,createTime,duration,status,images[0])"

# ------------------------------------------------------------------------------
# 6. Verify Built Images in Artifact Registry
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 6/6] Verifying Container Images in Google Artifact Registry..."
gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}" \
    --format="table(package,image.basename(),createTime,size)"

echo ""
echo "=============================================================================="
echo "✅ Module 02 Container Build Complete! Image is ready for Module 03."
echo "=============================================================================="
