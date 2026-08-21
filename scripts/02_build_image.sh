#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -a
source "${ROOT_DIR}/config.env"
set +a

echo "=========================================================="
echo "Step 02: Building JAX Container Images via Cloud Build"
echo "=========================================================="

echo "Creating Artifact Registry Repository ${ARTIFACT_REGISTRY_REPO}..."
gcloud artifacts repositories create "${ARTIFACT_REGISTRY_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="JAX Multi-Node Container Repository" || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

CPU_FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${CPU_IMAGE_NAME}:${IMAGE_TAG}"

echo "Building CPU JAX Image via Cloud Build: ${CPU_FULL_IMAGE}..."
gcloud builds submit "${ROOT_DIR}/src/" \
    --config=- <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', '${CPU_FULL_IMAGE}', '-f', 'Dockerfile.cpu', '.']
images:
- '${CPU_FULL_IMAGE}'
EOF

echo "Container Image Build Complete!"
gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}"
