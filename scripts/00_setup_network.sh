#!/usr/bin/env bash
set -e

# Load central configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${ROOT_DIR}/config.env" ]; then
    set -a
    source "${ROOT_DIR}/config.env"
    set +a
else
    echo "Error: config.env not found!"
    exit 1
fi

echo "=========================================================="
echo "Step 00: Configuring GCP Project & Provisioning Custom VPC"
echo "=========================================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region:     ${REGION}"
echo "Zone:       ${ZONE}"
echo "Network:    ${NETWORK_NAME}"
echo "Subnet:     ${SUBNET_NAME}"
echo "=========================================================="

gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"

echo "Enabling GCP APIs..."
gcloud services enable \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    iam.googleapis.com \
    compute.googleapis.com

echo "Configuring Cloud Build Service Account IAM Permissions..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/storage.objectViewer" || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/logs.writer" || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/artifactregistry.writer" || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CLOUDBUILD_SA}" \
    --role="roles/cloudbuild.builds.builder" || true

echo "Creating Custom VPC Network..."
gcloud compute networks create "${NETWORK_NAME}" --subnet-mode=custom || true

echo "Creating Custom Subnet with IP Aliasing..."
gcloud compute networks subnets create "${SUBNET_NAME}" \
    --network="${NETWORK_NAME}" \
    --region="${REGION}" \
    --range=10.0.0.0/20 \
    --secondary-range=pods-range=10.4.0.0/14,services-range=10.8.0.0/20 || true

echo "Step 00 Setup Complete!"
