#!/usr/bin/env bash
# ==============================================================================
# Module 00: GCP Environment, Authentication & Custom VPC Setup
# ==============================================================================
# Learning Objectives:
# 1. Load central environment variables (PROJECT_ID, REGION, ZONE, etc.).
# 2. Configure active gcloud CLI project, compute region, and compute zone.
# 3. Enable essential GCP APIs (container, artifactregistry, cloudbuild, iam, compute).
# 4. Configure IAM permissions for Cloud Build & Compute Engine default service accounts.
# 5. Provision custom VPC Network and Subnet with IP-Aliasing (pods-range, services-range).
# 6. Verify VPC & Subnet status.
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
echo "📘 MODULE 00: GCP Environment, Authentication & Custom VPC Setup"
echo "=============================================================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region:     ${REGION}"
echo "Zone:       ${ZONE}"
echo "Network:    ${NETWORK_NAME}"
echo "Subnet:     ${SUBNET_NAME}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. Configure gcloud CLI Active Project & Enable Essential GCP APIs
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 1/5] Setting active gcloud project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}" --quiet

echo ""
echo "▶ [Step 2/5] Enabling required GKE, Container, Compute & Cloud Build APIs..."
gcloud services enable \
    compute.googleapis.com \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    iam.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet

echo "Setting default compute region and zone..."
gcloud config set compute/region "${REGION}" --quiet
gcloud config set compute/zone "${ZONE}" --quiet

# ------------------------------------------------------------------------------
# 4. Configure IAM Permissions for Cloud Build & Compute Default Service Account
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 3/5] Configuring IAM Service Account Permissions for Cloud Build..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo "Granting roles/storage.objectViewer, roles/logging.logWriter, roles/artifactregistry.writer to ${COMPUTE_SA}..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/storage.objectViewer" --quiet || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/logging.logWriter" --quiet || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/artifactregistry.writer" --quiet || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CLOUDBUILD_SA}" \
    --role="roles/cloudbuild.builds.builder" --quiet || true

# ------------------------------------------------------------------------------
# 5. Provision Custom VPC & Subnet with Secondary IP Ranges (IP-Aliasing)
# ------------------------------------------------------------------------------
echo ""
echo "▶ [Step 4/5] Provisioning Custom VPC Network (${NETWORK_NAME})..."
if ! gcloud compute networks describe "${NETWORK_NAME}" >/dev/null 2>&1; then
    gcloud compute networks create "${NETWORK_NAME}" --subnet-mode=custom
else
    echo "VPC network '${NETWORK_NAME}' already exists."
fi

echo ""
echo "▶ [Step 5/5] Provisioning Custom Subnet (${SUBNET_NAME}) with secondary IP ranges for Pods and Services..."
if ! gcloud compute networks subnets describe "${SUBNET_NAME}" --region="${REGION}" >/dev/null 2>&1; then
    gcloud compute networks subnets create "${SUBNET_NAME}" \
        --network="${NETWORK_NAME}" \
        --region="${REGION}" \
        --range=10.0.0.0/20 \
        --secondary-range=pods-range=10.4.0.0/14,services-range=10.8.0.0/20 \
        --enable-private-ip-google-access
else
    echo "Subnet '${SUBNET_NAME}' already exists. Ensuring Private Google Access is enabled..."
    gcloud compute networks subnets update "${SUBNET_NAME}" --region="${REGION}" --enable-private-ip-google-access --quiet || true
fi

# Configure Cloud Router & NAT for secure outbound internet access without public node IPs (Org Policy compliant)
echo "Configuring Cloud Router and Cloud NAT for private node outbound connectivity..."
if ! gcloud compute routers describe "${NETWORK_NAME}-router" --region="${REGION}" >/dev/null 2>&1; then
    gcloud compute routers create "${NETWORK_NAME}-router" --network="${NETWORK_NAME}" --region="${REGION}" --quiet || true
fi

if ! gcloud compute routers nats describe "${NETWORK_NAME}-nat" --router="${NETWORK_NAME}-router" --region="${REGION}" >/dev/null 2>&1; then
    gcloud compute routers nats create "${NETWORK_NAME}-nat" \
        --router="${NETWORK_NAME}-router" \
        --region="${REGION}" \
        --auto-allocate-nat-external-ips \
        --nat-all-subnet-ip-ranges \
        --quiet || true
fi

echo ""
echo "=============================================================================="
echo "✅ Module 00 Setup Complete! Subnet Status:"
echo "=============================================================================="
gcloud compute networks subnets describe "${SUBNET_NAME}" --region="${REGION}" --format="table(name,network.basename(),ipCidrRange,secondaryIpRanges[].rangeName,privateIpGoogleAccess)"
