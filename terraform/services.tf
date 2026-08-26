# ==============================================================================
# Module 00: GCP API Services
# ==============================================================================
# Enables required APIs for GKE, Cloud Build, Artifact Registry, and Compute.
# ==============================================================================

locals {
  required_services = [
    "compute.googleapis.com",
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "orgpolicy.googleapis.com",
  ]
}

resource "google_project_service" "enabled_services" {
  for_each                   = toset(local.required_services)
  project                    = google_project.disposable_project.project_id
  service                    = each.value
  disable_dependent_services = false
  disable_on_destroy         = false
}
