# ==============================================================================
# Module 00: IAM Role Bindings for Service Accounts
# ==============================================================================
# Configures required permissions for Cloud Build and Compute Engine default
# service accounts to pull storage sources, write build logs, and push images.
# ==============================================================================

locals {
  compute_sa_roles = [
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/artifactregistry.writer",
  ]
}

# Grant Compute Default SA required roles for Cloud Build operations
resource "google_project_iam_member" "compute_sa_roles" {
  for_each = toset(local.compute_sa_roles)
  project  = google_project.disposable_project.project_id
  role     = each.value
  member   = "serviceAccount:${google_project.disposable_project.number}-compute@developer.gserviceaccount.com"

  depends_on = [
    google_project_service.enabled_services["compute.googleapis.com"],
    google_project_service.enabled_services["cloudbuild.googleapis.com"],
    google_project_service.enabled_services["artifactregistry.googleapis.com"],
  ]
}

# Grant Cloud Build SA builder role
resource "google_project_iam_member" "cloudbuild_sa_builder" {
  project = google_project.disposable_project.project_id
  role    = "roles/cloudbuild.builds.builder"
  member  = "serviceAccount:${google_project.disposable_project.number}@cloudbuild.gserviceaccount.com"

  depends_on = [
    google_project_service.enabled_services["cloudbuild.googleapis.com"],
  ]
}
