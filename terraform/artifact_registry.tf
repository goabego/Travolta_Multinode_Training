# ==============================================================================
# Module 02: Artifact Registry Repository
# ==============================================================================
# Provisions Docker Artifact Registry repository in the disposable GCP project.
# ==============================================================================

resource "google_artifact_registry_repository" "jax_repo" {
  repository_id = var.artifact_registry_repo
  project       = google_project.disposable_project.project_id
  location      = var.region
  format        = "DOCKER"
  description   = "JAX Multi-Node Container Repository"

  depends_on = [
    google_project_service.enabled_services["artifactregistry.googleapis.com"],
  ]
}
