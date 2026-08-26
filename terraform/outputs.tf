output "project_id" {
  description = "The generated disposable GCP Project ID."
  value       = google_project.disposable_project.project_id
}

output "project_number" {
  description = "The numeric project number assigned by Google Cloud."
  value       = google_project.disposable_project.number
}

output "project_name" {
  description = "The display name of the disposable GCP Project."
  value       = google_project.disposable_project.name
}

output "pet_name" {
  description = "The random pet name suffix generated for the project."
  value       = random_pet.project_name.id
}

output "region" {
  description = "The GCP region."
  value       = var.region
}

output "zone" {
  description = "The GCP compute zone."
  value       = var.zone
}

# ==============================================================================
# Module 00: Networking Outputs
# ==============================================================================

output "network_name" {
  description = "The name of the custom VPC network."
  value       = google_compute_network.custom_vpc.name
}

output "network_id" {
  description = "The URI/ID of the custom VPC network."
  value       = google_compute_network.custom_vpc.id
}

output "subnet_name" {
  description = "The name of the custom subnet."
  value       = google_compute_subnetwork.custom_subnet.name
}

output "subnet_id" {
  description = "The URI/ID of the custom subnet."
  value       = google_compute_subnetwork.custom_subnet.id
}

output "pods_range_name" {
  description = "The secondary range name configured for GKE pods."
  value       = var.pods_range_name
}

output "services_range_name" {
  description = "The secondary range name configured for GKE services."
  value       = var.services_range_name
}

output "nat_router_name" {
  description = "The name of the Cloud Router."
  value       = google_compute_router.nat_router.name
}

output "nat_gateway_name" {
  description = "The name of the Cloud NAT gateway."
  value       = google_compute_router_nat.nat_gateway.name
}

# ==============================================================================
# Module 01: GKE Cluster Outputs
# ==============================================================================

output "cluster_name" {
  description = "The name of the GKE Standard cluster."
  value       = google_container_cluster.primary.name
}

output "cluster_id" {
  description = "The resource ID of the GKE cluster."
  value       = google_container_cluster.primary.id
}

output "cluster_endpoint" {
  description = "The IP endpoint of the GKE cluster master."
  value       = google_container_cluster.primary.endpoint
}

output "get_credentials_command" {
  description = "Command to fetch kubectl credentials for the cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --project=${google_project.disposable_project.project_id} --zone=${var.zone}"
}

# ==============================================================================
# Module 02: Artifact Registry Outputs
# ==============================================================================

output "artifact_registry_repo" {
  description = "The name of the Artifact Registry repository."
  value       = google_artifact_registry_repository.jax_repo.name
}

output "artifact_registry_id" {
  description = "The full resource ID of the Artifact Registry repository."
  value       = google_artifact_registry_repository.jax_repo.id
}
