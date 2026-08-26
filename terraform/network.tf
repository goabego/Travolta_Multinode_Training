# ==============================================================================
# Module 00: Custom VPC Network, Subnet & Cloud NAT
# ==============================================================================
# Provisions isolated custom VPC with secondary IP ranges for GKE Pods & Services,
# plus Cloud Router and Cloud NAT for private node internet egress.
# ==============================================================================

# Custom VPC Network
resource "google_compute_network" "custom_vpc" {
  name                    = var.network_name
  project                 = google_project.disposable_project.project_id
  auto_create_subnetworks = false

  depends_on = [
    google_project_service.enabled_services["compute.googleapis.com"],
  ]
}

# Subnet with Secondary Ranges for IP-Aliased GKE Clusters
resource "google_compute_subnetwork" "custom_subnet" {
  name                     = var.subnet_name
  project                  = google_project.disposable_project.project_id
  region                   = var.region
  network                  = google_compute_network.custom_vpc.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = var.pods_range_name
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = var.services_range_name
    ip_cidr_range = var.services_cidr
  }
}

# Cloud Router for Cloud NAT Gateway
resource "google_compute_router" "nat_router" {
  name    = "${var.network_name}-router"
  project = google_project.disposable_project.project_id
  region  = var.region
  network = google_compute_network.custom_vpc.name
}

# Cloud NAT Gateway for Private Node Outbound Internet Access
resource "google_compute_router_nat" "nat_gateway" {
  name                               = "${var.network_name}-nat"
  project                            = google_project.disposable_project.project_id
  router                             = google_compute_router.nat_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
