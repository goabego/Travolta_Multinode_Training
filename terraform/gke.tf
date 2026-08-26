# ==============================================================================
# Module 01: Private GKE Standard Cluster, Node Pools & JobSet Controller
# ==============================================================================
# Provisions an enterprise-ready private GKE cluster with Workload Identity,
# IP-aliasing, dedicated node pools, and automatically registers JobSet Operator.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Base GKE Standard Cluster Control Plane
# ------------------------------------------------------------------------------
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  project  = google_project.disposable_project.project_id
  location = var.zone

  # Network & Subnet configuration from Module 00
  network    = google_compute_network.custom_vpc.id
  subnetwork = google_compute_subnetwork.custom_subnet.id

  # IP-Aliasing with dedicated Pod and Service secondary IP ranges
  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_range_name
    services_secondary_range_name = var.services_range_name
  }

  # Workload Identity Pool
  workload_identity_config {
    workload_pool = "${google_project.disposable_project.project_id}.svc.id.goog"
  }

  # Private Cluster Configuration (Private Nodes, Public Master Endpoint)
  private_cluster_config {
    enable_private_nodes   = true
    master_ipv4_cidr_block = var.master_ipv4_cidr_block
  }

  # Separate node pool lifecycle management
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false

  release_channel {
    channel = "REGULAR"
  }

  # Ensure Cloud NAT and APIs are fully provisioned before cluster creation
  depends_on = [
    google_project_service.enabled_services["container.googleapis.com"],
    google_compute_router_nat.nat_gateway,
  ]

  lifecycle {
    ignore_changes = [
      node_pool,
      initial_node_count,
    ]
  }
}

# ------------------------------------------------------------------------------
# 2. Default CPU Node Pool (Zero-Quota Testing)
# ------------------------------------------------------------------------------
resource "google_container_node_pool" "cpu_pool" {
  name       = "default-pool"
  project    = google_project.disposable_project.project_id
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.cpu_node_count

  node_config {
    machine_type = var.cpu_machine_type
    oauth_scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append",
    ]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  lifecycle {
    ignore_changes = [
      node_count,
      initial_node_count,
      node_config[0].oauth_scopes,
      node_config[0].labels,
      node_config[0].tags,
    ]
  }

  # Automatically fetch credentials and install JobSet operator upon node pool creation
  provisioner "local-exec" {
    command = <<-EOT
      echo "▶ Configuring kubectl credentials for cluster '${google_container_cluster.primary.name}'..."
      gcloud container clusters get-credentials "${google_container_cluster.primary.name}" --project="${google_project.disposable_project.project_id}" --zone="${var.zone}"
      
      echo "▶ Installing Kubernetes JobSet Controller (${var.jobset_version})..."
      kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${var.jobset_version}/manifests.yaml"
      
      echo "✅ GKE Cluster and JobSet Controller Ready!"
    EOT
  }
}

# ------------------------------------------------------------------------------
# 3. Optional GPU Node Pool (NVIDIA L4)
# ------------------------------------------------------------------------------
resource "google_container_node_pool" "gpu_pool" {
  count      = var.enable_gpu_pool ? 1 : 0
  name       = var.gpu_node_pool_name
  project    = google_project.disposable_project.project_id
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.gpu_node_count

  node_config {
    machine_type = var.gpu_machine_type
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count_per_node
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    labels = {
      "cloud.google.com/gke-nodepool" = var.gpu_node_pool_name
    }

    tags = ["jax-node", "gpu-node"]
  }
}

# ------------------------------------------------------------------------------
# 4. Optional TPU Node Pool (TPU v5e Slice)
# ------------------------------------------------------------------------------
resource "google_container_node_pool" "tpu_pool" {
  count          = var.enable_tpu_pool ? 1 : 0
  name           = var.tpu_node_pool_name
  project        = google_project.disposable_project.project_id
  location       = var.zone
  node_locations = [var.zone]
  cluster        = google_container_cluster.primary.name
  node_count     = var.tpu_node_count

  node_config {
    machine_type = var.tpu_machine_type
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      "cloud.google.com/gke-nodepool" = var.tpu_node_pool_name
    }

    tags = ["jax-node", "tpu-node"]
  }
}
