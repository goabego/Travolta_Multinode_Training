variable "project_prefix" {
  description = "Prefix for the disposable GCP project name and ID (e.g., 'travolta' or 'jax-dev')."
  type        = string
  default     = "travolta"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,15}$", var.project_prefix))
    error_message = "Project prefix must start with a lowercase letter, contain only lowercase letters, digits, or hyphens, and be between 2 and 16 characters."
  }
}

variable "pet_length" {
  description = "The number of words in the generated random pet name."
  type        = number
  default     = 2
}

variable "pet_separator" {
  description = "The separator character used in the random pet name."
  type        = string
  default     = "-"
}

variable "billing_account" {
  description = "The GCP billing account ID to associate with the project (e.g., '012345-6789AB-CDEF01')."
  type        = string
  default     = null
}

variable "folder_id" {
  description = "The numeric GCP Folder ID to create the project under (e.g. '123456789012'). Required if you lack project creation permissions at the root organization level."
  type        = string
  default     = null
}

variable "org_id" {
  description = "The numeric GCP Organization ID (e.g., '433637338589'). Only use if you have 'resourcemanager.projects.create' at the root org level. If folder_id is set, folder_id takes precedence."
  type        = string
  default     = null
}

variable "region" {
  description = "The default Google Cloud region for provider operations."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The default Google Cloud compute zone for GKE cluster and node pools."
  type        = string
  default     = "us-central1-a"
}

variable "auto_create_network" {
  description = "Whether to create the default network. Recommended false for custom networking."
  type        = bool
  default     = false
}

variable "deletion_policy" {
  description = "The deletion policy for the project. 'DELETE' ensures complete cleanup upon destroy."
  type        = string
  default     = "DELETE"
}

# ==============================================================================
# Organization Policy Overrides
# ==============================================================================

variable "disable_require_shielded_vm" {
  description = "Whether to disable 'constraints/compute.requireShieldedVm' org policy enforcement at the project level."
  type        = bool
  default     = true
}

variable "custom_org_policies_to_disable" {
  description = "List of additional boolean org policy constraint names to disable (enforce = false) on the created project."
  type        = list(string)
  default     = []
}

# ==============================================================================
# Module 00: Networking Configuration
# ==============================================================================

variable "network_name" {
  description = "The name of the custom VPC network."
  type        = string
  default     = "jax-network"
}

variable "subnet_name" {
  description = "The name of the custom subnet."
  type        = string
  default     = "jax-subnet"
}

variable "subnet_cidr" {
  description = "The primary CIDR range for the GKE nodes subnet."
  type        = string
  default     = "10.0.0.0/20"
}

variable "pods_range_name" {
  description = "The secondary range name for GKE Pods."
  type        = string
  default     = "pods-range"
}

variable "pods_cidr" {
  description = "The secondary CIDR range for GKE Pods."
  type        = string
  default     = "10.4.0.0/14"
}

variable "services_range_name" {
  description = "The secondary range name for GKE Services."
  type        = string
  default     = "services-range"
}

variable "services_cidr" {
  description = "The secondary CIDR range for GKE Services."
  type        = string
  default     = "10.8.0.0/20"
}

# ==============================================================================
# Module 01: GKE Cluster & Node Pools Configuration
# ==============================================================================

variable "cluster_name" {
  description = "The name of the GKE Standard cluster."
  type        = string
  default     = "jax-distributed-cluster"
}

variable "master_ipv4_cidr_block" {
  description = "The CIDR block for the GKE master control plane network."
  type        = string
  default     = "172.16.0.0/28"
}

variable "cpu_node_count" {
  description = "The number of nodes in the default CPU node pool."
  type        = number
  default     = 2
}

variable "cpu_machine_type" {
  description = "The machine type for the CPU node pool."
  type        = string
  default     = "e2-standard-4"
}

variable "enable_gpu_pool" {
  description = "Whether to provision the GPU (NVIDIA L4) node pool."
  type        = bool
  default     = false
}

variable "gpu_node_pool_name" {
  description = "The name of the GPU node pool."
  type        = string
  default     = "gpu-pool"
}

variable "gpu_machine_type" {
  description = "The machine type for GPU nodes."
  type        = string
  default     = "g2-standard-8"
}

variable "gpu_type" {
  description = "The GPU accelerator type."
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count_per_node" {
  description = "The number of GPUs per node."
  type        = number
  default     = 1
}

variable "gpu_node_count" {
  description = "The number of nodes in the GPU pool."
  type        = number
  default     = 2
}

variable "enable_tpu_pool" {
  description = "Whether to provision the TPU (v5e) node pool."
  type        = bool
  default     = false
}

variable "tpu_node_pool_name" {
  description = "The name of the TPU node pool."
  type        = string
  default     = "tpu-v5e-pool"
}

variable "tpu_machine_type" {
  description = "The machine type for TPU nodes."
  type        = string
  default     = "ct5lp-hightpu-4t"
}

variable "tpu_topology" {
  description = "The TPU slice topology (e.g. '2x4')."
  type        = string
  default     = "2x4"
}

variable "tpu_node_count" {
  description = "The number of nodes in the TPU pool."
  type        = number
  default     = 2
}

variable "jobset_version" {
  description = "The version tag of the Kubernetes JobSet controller release."
  type        = string
  default     = "v0.12.0"
}

# ==============================================================================
# Module 02: Artifact Registry Configuration
# ==============================================================================

variable "artifact_registry_repo" {
  description = "The name of the Artifact Registry Docker repository."
  type        = string
  default     = "jax-gke-repo"
}
