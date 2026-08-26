# ==============================================================================
# Disposable GCP Project Provisioning with Prefix-Pet-Name Convention
# ==============================================================================
# Uses random_pet with keepers to guarantee deterministic, stable project IDs
# across subsequent terraform plan and apply operations.
# ==============================================================================

resource "random_pet" "project_name" {
  length    = var.pet_length
  separator = var.pet_separator

  keepers = {
    # Keepers ensure the generated pet name is preserved across future applies
    # unless the project prefix is explicitly changed.
    project_prefix = var.project_prefix
  }
}

locals {
  # GCP project IDs must be 6-30 characters, lowercase letters, numbers, and hyphens.
  project_id = lower("${var.project_prefix}-${random_pet.project_name.id}")
}

resource "google_project" "disposable_project" {
  name            = local.project_id
  project_id      = local.project_id
  billing_account = var.billing_account

  # If folder_id is provided, the project is created under that folder.
  # Otherwise, falls back to org_id if specified.
  folder_id = var.folder_id
  org_id    = var.folder_id != null ? null : var.org_id

  auto_create_network = var.auto_create_network
  deletion_policy     = var.deletion_policy

  labels = {
    disposable = "true"
    managed_by = "terraform"
  }
}

# ==============================================================================
# Organization Policy Overrides (Project-Level)
# ==============================================================================

# Disables enforcement of constraints/compute.requireShieldedVm at the project level
# as required for accelerator and custom node configurations.
resource "google_project_organization_policy" "disable_require_shielded_vm" {
  count      = var.disable_require_shielded_vm ? 1 : 0
  project    = google_project.disposable_project.project_id
  constraint = "constraints/compute.requireShieldedVm"

  boolean_policy {
    enforced = false
  }
}

# Allows additional custom boolean organization policy constraints to be disabled
resource "google_project_organization_policy" "custom_disabled_policies" {
  for_each   = toset(var.custom_org_policies_to_disable)
  project    = google_project.disposable_project.project_id
  constraint = each.value

  boolean_policy {
    enforced = false
  }
}
