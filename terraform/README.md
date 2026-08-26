# Multi-Node Training Terraform Module

This Terraform module provisions the entire foundation for **JAX Multi-Node Distributed Training on GKE**:
1. **Disposable GCP Project** with a **prefix-pet-name naming convention** (e.g., `travolta-dashing-lemming`) and keeper stability.
2. **Project-Level Org Policy Overrides** (`constraints/compute.requireShieldedVm = false`).
3. **Core GCP APIs & IAM Service Account Roles**.
4. **Custom VPC & Subnet** with secondary IP ranges for GKE Pods (`10.4.0.0/14`) and Services (`10.8.0.0/20`).
5. **Cloud Router & Cloud NAT** for secure outbound internet egress from private nodes.
6. **Artifact Registry Repository** (`jax-gke-repo`) for Docker container images.
7. **Private GKE Standard Cluster & CPU Node Pool** with Workload Identity.
8. **Automated Bootstrapping**: Runs `gcloud container clusters get-credentials` and installs the **Kubernetes JobSet Operator (v0.12.0)** on apply.

---

## 🔑 Key Architectural Features

1. **Deterministic Pet Name via Keepers**:
   ```hcl
   resource "random_pet" "project_name" {
     length    = var.pet_length
     separator = var.pet_separator

     keepers = {
       project_prefix = var.project_prefix
     }
   }
   ```
   - **Idempotent Future Applies**: Preserves the generated pet name across subsequent `terraform plan` and `terraform apply` runs.
   - **Triggered Recreation**: Modifying `var.project_prefix` updates the keeper, triggering generation of a new project.

2. **Enterprise Organization & Folder Support**:
   - Setting `folder_id = "YOUR_FOLDER_ID"` directs project creation to a sandbox or team folder where you have Project Creator permissions.
   - Alternatively, set `org_id = "YOUR_ORG_ID"` if you have root org project creation privileges.

3. **Project-Level Organization Policy Overrides**:
   - Disables `constraints/compute.requireShieldedVm` at the project level via `google_project_organization_policy`, enabling seamless node pool provisioning.

4. **Zero-Quota GKE Control Plane & Node Pools**:
   - Provisions a private GKE cluster in `var.zone` (`us-central1-a`).
   - Default CPU pool creates `e2-standard-4` nodes that host 4 virtual XLA devices each.
   - Optional GPU (`var.enable_gpu_pool`) and TPU (`var.enable_tpu_pool`) flags for accelerator testing.

---

## 🚀 Quickstart

1. **Copy example variables**:
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   ```

2. **Set your configuration in `terraform.tfvars`**:
   ```hcl
   project_prefix  = "travolta"
   billing_account = "01C57A-D0DCD2-49BE8B"
   org_id          = "101096319108"
   # folder_id     = "YOUR_FOLDER_ID"  # Use if creating under a specific folder
   region          = "us-central1"
   zone            = "us-central1-a"
   ```

3. **Initialize and apply**:
   ```bash
   make tf-init
   make tf-plan
   make tf-apply
   make sync-config
   ```

4. **Teardown**:
   ```bash
   make tf-destroy
   ```
