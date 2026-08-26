# 🧪 JAX Multi-Node GKE: CPU Testing Runbook (Modules 00 - 06)

This runbook guides you through testing the **zero-quota CPU multi-node workflow** and **toy model training** on Google Cloud GKE from start to finish. It requires **no GPU or TPU quotas**.

---

## 🛠️ 1. Prerequisites

1. **Google Cloud SDK (`gcloud`)** and **`kubectl`** installed and authenticated:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
2. A GCP project or folder with **billing enabled** where you have project creation/admin roles.
3. Billing Account ID (e.g. `01C57A-D0DCD2-49BE8B`).

---

## 🚀 2. Two Execution Paths

### 🅰️ Path A: Fast Terraform + Make Path (Recommended)

1. **Configure `terraform/terraform.tfvars`**:
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   ```
   Set `billing_account`, `org_id` (or `folder_id`).

2. **Provision Everything**:
   ```bash
   make tf-init
   make tf-apply
   make sync-config
   ```
   *(Provisions disposable project, sets org policy overrides, VPC, Cloud NAT, Artifact Registry, GKE cluster, and registers the JobSet operator).*

3. **Build CPU Image**:
   ```bash
   make 02-build-cpu
   ```

4. **Run Multi-Node Tests**:
   ```bash
   # All-reduce proof (2 nodes & 20 nodes)
   make 03-run-cpu

   # Distributed toy neural network training
   make 03b-train-toy-model
   ```

5. **Teardown**:
   ```bash
   make tf-destroy
   ```

---

### 🅱️ Path B: Standalone Shell Scripts (Step-by-Step)

1. **Set `config.env`**:
   ```bash
   PROJECT_ID="your-gcp-project-id"
   REGION="us-central1"
   ZONE="us-central1-a"
   ```

2. **Step 00: VPC Network, Subnet & Cloud NAT**:
   ```bash
   ./scripts/00_setup_network.sh
   ```
   *Expected Output*: `✅ Module 00 Setup Complete!`

3. **Step 01: GKE Cluster & JobSet Operator**:
   ```bash
   ./scripts/01_create_cluster.sh
   ```
   *Expected Output*: `✅ Module 01 GKE Cluster & Operator Setup Complete!`

4. **Step 02: Build CPU Container Image**:
   ```bash
   ./scripts/02_build_image.sh cpu
   ```
   *Expected Output*: `STATUS: SUCCESS` and image tag `jax-cpu-multinode:latest`.

5. **Step 03: CPU Multi-Node All-Reduce & 10x Scale-Up**:
   ```bash
   ./scripts/03_run_cpu_multinode.sh --all
   ```
   *Expected Verification*:
   - **Part A (2 Nodes / 8 Virtual Devices)**: `Synchronized all-reduce Output: 3.0 (Expected: 3.0)` $\rightarrow$ `✅ MATHEMATICAL VERIFICATION PASSED!`
   - **Part B (20 Nodes / 80 Virtual Devices)**: `Synchronized all-reduce Output: 210.0 (Expected: 210.0)` $\rightarrow$ `✅ MATHEMATICAL VERIFICATION PASSED!`

6. **Step 03b: CPU Multi-Node Toy Model Training**:
   ```bash
   ./scripts/03b_run_cpu_model_training.sh
   ```
   *Expected Verification*:
   - Loss decreases across 10 epochs (>70% loss reduction) $\rightarrow$ `✅ MATHEMATICAL CONVERGENCE VERIFIED!`

7. **Step 06: Cost Teardown & Resource Cleanup**:
   ```bash
   ./scripts/06_cleanup.sh
   ```
   *Expected Output*: `✅ CLEANUP COMPLETE: All created resources have been removed.`

---

## 📝 3. Reporting Feedback

If you encounter any errors, please copy and reply with:
1. **Failing Script / Step** (e.g., `make 01-create-cluster` or `01_create_cluster.sh`)
2. **Error message or terminal output**
3. **Your GCP Region & Zone**
