SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# ==============================================================================
# Help
# ==============================================================================
.PHONY: help
help: ## Show this help message
	@echo "=============================================================================="
	@echo "  Travolta Multi-Node Training - Makefile"
	@echo "=============================================================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mUsage:\033[0m make \033[36m<target>\033[0m\n"} \
		/^[a-zA-Z0-9_.-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ ⚙️  Configuration & Utilities
.PHONY: config
config: ## Display active configuration variables from config.env
	@python3 config.py

.PHONY: sync-config
sync-config: ## Sync PROJECT_ID in config.env from Terraform output
	@PROJECT_ID=$$(terraform -chdir=terraform output -raw project_id 2>/dev/null); \
	if [ -n "$$PROJECT_ID" ]; then \
		sed -i 's/^PROJECT_ID=.*/PROJECT_ID="'"$$PROJECT_ID"'"/' config.env; \
		echo "✅ Synced config.env with PROJECT_ID: $$PROJECT_ID"; \
	else \
		echo "❌ Could not retrieve project_id from terraform output. Run 'make tf-apply' first."; \
	fi

.PHONY: tf-fmt
tf-fmt: ## Format all Terraform configuration files
	terraform fmt -recursive terraform/

##@ 🏗️  Infrastructure (Terraform)
.PHONY: tf-init
tf-init: ## Initialize Terraform in the terraform/ directory
	terraform -chdir=terraform init

.PHONY: tf-plan
tf-plan: ## Generate and show Terraform execution plan
	terraform -chdir=terraform plan

.PHONY: tf-apply
tf-apply: ## Provision infrastructure (project, VPC, NAT, GKE, Repo)
	terraform -chdir=terraform apply

.PHONY: tf-creds
tf-creds: ## Authenticate kubectl against the Terraform GKE cluster
	@CMD=$$(terraform -chdir=terraform output -raw get_credentials_command 2>/dev/null); \
	if [ -n "$$CMD" ]; then \
		echo "Executing: $$CMD"; \
		$$CMD; \
	else \
		echo "❌ Could not retrieve credentials command. Run 'make tf-apply' first."; \
	fi

.PHONY: tf-output
tf-output: ## Show Terraform output values (project_id, endpoints, etc.)
	terraform -chdir=terraform output

.PHONY: tf-destroy
tf-destroy: ## Destroy Terraform-managed infrastructure
	terraform -chdir=terraform destroy

##@ 🚀 Multi-Node Workflow Modules (Numbered Steps)
.PHONY: 00-setup-network setup-network
00-setup-network: ## [Step 00] Enable GCP APIs, VPC network & Cloud NAT
	./scripts/00_setup_network.sh
setup-network: 00-setup-network

.PHONY: 01-create-cluster create-cluster
01-create-cluster: ## [Step 01] Provision GKE cluster & install JobSet operator
	./scripts/01_create_cluster.sh
create-cluster: 01-create-cluster

.PHONY: 01-create-cluster-all
01-create-cluster-all: ## [Step 01] Provision GKE cluster with CPU, GPU & TPU node pools
	./scripts/01_create_cluster.sh --all

.PHONY: 02-build-images build-images
02-build-images: ## [Step 02] Build all JAX container images (CPU, GPU, TPU) via Cloud Build
	./scripts/02_build_image.sh all
build-images: 02-build-images

.PHONY: 02-build-cpu build-cpu
02-build-cpu: ## [Step 02] Build CPU container image via Cloud Build
	./scripts/02_build_image.sh cpu
build-cpu: 02-build-cpu

.PHONY: 02-build-gpu build-gpu
02-build-gpu: ## [Step 02] Build GPU container image via Cloud Build
	./scripts/02_build_image.sh gpu
build-gpu: 02-build-gpu

.PHONY: 02-build-tpu build-tpu
02-build-tpu: ## [Step 02] Build TPU container image via Cloud Build
	./scripts/02_build_image.sh tpu
build-tpu: 02-build-tpu

.PHONY: 03-run-cpu run-cpu
03-run-cpu: ## [Step 03] Run CPU multi-node training & 10x scale-up (Parts A & B)
	./scripts/03_run_cpu_multinode.sh --all
run-cpu: 03-run-cpu

.PHONY: 03-run-cpu-part-a run-cpu-part-a
03-run-cpu-part-a: ## [Step 03] Run CPU 2-node baseline (8 virtual devices)
	./scripts/03_run_cpu_multinode.sh --part-a
run-cpu-part-a: 03-run-cpu-part-a

.PHONY: 03-run-cpu-part-b run-cpu-part-b
03-run-cpu-part-b: ## [Step 03] Run CPU 20-node scale-up (80 virtual devices)
	./scripts/03_run_cpu_multinode.sh --part-b
run-cpu-part-b: 03-run-cpu-part-b

.PHONY: 03b-train-toy-model train-toy-model 03b-train
03b-train-toy-model: ## [Step 03b] Train toy neural network on CPU multi-node cluster
	./scripts/03b_run_cpu_model_training.sh
train-toy-model: 03b-train-toy-model
03b-train: 03b-train-toy-model

.PHONY: 04-run-gpu run-gpu
04-run-gpu: ## [Step 04] Run GPU multi-node training on NVIDIA L4 (NCCL all-reduce)
	./scripts/04_run_gpu_multinode.sh
run-gpu: 04-run-gpu

.PHONY: 05-run-tpu run-tpu
05-run-tpu: ## [Step 05] Run TPU multi-host slice training on TPU v5e (ICI all-reduce)
	./scripts/05_run_tpu_multinode.sh
run-tpu: 05-run-tpu

.PHONY: 06-cleanup cleanup
06-cleanup: ## [Step 06] Teardown all GKE and GCP resources
	./scripts/06_cleanup.sh
cleanup: 06-cleanup

##@ 🔍 Monitoring & Diagnostics
.PHONY: status
status: ## View GKE cluster nodes, JobSets, and active worker pods
	@echo "=== GKE Nodes ==="
	@kubectl get nodes -o wide || true
	@echo -e "\n=== JobSets ==="
	@kubectl get jobset || true
	@echo -e "\n=== Worker Pods ==="
	@kubectl get pods -o wide || true

.PHONY: logs-cpu
logs-cpu: ## Stream logs from active CPU worker pods
	kubectl logs -l jobset.x-k8s.io/jobset-name=jax-cpu-job --all-containers -f || true

.PHONY: logs-toy-model logs-cpu-train
logs-toy-model: ## Stream logs from CPU toy model training worker pods
	kubectl logs -l jobset.x-k8s.io/jobset-name=jax-cpu-train-job --all-containers -f || true
logs-cpu-train: logs-toy-model

.PHONY: logs-gpu
logs-gpu: ## Stream logs from active GPU worker pods
	kubectl logs -l jobset.x-k8s.io/jobset-name=jax-gpu-job --all-containers -f || true

.PHONY: logs-tpu
logs-tpu: ## Stream logs from active TPU worker pods
	kubectl logs -l jobset.x-k8s.io/jobset-name=jax-tpu-job --all-containers -f || true
