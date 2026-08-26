# JAX Multi-Node Training on GKE (CPU, GPU, and TPU)

Welcome to the production-ready learning plan and runbook for scaling **JAX multi-node distributed training** across **CPUs, GPUs, and Google Cloud TPUs** using **Google Kubernetes Engine (GKE)** and **Kubernetes JobSet**.

---

## 🌟 What's Included & Key Learnings

1. **Zero-Quota CPU Multi-Node & 10x Scale-Up**:
   - Features a **100% accessible CPU multi-node path** requiring zero GPU/TPU quota.
   - Uses `XLA_FLAGS="--xla_force_host_platform_device_count=4"` to configure 4 virtual devices per host while executing real inter-node multiprocess distribution across Kubernetes VM hosts.
   - Demonstrates **10x scale-up** from **2 Nodes (8 virtual devices)** to **20 Nodes (80 virtual devices)**.

2. **Hardware Accelerator Support (GPU & TPU)**:
   - **GPU Multi-Node (NVIDIA L4)**: Multi-node GPU training across `g2-standard-8` nodes using NVIDIA NCCL inter-node all-reduce over GKE.
   - **TPU Multi-Host Slices (TPU v5e)**: Multi-host TPU slice training on `ct5lp-hightpu-4t` in `2x4` topology (8 total TPU chips) interconnecting hosts over high-speed Inter-Chip Interconnect (ICI).

3. **Mathematical `psum` Gradient All-Reduce Proof**:
   - Validates gradient all-reduce synchronization (`jax.lax.psum`).
   - For $N$ ranks with input rank value $i+1$, synchronized sum equals $\sum_{k=1}^N k = \frac{N(N+1)}{2}$.
   - **2 Ranks (CPU/GPU/TPU)**: Synchronized sum = **`3.0`**.
   - **20 Ranks (CPU 10x Scale-Up)**: Synchronized sum = **`210.0`**.

4. **Kubernetes JobSet & Headless DNS Discovery**:
   - Deterministic headless DNS discovery: `COORDINATOR_ADDRESS="<jobset-name>-workers-0-0.<subdomain>.default.svc.cluster.local:<port>"`.
   - Index tracking via `batch.kubernetes.io/job-completion-index`.
   - **Pod Anti-Affinity**: Forces worker pods onto distinct physical VM machines (`topologyKey: "kubernetes.io/hostname"`).
   - **JobSet v0.12.0+ Native DNS**: JobSet automatically provisions and manages the headless `Service` when `spec.network.subdomain` is declared.

5. **GCP Networking & Security Best Practices**:
   - Custom VPC & Subnet with IP-Aliasing (`pods-range`, `services-range`).
   - Private GKE Nodes with **Cloud NAT & Cloud Router** for outbound internet egress (pulling base images and installing python packages) while preserving private network isolation.

---

## 📋 Prerequisites & GCP Account Setup

Before running the terminal scripts, ensure you have:

1. **Google Cloud Project & Billing**:
   - A GCP Project ID with active billing enabled.
2. **Required IAM Permissions**:
   - `Kubernetes Engine Admin` (`roles/container.admin`)
   - `Artifact Registry Administrator` (`roles/artifactregistry.admin`)
   - `Cloud Build Editor` (`roles/cloudbuild.builds.editor`)
   - `Compute Network Admin` (`roles/compute.networkAdmin`)
   - `Storage Admin` (`roles/storage.admin`)
3. **Authentication & Service Account Permissions**:
   - **Local Terminal**: Run `gcloud auth login` and `gcloud auth application-default login`.
   - **Cloud Build Service Account**: Ensure the Compute default service account (`[PROJECT_NUMBER]-compute@developer.gserviceaccount.com`) has `roles/storage.objectViewer`, `roles/logging.logWriter`, and `roles/artifactregistry.writer` permissions (automatically configured in Step 00 & Step 02).

> [!TIP]
> **Troubleshooting `gcloud builds submit` 403 Permission Error (`could not resolve source ... storage.objects.get`):**
> Newly created GCP projects enforce secure default service account policies where the default Compute Engine service account lacks storage object access. `scripts/00_setup_network.sh` automatically binds `roles/storage.objectViewer` and `roles/logging.logWriter`. If manually required, run:
> ```bash
> PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")
> gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
>     --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
>     --role="roles/storage.objectViewer"
> ```

---

## 📐 System Architecture Overview

For module-by-module architecture diagrams, see **[`ARCHITECTURE.md`](ARCHITECTURE.md)**.

![JAX Multi-Node GKE Architecture Diagram](assets/jax_gke_architecture_diagram.jpg)

---

## 🖥️ Primary CLI Execution Workflow (Standalone Scripts)

The repository provides a complete, standalone, progressive shell script workflow in the [`scripts/`](scripts/) directory. Each script is self-contained, idempotent, and includes educational step-by-step logs and mathematical proofs.

### 1. Configure your GCP Project
Edit [`config.env`](config.env) and set your target `PROJECT_ID`:
```bash
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
ZONE="us-central1-a"
```

### 2. Execute Progressive Shell Scripts

```bash
# ------------------------------------------------------------------------------
# Module 00: Enable GCP APIs, Provision Custom VPC Network & Cloud NAT
# ------------------------------------------------------------------------------
./scripts/00_setup_network.sh

# ------------------------------------------------------------------------------
# Module 01: Create GKE Standard Cluster (Private Nodes) & Install JobSet Operator
# ------------------------------------------------------------------------------
./scripts/01_create_cluster.sh

# ------------------------------------------------------------------------------
# Module 02: Build JAX Container Images via Google Cloud Build
# (Targets: cpu, gpu, tpu, or all)
# ------------------------------------------------------------------------------
./scripts/02_build_image.sh all

# ------------------------------------------------------------------------------
# Module 03: Run CPU Multi-Node Training & 10x Scale-Up Verification (Zero Quota)
#   - Part A: 2 Nodes (8 Virtual Devices)   -> Validates Sum = 3.0
#   - Part B: 20 Nodes (80 Virtual Devices) -> Validates Sum = 210.0
# ------------------------------------------------------------------------------
./scripts/03_run_cpu_multinode.sh --all

# ------------------------------------------------------------------------------
# Module 04: Multi-Node GPU Training (NVIDIA L4 / GPU Node Pool)
# ------------------------------------------------------------------------------
./scripts/04_run_gpu_multinode.sh

# ------------------------------------------------------------------------------
# Module 05: Multi-Host TPU Slice Training (TPU v5e 2x4 Topology)
# ------------------------------------------------------------------------------
./scripts/05_run_tpu_multinode.sh

# ------------------------------------------------------------------------------
# Module 06: Complete Resource Teardown & Cost Cleanup
# ------------------------------------------------------------------------------
./scripts/06_cleanup.sh
```

---

## 💡 Key Architectural Learnings from Live Testing

1. **Private Clusters Require Cloud NAT for Outbound Internet**:
   - Because worker nodes run on private IPs (`--enable-private-nodes`), they cannot communicate directly with the public internet.
   - A Cloud Router and Cloud NAT gateway (`jax-network-nat`) must be attached to the VPC subnet to allow nodes to pull external packages and base images without requiring public IPs.

2. **JobSet v0.12.0+ Automatic Headless DNS**:
   - When specifying `subdomain: <job-name>` under `spec.network`, the Kubernetes JobSet controller automatically instantiates and manages the headless Service. Manual `kind: Service` definitions in manifests are redundant and cause naming conflicts.

3. **Cloud Build IAM Service Account Role Requirements**:
   - The default Compute Engine service account (`[PROJECT_NUMBER]-compute@developer.gserviceaccount.com`) used by Cloud Build requires `roles/storage.objectViewer` to download the uploaded source tarball from the Cloud Build GCS staging bucket and `roles/logging.logWriter` to stream build logs.

4. **Deterministic Coordinator Discovery**:
   - Distributed JAX on Kubernetes requires worker rank 0 to bind to `0.0.0.0:<port>` while all worker ranks connect to `COORDINATOR_ADDRESS="<coordinator-pod-fqdn>:<port>"`.

---

## 📁 Repository Structure

```
.
├── config.env                      # Central environment variables (User customizable)
├── config.py                       # Python module to inspect config.env variables
├── README.md                       # Comprehensive guide, learnings, and runbook
├── ARCHITECTURE.md                 # Detailed visual architecture diagrams for each module
├── assets/                         # Architecture diagram assets & images
│   └── jax_gke_architecture_diagram.jpg
├── scripts/                        # CLI Bash Scripts (Primary standalone execution flow)
│   ├── 00_setup_network.sh         # Step 00: VPC network, Cloud NAT & API setup script
│   ├── 01_create_cluster.sh        # Step 01: GKE cluster & JobSet operator script
│   ├── 02_build_image.sh           # Step 02: Container build via Cloud Build script
│   ├── 03_run_cpu_multinode.sh     # Step 03: CPU multi-node JobSet execution script
│   ├── 04_run_gpu_multinode.sh     # Step 04: GPU multi-node JobSet execution script
│   ├── 05_run_tpu_multinode.sh     # Step 05: TPU multi-host slice JobSet execution script
│   └── 06_cleanup.sh               # Step 06: Complete teardown & cleanup script
├── src/                            # JAX Distributed Scripts & Dockerfiles
│   ├── jax_cpu_test.py             # CPU multi-node JAX script with psum proof
│   ├── jax_gpu_test.py             # GPU multi-node JAX script with NCCL psum proof
│   ├── jax_tpu_test.py             # TPU multi-host JAX script with ICI psum proof
│   ├── Dockerfile.cpu              # JAX CPU container definition
│   ├── Dockerfile.gpu              # CUDA 12 + JAX GPU container definition
│   └── Dockerfile.tpu              # libtpu + JAX TPU container definition
└── manifests/                      # Kubernetes JobSet Manifests
    ├── jobset-cpu.yaml             # CPU multi-node JobSet manifest template
    ├── jobset-gpu.yaml             # GPU multi-node JobSet manifest template
    └── jobset-tpu.yaml             # TPU v5e multi-host slice JobSet manifest template
```

---

## ⚙️ Configurable Variables (`config.env`)

Edit `config.env` to customize your GCP Project ID and cluster settings:

```bash
PROJECT_ID="[ENTER_PROJECT_ID]"
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="jax-distributed-cluster"
NETWORK_NAME="jax-network"
SUBNET_NAME="jax-subnet"
ARTIFACT_REGISTRY_REPO="jax-gke-repo"
CPU_IMAGE_NAME="jax-cpu-multinode"
GPU_IMAGE_NAME="jax-gpu-multinode"
TPU_IMAGE_NAME="jax-tpu-multinode"
IMAGE_TAG="latest"
```

---

## 🔗 Reference Assets & AI Hypercomputer Resources

1. **[Travolta JAX Multi-Node Training Repository](https://github.com/goabego/Travolta_Multinode_Training)**:
   - The core hands-on learning repository containing automated CLI shell scripts, Docker container definitions, and Kubernetes JobSet manifests for scaling distributed JAX workloads across CPUs, GPUs, and TPUs on GKE.

2. **[Google Cloud AI Hypercomputer GPU Recipes](https://github.com/AI-Hypercomputer/gpu-recipes)**:
   - Official Google Cloud repository featuring benchmarked recipes, deployment patterns, and performance optimization guides for LLMs and generative AI workloads using NVIDIA GPUs on GCP.

3. **[Custom AI-Optimized GKE Clusters with A4X Max Documentation](https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x-max)**:
   - Step-by-step Google Cloud guide for creating custom AI-optimized GKE clusters using high-performance A4X Max instances (`a4x-maxgpu-4g-metal`) paired with GPUDirect RDMA.

---

## Questions
Contact -> abrahamgomez@google.com
