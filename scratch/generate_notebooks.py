import json
import os
from pathlib import Path

notebooks_dir = Path("/Users/abrahamgomez/travolta/notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def markdown_cell(source):
    # Strip leading/trailing blank newlines and format lines cleanly without trailing empty lines
    lines = source.strip("\n").splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code_cell(source):
    # Strip leading/trailing blank newlines and format lines cleanly without trailing empty lines
    lines = source.strip("\n").splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

# ---------------------------------------------------------
# Notebook 00: Config, Install, Auth & Custom VPC Setup
# ---------------------------------------------------------
nb0_cells = [
    markdown_cell("""# Module 00: GCP Environment, Authentication & Custom VPC Setup
Welcome to the JAX Multi-Node Training on GKE Learning Plan!

### Learning Objectives:
1. Install essential GCP, GKE, and JAX dependencies.
2. Authenticate Google Colab against Google Cloud (Application Default Credentials).
3. Load and inspect central environment variables (`PROJECT_ID`, `REGION`, `ZONE`, `CLUSTER_NAME`, etc.).
4. Provision custom VPC Network and Subnet with IP-Aliasing (`pods-range`, `services-range`)."""),
    markdown_cell("## Install Essential Dependencies"),
    code_cell("""# Essential GCP, GKE, and JAX Multi-Node Dependencies
!pip install -U -q google-cloud-storage
!pip install -U -q google-cloud-container
!pip install -U -q google-auth
!pip install -U -q kubernetes
!pip install -U -q requests
!pip install -U -q jax jaxlib

# Optional: Vertex AI & Gemini SDKs (for Vertex AI tracking)
!pip install -U -q google-genai
!pip install -U -q google-cloud-aiplatform"""),
    markdown_cell("""## Restart the Colab Session

To use the newly installed packages in this Colab runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.

*Technically optional since Colab should prompt to restart the session after installing the dependencies above.*"""),
    code_cell("""import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)"""),
    markdown_cell("""## Authenticate with Google Cloud (Colab Only)
Google Colab projects can authenticate against Google Cloud via the following calls. It is not required if running on Colab Enterprise or Vertex AI Workbench.

Running the cell below sets up the "[Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc)" which are used by our SDKs to automatically authenticate against Google Cloud Services.

In short, this is equivalent to the following gcloud CLI commands:
```bash
$ gcloud auth login
$ gcloud auth application-default login
```

An authentication pop-up will appear, please accept the permissions before proceeding.

### Alternative: Using Gemini API Keys
Some of these code samples will only work with a Google Cloud account, but the basic Gemini SDK will also work via an API key that can be obtained from [Google AI Studio](https://aistudio.google.com).

**Steps:**
1. Get an API key from: https://aistudio.google.com/app/apikey
2. Create a Colab "secret" called `AI_STUDIO_API_KEY` in the "Secrets" tab on the left hand side of Colab.
3. Make sure that `PROJECT_ID` *is not* defined. This ensures that AI Studio's API key will be used instead."""),
    code_cell("""from google.colab import userdata
from google.colab import auth
import sys

# Leave PROJECT_ID Blank to use an API Key instead
PROJECT_ID = "cloud-llm-preview1" # @param {type: "string"}
LOCATION = "us-central1" # @param {type: "string"}
# BUCKET = "" # @param{type:"string"}

if "google.colab" in sys.modules and PROJECT_ID != "":
    from google.colab import auth
    auth.authenticate_user(project_id=PROJECT_ID)"""),
    markdown_cell("## Enable Google Cloud Services"),
    code_cell("""# !gcloud services enable \\
#   cloudresourcemanager.googleapis.com \\
#   aiplatform.googleapis.com \\
#   documentai.googleapis.com \\
#   notebooks.googleapis.com \\
#   visionai.googleapis.com \\
#   storage-component.googleapis.com \\
#   cloudaicompanion.googleapis.com \\
#   discoveryengine.googleapis.com \\
#   --project {PROJECT_ID}"""),
    markdown_cell("## Load Central Configuration Variables"),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import config
cfg = config.load_config("../config.env")
config.print_config()"""),
    markdown_cell("## Configure `gcloud` CLI & Active Project"),
    code_cell("""import os

PROJECT_ID = cfg.get("PROJECT_ID", "travolta-505921")
REGION = cfg.get("REGION", "us-central1")
ZONE = cfg.get("ZONE", "us-central1-a")
NETWORK_NAME = cfg.get("NETWORK_NAME", "jax-network")
SUBNET_NAME = cfg.get("SUBNET_NAME", "jax-subnet")

!gcloud config set project {PROJECT_ID}
!gcloud config set compute/region {REGION}
!gcloud config set compute/zone {ZONE}"""),
    code_cell("""print("Enabling required GKE & Container APIs...")
!gcloud services enable \\
    container.googleapis.com \\
    artifactregistry.googleapis.com \\
    cloudbuild.googleapis.com \\
    iam.googleapis.com \\
    compute.googleapis.com"""),
    markdown_cell("## Configure IAM Permissions for Cloud Build Service Account"),
    code_cell("""# Grant Cloud Build & GCS Storage permissions to the Compute Engine Default Service Account
!PROJECT_NUMBER=$(gcloud projects describe {PROJECT_ID} --format="value(projectNumber)") && \\
 gcloud projects add-iam-policy-binding {PROJECT_ID} \\
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \\
    --role="roles/storage.objectViewer" && \\
 gcloud projects add-iam-policy-binding {PROJECT_ID} \\
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \\
    --role="roles/logs.writer" && \\
 gcloud projects add-iam-policy-binding {PROJECT_ID} \\
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \\
    --role="roles/artifactregistry.writer"
"""),
    markdown_cell("## Provision Custom VPC & Subnet with IP Aliasing"),
    code_cell("""!gcloud compute networks create {NETWORK_NAME} --subnet-mode=custom"""),
    code_cell("""!gcloud compute networks subnets create {SUBNET_NAME} \\
    --network={NETWORK_NAME} \\
    --region={REGION} \\
    --range=10.0.0.0/20 \\
    --secondary-range=pods-range=10.4.0.0/14,services-range=10.8.0.0/20"""),
    markdown_cell("## Verify GCP Credentials & VPC Status"),
    code_cell("""!gcloud compute networks subnets describe {SUBNET_NAME} --region={REGION}""")
]

# ---------------------------------------------------------
# Notebook 01: GKE Cluster Setup (Shielded Nodes + JobSet)
# ---------------------------------------------------------
nb1_cells = [
    markdown_cell("""# Module 01: GKE Standard Cluster & Accelerator Node Pools Setup
In this module, you will provision a GKE Standard Cluster with Shielded VM security flags and custom VPC networking.

### Learning Objectives:
1. Create a GKE Standard Cluster with Workload Identity, Shielded VM, and IP-Aliasing enabled.
2. Verify default CPU node pool (used for zero-quota multi-node CPU testing).
3. Optional: Add dedicated GPU Node Pool (NVIDIA L4/A100) or TPU Node Pool (v5e 2x4 slice).
4. Install Kubernetes JobSet Operator (`jobset.x-k8s.io`)."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import config
cfg = config.load_config("../config.env")

PROJECT_ID = cfg["PROJECT_ID"]
REGION = cfg["REGION"]
ZONE = cfg["ZONE"]
CLUSTER_NAME = cfg["CLUSTER_NAME"]
NETWORK_NAME = cfg["NETWORK_NAME"]
SUBNET_NAME = cfg["SUBNET_NAME"]

GPU_NODE_POOL = cfg["GPU_NODE_POOL_NAME"]
GPU_MACHINE = cfg["GPU_MACHINE_TYPE"]
GPU_TYPE = cfg["GPU_TYPE"]
GPU_COUNT = cfg["GPU_COUNT_PER_NODE"]
GPU_NODES = cfg["GPU_NODE_COUNT"]

TPU_NODE_POOL = cfg["TPU_NODE_POOL_NAME"]
TPU_MACHINE = cfg["TPU_MACHINE_TYPE"]
TPU_TOPOLOGY = cfg["TPU_TOPOLOGY"]
TPU_NODES = cfg["TPU_NODE_COUNT"]

JOBSET_VERSION = cfg["JOBSET_VERSION"]"""),
    markdown_cell("## 1. Create Base GKE Cluster with Shielded Nodes & IP-Alias"),
    code_cell("""!gcloud container clusters create {CLUSTER_NAME} \\
    --zone={ZONE} \\
    --release-channel=regular \\
    --workload-pool={PROJECT_ID}.svc.id.goog \\
    --enable-ip-alias \\
    --network={NETWORK_NAME} \\
    --subnetwork={SUBNET_NAME} \\
    --cluster-secondary-range-name=pods-range \\
    --services-secondary-range-name=services-range \\
    --enable-shielded-nodes \\
    --shielded-secure-boot \\
    --shielded-integrity-monitoring \\
    --num-nodes=2 \\
    --machine-type=e2-standard-4"""),
    code_cell("""!gcloud container clusters get-credentials {CLUSTER_NAME} --zone={ZONE}"""),
    markdown_cell("## 2. Install JobSet Operator"),
    code_cell("""print("Installing Kubernetes JobSet Controller...")
!kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/{JOBSET_VERSION}/manifests.yaml"""),
    markdown_cell("## 3. Optional: Provision GPU Node Pool (If GPU Quota Available)"),
    code_cell("""# Uncomment below to provision GPU node pool:
# !gcloud container node-pools create {GPU_NODE_POOL} \\
#     --cluster={CLUSTER_NAME} \\
#     --zone={ZONE} \\
#     --machine-type={GPU_MACHINE} \\
#     --accelerator=type={GPU_TYPE},count={GPU_COUNT} \\
#     --num-nodes={GPU_NODES} \\
#     --shielded-secure-boot \\
#     --shielded-integrity-monitoring

# !kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/ubuntu/daemonset-unified.yaml"""),
    markdown_cell("## 4. Optional: Provision TPU Node Pool (If TPU Quota Available)"),
    code_cell("""# Uncomment below to provision TPU v5e slice pool:
# !gcloud container node-pools create {TPU_NODE_POOL} \\
#     --cluster={CLUSTER_NAME} \\
#     --zone={ZONE} \\
#     --node-locations={ZONE} \\
#     --machine-type={TPU_MACHINE} \\
#     --tpu-topology={TPU_TOPOLOGY} \\
#     --shielded-secure-boot \\
#     --shielded-integrity-monitoring \\
#     --num-nodes={TPU_NODES}"""),
    markdown_cell("## 5. Verify Cluster Readiness"),
    code_cell("""!kubectl get nodes -o wide
!kubectl get crds | grep jobsets""")
]

# ---------------------------------------------------------
# Notebook 02: Container Build (CPU, GPU, TPU)
# ---------------------------------------------------------
nb2_cells = [
    markdown_cell("""# Module 02: Container Build & Artifact Registry Setup
In this module, you will set up Google Artifact Registry and build Docker container images for CPU, GPU, and TPU JAX workloads using Google Cloud Build.

### Learning Objectives:
1. Create a Docker Artifact Registry repository in GCP.
2. Build the zero-quota **CPU JAX image** (`src/Dockerfile.cpu`).
3. Build the **GPU JAX image** (`src/Dockerfile.gpu`) and **TPU JAX image** (`src/Dockerfile.tpu`)."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import config
cfg = config.load_config("../config.env")

PROJECT_ID = cfg["PROJECT_ID"]
REGION = cfg["REGION"]
REPO = cfg["ARTIFACT_REGISTRY_REPO"]
CPU_IMAGE = cfg["CPU_IMAGE_NAME"]
GPU_IMAGE = cfg["GPU_IMAGE_NAME"]
TPU_IMAGE = cfg["TPU_IMAGE_NAME"]
TAG = cfg["IMAGE_TAG"]

CPU_FULL_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/{CPU_IMAGE}:{TAG}"
GPU_FULL_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/{GPU_IMAGE}:{TAG}"
TPU_FULL_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/{TPU_IMAGE}:{TAG}"

print(f"CPU Image: {CPU_FULL_IMAGE}")
print(f"GPU Image: {GPU_FULL_IMAGE}")
print(f"TPU Image: {TPU_FULL_IMAGE}")"""),
    markdown_cell("## 1. Create Google Artifact Registry Repository"),
    code_cell("""!gcloud artifacts repositories create {REPO} \\
    --repository-format=docker \\
    --location={REGION} \\
    --description="JAX Multi-Node Container Repository\""""),
    code_cell("""!gcloud auth configure-docker {REGION}-docker.pkg.dev --quiet"""),
    markdown_cell("## 2. Build JAX CPU Container Image (Primary Zero-Quota Image)"),
    code_cell("""!gcloud builds submit ../src/ \\
    --config=- <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', '{CPU_FULL_IMAGE}', '-f', 'Dockerfile.cpu', '.']
images:
- '{CPU_FULL_IMAGE}'
EOF"""),
    markdown_cell("## 3. Build JAX GPU & TPU Container Images"),
    code_cell("""!gcloud builds submit ../src/ \\
    --config=- <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', '{GPU_FULL_IMAGE}', '-f', 'Dockerfile.gpu', '.']
images:
- '{GPU_FULL_IMAGE}'
EOF"""),
    code_cell("""!gcloud builds submit ../src/ \\
    --config=- <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', '{TPU_FULL_IMAGE}', '-f', 'Dockerfile.tpu', '.']
images:
- '{TPU_FULL_IMAGE}'
EOF"""),
    markdown_cell("## 4. Verify Built Images in Artifact Registry"),
    code_cell("""!gcloud artifacts docker images list {REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}""")
]

# ---------------------------------------------------------
# Notebook 03: JAX CPU Multi-Node Training & 10x Scale-Up
# ---------------------------------------------------------
nb3_cells = [
    markdown_cell(r"""# Module 03: Multi-Node JAX CPU Training & 10x Scale-Up
Because GPU/TPU quota can be difficult to acquire, this notebook provides a **100% accessible, zero-quota CPU multi-node path**.

Using `--xla_force_host_platform_device_count=4`, JAX simulates 4 virtual accelerator devices per CPU host while executing true inter-node multiprocess distribution across Kubernetes nodes!

### Key Concepts Covered:
1. **Pod Anti-Affinity**: Forcing Kubernetes to schedule worker pods on separate physical VM nodes (`topologyKey: "kubernetes.io/hostname"`).
2. **Headless Discovery & Batch Indexing**: Headless DNS discovery (`COORDINATOR_ADDRESS="jax-cpu-job-workers-0-0.jax-cpu-job:1234"`) and `batch.kubernetes.io/job-completion-index`.
3. **Mathematical Proof (`psum`)**: Validating gradient all-reduce synchronization mathematically.
   - For $N$ ranks with input $i+1$, expected sum is $\sum_{k=1}^N k = \frac{N(N+1)}{2}$.
   - 2 Nodes ($N=2$): Expected sum = **`3.0`** across 8 virtual devices.
   - 20 Nodes ($N=20$): Expected sum = **`210.0`** across 80 virtual devices.
4. **10x Scale-Up Experiment**: Resizing GKE cluster to 20 nodes and scaling JobSet to 20 pods in parallel."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import config
cfg = config.load_config("../config.env")

PROJECT_ID = cfg["PROJECT_ID"]
REGION = cfg["REGION"]
ZONE = cfg["ZONE"]
CLUSTER_NAME = cfg["CLUSTER_NAME"]
REPO = cfg["ARTIFACT_REGISTRY_REPO"]
CPU_IMAGE = cfg["CPU_IMAGE_NAME"]
TAG = cfg["IMAGE_TAG"]

CPU_FULL_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/{CPU_IMAGE}:{TAG}"
print(f"Target CPU Image: {CPU_FULL_IMAGE}")"""),
    markdown_cell("## PART A: 2-Node CPU Multi-Node Experiment (8 Virtual Devices)"),
    markdown_cell("### 1. Render 2-Node CPU JobSet Manifest"),
    code_cell("""manifest_path = Path("../manifests/jobset-cpu.yaml")
rendered_path = Path("../manifests/jobset-cpu-rendered.yaml")

with open(manifest_path, "r") as f:
    content = f.read()

content = content.replace("LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/CPU_IMAGE_NAME:IMAGE_TAG", CPU_FULL_IMAGE)

with open(rendered_path, "w") as f:
    f.write(content)

print(f"Saved rendered JobSet manifest to {rendered_path}")"""),
    markdown_cell("### 2. Deploy 2-Node JobSet & Verify Pod Anti-Affinity Placement"),
    code_cell("""!kubectl apply -f ../manifests/jobset-cpu-rendered.yaml"""),
    code_cell("""import time
print("Waiting for worker pods to start...")
for _ in range(6):
    !kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-cpu-job -o custom-columns=POD_NAME:.metadata.name,POD_IP:.status.podIP,NODE:.spec.nodeName,STATUS:.status.phase
    time.sleep(5)"""),
    markdown_cell("### 3. Verify Headless DNS Service & Endpoints"),
    code_cell("""!kubectl get service jax-cpu-job
!kubectl get endpoints jax-cpu-job"""),
    markdown_cell("### 4. Stream Logs & Check Mathematical Proof (Expected Sum: 3.0)"),
    code_cell("""!kubectl logs -l jobset.x-k8s.io/jobset-name=jax-cpu-job --all-containers --tail=100"""),
    markdown_cell("---"),
    markdown_cell("## PART B: 10x Scale-Up Experiment (20 Nodes, 80 Virtual Devices)\nNow we scale our cluster from 2 nodes to **20 nodes** to prove multi-node JAX scaling at 10x capacity!"),
    markdown_cell("### 1. Resize GKE Cluster to 20 Nodes"),
    code_cell("""!gcloud container clusters resize {CLUSTER_NAME} \\
    --node-pool=default-pool \\
    --num-nodes=20 \\
    --zone={ZONE} \\
    --quiet"""),
    code_cell("""!kubectl get nodes -l cloud.google.com/gke-nodepool=default-pool --no-headers | wc -l"""),
    markdown_cell("### 2. Render 20-Node JobSet Manifest"),
    code_cell("""scale_path = Path("../manifests/jobset-scale-20-rendered.yaml")

with open(manifest_path, "r") as f:
    scale_content = f.read()

scale_content = scale_content.replace("LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/CPU_IMAGE_NAME:IMAGE_TAG", CPU_FULL_IMAGE)
scale_content = scale_content.replace("name: jax-cpu-job", "name: jax-scale-job")
scale_content = scale_content.replace('values: ["jax-cpu-job"]', 'values: ["jax-scale-job"]')
scale_content = scale_content.replace("parallelism: 2", "parallelism: 20")
scale_content = scale_content.replace("completions: 2", "completions: 20")
scale_content = scale_content.replace('value: "2"', 'value: "20"')
scale_content = scale_content.replace("jax-cpu-job-workers-0-0.jax-cpu-job", "jax-scale-job-workers-0-0.jax-scale-job")

with open(scale_path, "w") as f:
    f.write(scale_content)

print(f"Saved 20-Node Scale JobSet manifest to {scale_path}")"""),
    markdown_cell("### 3. Deploy 20-Node JobSet"),
    code_cell("""!kubectl apply -f ../manifests/jobset-scale-20-rendered.yaml"""),
    code_cell("""print("Monitoring 20 Pods scheduling across 20 Nodes...")
for _ in range(8):
    !kubectl get jobset jax-scale-job
    time.sleep(5)"""),
    markdown_cell("### 4. Verify 20-Pod Physical Node Distribution"),
    code_cell("""!kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-scale-job -o custom-columns=POD_NAME:.metadata.name,POD_IP:.status.podIP,NODE:.spec.nodeName,STATUS:.status.phase"""),
    markdown_cell("### 5. Stream Cross-Node Logs & Validate Mathematical Sum (Expected Sum: 210.0)"),
    code_cell("""!kubectl logs -l jobset.x-k8s.io/jobset-name=jax-scale-job --all-containers=true --prefix=true --max-log-requests=25""")
]

# ---------------------------------------------------------
# Notebook 04: JAX GPU Multi-Node Training
# ---------------------------------------------------------
nb4_cells = [
    markdown_cell("""# Module 04: Multi-Node JAX GPU Training with JobSet
In this module, you will launch and inspect a multi-node JAX GPU training workload on GKE using Kubernetes JobSet.

### Key Concepts Covered:
1. **Multi-Host JAX Initialization**: `jax.distributed.initialize()` with `COORDINATOR_ADDRESS` and rank indexing over Headless DNS.
2. **GPU Inter-Node All-Reduce**: `jax.lax.psum` across GPU nodes via NCCL.
3. **SPMD Mesh Sharding**: Sharding JAX tensors across multi-node GPU memories using `jax.sharding.Mesh`."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import config
cfg = config.load_config("../config.env")

PROJECT_ID = cfg["PROJECT_ID"]
REGION = cfg["REGION"]
REPO = cfg["ARTIFACT_REGISTRY_REPO"]
GPU_IMAGE = cfg["GPU_IMAGE_NAME"]
TAG = cfg["IMAGE_TAG"]

GPU_FULL_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/{GPU_IMAGE}:{TAG}"
print(f"Target GPU Image: {GPU_FULL_IMAGE}")"""),
    markdown_cell("## 1. Render GPU JobSet Manifest"),
    code_cell("""manifest_path = Path("../manifests/jobset-gpu.yaml")
rendered_path = Path("../manifests/jobset-gpu-rendered.yaml")

with open(manifest_path, "r") as f:
    content = f.read()

content = content.replace("LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/GPU_IMAGE_NAME:IMAGE_TAG", GPU_FULL_IMAGE)

with open(rendered_path, "w") as f:
    f.write(content)

print(f"Rendered JobSet manifest saved to {rendered_path}")"""),
    markdown_cell("## 2. Deploy JAX GPU Multi-Node JobSet"),
    code_cell("""!kubectl apply -f ../manifests/jobset-gpu-rendered.yaml"""),
    markdown_cell("## 3. Monitor JobSet & Worker Pod Status"),
    code_cell("""import time
print("Waiting for GPU worker pods to initialize...")
for i in range(8):
    !kubectl get jobset jax-gpu-job
    !kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-gpu-job -o custom-columns=POD_NAME:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase
    time.sleep(5)"""),
    markdown_cell("## 4. Stream Logs from Multi-Node GPU Workers"),
    code_cell("""!kubectl logs -l jobset.x-k8s.io/jobset-name=jax-gpu-job --all-containers --tail=100"""),
    markdown_cell("## 5. Verify GPU Execution & Mathematical Sum Proof\nInspect the logs to confirm:\n- `JAX Distributed Initialized Successfully!` across all GPU ranks.\n- GPU NCCL `lax.psum` output equals `3.0`.")
]

# ---------------------------------------------------------
# Notebook 05: JAX TPU Multi-Node Training
# ---------------------------------------------------------
nb5_cells = [
    markdown_cell("""# Module 05: Multi-Node JAX TPU Slice Training with JobSet
In this module, you will launch a multi-node JAX TPU slice training workload on GKE using JobSet.

### Key Concepts Covered:
1. **TPU Multi-Host Slices**: Interconnecting multiple TPU hosts over high-speed Inter-Chip Interconnect (ICI).
2. **Exclusive Topology & Ports**: Specifying `alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool` and exposing TPU container ports `8471` (Data Link) and `8080` (Coordinator).
3. **SPMD Mesh Sharding on TPU**: Sharding large matrix operations across TPU pod slices."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import config
cfg = config.load_config("../config.env")

PROJECT_ID = cfg["PROJECT_ID"]
REGION = cfg["REGION"]
REPO = cfg["ARTIFACT_REGISTRY_REPO"]
TPU_IMAGE = cfg["TPU_IMAGE_NAME"]
TAG = cfg["IMAGE_TAG"]

TPU_FULL_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/{TPU_IMAGE}:{TAG}"
print(f"Target TPU Image: {TPU_FULL_IMAGE}")"""),
    markdown_cell("## 1. Render TPU JobSet Manifest"),
    code_cell("""manifest_path = Path("../manifests/jobset-tpu.yaml")
rendered_path = Path("../manifests/jobset-tpu-rendered.yaml")

with open(manifest_path, "r") as f:
    content = f.read()

content = content.replace("LOCATION-docker.pkg.dev/PROJECT_ID/ARTIFACT_REGISTRY_REPO/TPU_IMAGE_NAME:IMAGE_TAG", TPU_FULL_IMAGE)

with open(rendered_path, "w") as f:
    f.write(content)

print(f"Rendered TPU JobSet manifest saved to {rendered_path}")"""),
    markdown_cell("## 2. Deploy JAX TPU Multi-Node JobSet"),
    code_cell("""!kubectl apply -f ../manifests/jobset-tpu-rendered.yaml"""),
    markdown_cell("## 3. Monitor TPU JobSet & Pod Status"),
    code_cell("""import time
print("Waiting for TPU worker pods to initialize...")
for i in range(8):
    !kubectl get jobset jax-tpu-job
    !kubectl get pods -l jobset.x-k8s.io/jobset-name=jax-tpu-job -o custom-columns=POD_NAME:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase
    time.sleep(5)"""),
    markdown_cell("## 4. Stream Logs from Multi-Node TPU Slice Workers"),
    code_cell("""!kubectl logs -l jobset.x-k8s.io/jobset-name=jax-tpu-job --all-containers --tail=100"""),
    markdown_cell("## 5. Verify TPU Execution & ICI Mathematical Sum Proof\nVerify from logs:\n- TPU multi-host slice discovery.\n- ICI interconnect `lax.psum` all-reduce output equals `3.0`.")
]

# ---------------------------------------------------------
# Notebook 06: Cleanup & Resource Teardown
# ---------------------------------------------------------
nb6_cells = [
    markdown_cell("""# Module 06: Resource Teardown and Cost Management
In this final module, you will clean up all deployed Google Cloud resources to prevent unexpected compute billing.

### Cleanup Steps:
1. Delete active Kubernetes JobSets (`jax-cpu-job`, `jax-scale-job`, `jax-gpu-job`, `jax-tpu-job`).
2. Optional: Delete TPU and GPU accelerator node pools.
3. Scale default CPU node pool back down to 1 node.
4. Delete Google Artifact Registry images and repository."""),
    code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import config
cfg = config.load_config("../config.env")

PROJECT_ID = cfg["PROJECT_ID"]
REGION = cfg["REGION"]
ZONE = cfg["ZONE"]
CLUSTER_NAME = cfg["CLUSTER_NAME"]
REPO = cfg["ARTIFACT_REGISTRY_REPO"]
TPU_NODE_POOL = cfg["TPU_NODE_POOL_NAME"]
GPU_NODE_POOL = cfg["GPU_NODE_POOL_NAME"]"""),
    markdown_cell("## 1. Delete Kubernetes JobSets"),
    code_cell("""!kubectl delete jobset jax-cpu-job jax-scale-job jax-gpu-job jax-tpu-job --ignore-not-found"""),
    markdown_cell("## 2. Scale Default CPU Pool Back Down to 1 Node"),
    code_cell("""!gcloud container clusters resize {CLUSTER_NAME} \\
    --node-pool=default-pool \\
    --num-nodes=1 \\
    --zone={ZONE} \\
    --quiet"""),
    markdown_cell("## 3. Delete Accelerator Node Pools (If Created)"),
    code_cell("""!gcloud container node-pools delete {TPU_NODE_POOL} --cluster={CLUSTER_NAME} --zone={ZONE} --quiet"""),
    code_cell("""!gcloud container node-pools delete {GPU_NODE_POOL} --cluster={CLUSTER_NAME} --zone={ZONE} --quiet"""),
    markdown_cell("## 4. Delete Artifact Registry Repository"),
    code_cell("""!gcloud artifacts repositories delete {REPO} --location={REGION} --quiet"""),
    markdown_cell("## 5. Delete GKE Cluster & Custom VPC Subnet (Complete Cleanup)"),
    code_cell("""!gcloud container clusters delete {CLUSTER_NAME} --zone={ZONE} --quiet"""),
    code_cell("""!gcloud compute networks subnets delete {cfg['SUBNET_NAME']} --region={REGION} --quiet"""),
    code_cell("""!gcloud compute networks delete {cfg['NETWORK_NAME']} --quiet"""),
    markdown_cell("## 6. Final GCP Status Check"),
    code_cell("""print("Teardown Complete! Verifying active cluster status:")
!gcloud container clusters list --zone={ZONE}""")
]

# Save notebooks
with open(notebooks_dir / "00_config_and_setup.ipynb", "w") as f:
    json.dump(make_notebook(nb0_cells), f, indent=2)

with open(notebooks_dir / "01_gke_cluster_setup.ipynb", "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

with open(notebooks_dir / "02_container_build.ipynb", "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

with open(notebooks_dir / "03_jax_cpu_multinode.ipynb", "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

with open(notebooks_dir / "04_jax_gpu_multinode.ipynb", "w") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

with open(notebooks_dir / "05_jax_tpu_multinode.ipynb", "w") as f:
    json.dump(make_notebook(nb5_cells), f, indent=2)

with open(notebooks_dir / "06_cleanup.ipynb", "w") as f:
    json.dump(make_notebook(nb6_cells), f, indent=2)

print("Successfully generated all 7 Jupyter Notebooks!")
