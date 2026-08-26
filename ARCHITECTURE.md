# JAX Multi-Node GKE Architecture & Lesson Breakdown

This document provides visual architecture diagrams and explanations for each module in the JAX Multi-Node training learning plan.

---

## 🏗️ High-Level System Architecture

![JAX Multi-Node GKE System Architecture](assets/jax_gke_architecture_diagram.jpg)

```mermaid
graph TD
    subgraph GCP["Google Cloud Project (PROJECT_ID)"]
        subgraph VPC["Custom VPC Network (jax-network)"]
            Router["Cloud Router (jax-network-router)"]
            NAT["Cloud NAT (jax-network-nat)"]
            subgraph Subnet["Subnet (jax-subnet: 10.0.0.0/20)"]
                PodRange["Pods Range: 10.4.0.0/14"]
                SvcRange["Services Range: 10.8.0.0/20"]
            end
        end

        subgraph AR["Google Artifact Registry"]
            CPUImg["jax-cpu-multinode:latest"]
            GPUImg["jax-gpu-multinode:latest"]
            TPUImg["jax-tpu-multinode:latest"]
        end

        subgraph GKE["GKE Standard Cluster (jax-distributed-cluster)"]
            ControlPlane["GKE Control Plane (JobSet Controller v0.12.0)"]
            HeadlessDNS["Headless DNS Service (jax-cpu-job / jax-cpu-train-job)"]
            
            subgraph CPUNodes["Default CPU Node Pool (2 to 20 Private Nodes)"]
                Pod0["Pod 0 (Rank 0 - Coordinator)\nIP: 10.4.0.5\n4 Virtual Devices"]
                Pod1["Pod 1 (Rank 1 - Worker)\nIP: 10.4.0.6\n4 Virtual Devices"]
                PodN["Pod N-1 (Rank N-1 - Worker)\nIP: 10.4.x.y\n4 Virtual Devices"]
            end
        end
    end

    Router --> NAT
    NAT -->|Internet Egress for Private Nodes| CPUNodes
    ControlPlane -->|Manages Lifecycle| HeadlessDNS
    ControlPlane -->|Gang Schedules| CPUNodes
    Pod0 <-->|TCP Port 1234 / Headless DNS| Pod1
    Pod0 <-->|TCP Port 1234 / Headless DNS| PodN
    Pod1 <-->|jax.lax.psum / SPMD All-Reduce| PodN
    AR -->|Pull Images| CPUNodes
```

---

## 🏛️ Infrastructure-as-Code Architecture (Terraform)

The `terraform/` directory manages the foundational cloud infrastructure declaratively:

```mermaid
flowchart TD
    subgraph TF["Terraform Engine (make tf-apply)"]
        Keepers["random_pet (Keepers: project_prefix)"]
        Proj["google_project (disposable project)"]
        OrgPol["google_project_organization_policy (requireShieldedVm: false)"]
        APIs["google_project_service (GKE, Compute, AR, CloudBuild, IAM)"]
        IAM["google_project_iam_member (Compute SA & CloudBuild SA)"]
        VPC["google_compute_network (jax-network)"]
        Subnet["google_compute_subnetwork (jax-subnet + secondary ranges)"]
        NAT["google_compute_router & nat (jax-network-nat)"]
        AR["google_artifact_registry_repository (jax-gke-repo)"]
        GKE["google_container_cluster & node_pool (Private GKE)"]
        Bootstrap["local-exec: get-credentials & apply JobSet Operator CRDs"]
    end

    Keepers --> Proj
    Proj --> OrgPol
    Proj --> APIs
    APIs --> IAM
    APIs --> VPC
    VPC --> Subnet
    Subnet --> NAT
    APIs --> AR
    NAT --> GKE
    Subnet --> GKE
    GKE --> Bootstrap
```

---

## 📘 Module 00: Environment & Custom VPC Subnet Architecture

Module 00 provisions an isolated, enterprise-compliant VPC network with dedicated secondary IP ranges for Kubernetes Pods and Services, along with a Cloud Router and Cloud NAT for secure outbound egress from private nodes.

```mermaid
flowchart LR
    User["gcloud CLI / 00_setup_network.sh"] -->|1. Enable Cloud APIs| APIs["Container, ArtifactRegistry, CloudBuild APIs"]
    User -->|2. Create Network| VPC["Custom VPC (jax-network)"]
    VPC -->|3. Create Subnet| Subnet["Custom Subnet (jax-subnet: 10.0.0.0/20)"]
    Subnet --> Secondary1["Secondary Range: pods-range (10.4.0.0/14)"]
    Subnet --> Secondary2["Secondary Range: services-range (10.8.0.0/20)"]
    Subnet --> NAT["Cloud Router & Cloud NAT (jax-network-nat)"]
```

---

## 📗 Module 01: GKE Cluster & JobSet Controller Setup

Module 01 creates a GKE Standard Cluster with Private Nodes and Workload Identity, then installs the **Kubernetes JobSet Operator (v0.12.0)**.

```mermaid
graph TD
    subgraph GKECluster["GKE Cluster Provisioning (Module 01)"]
        Flags["Cluster Architecture Flags:\n- Workload Identity\n- Private Nodes (No Public IPs)\n- Cloud NAT Outbound Egress\n- IP Aliasing"]
        
        JobSetCRD["Install JobSet Operator (v0.12.0)\nkubectl apply --server-side"]
        
        subgraph Pools["Node Pools"]
            DefaultPool["Default CPU Pool\n(e2-standard-4)"]
            GPUPool["GPU Pool (Optional)\n(g2-standard-8 / L4)"]
            TPUPool["TPU Pool (Optional)\n(ct5lp-hightpu-4t / v5e 2x4)"]
        end
    end

    Flags --> DefaultPool
    JobSetCRD -->|Watches JobSet CRD| GKECluster
```

---

## 📙 Module 02: Cloud Build & Artifact Registry Pipeline

Module 02 uses **Google Cloud Build** to compile container images serverlessly without requiring a local Docker daemon.

```mermaid
flowchart TD
    subgraph LocalRepo["Local Workspace (src/)"]
        DockerCPU["Dockerfile.cpu\n+ jax_cpu_test.py\n+ jax_cpu_train_toy_model.py"]
        DockerGPU["Dockerfile.gpu\n+ jax_gpu_test.py"]
        DockerTPU["Dockerfile.tpu\n+ jax_tpu_test.py"]
    end

    subgraph CloudBuild["Google Cloud Build"]
        BuildStep["gcloud builds submit"]
    end

    subgraph Registry["Artifact Registry (us-central1-docker.pkg.dev)"]
        Repo["jax-gke-repo"]
        ImgCPU["jax-cpu-multinode:latest"]
        ImgGPU["jax-gpu-multinode:latest"]
        ImgTPU["jax-tpu-multinode:latest"]
    end

    LocalRepo -->|Submit Source| BuildStep
    BuildStep -->|Push Image| ImgCPU
    BuildStep -->|Push Image| ImgGPU
    BuildStep -->|Push Image| ImgTPU
```

---

## 📕 Module 03: Multi-Node JAX CPU & 10x Scale-Up Architecture

Module 03 demonstrates true multiprocess multi-node JAX coordination without accelerator quota using `XLA_FLAGS="--xla_force_host_platform_device_count=4"`.

### 1. Headless DNS & Gang Scheduling Flow
```mermaid
sequenceDiagram
    autonumber
    participant K8s as JobSet Controller
    participant Rank0 as Pod 0 (Rank 0 - Coordinator)
    participant Rank1 as Pod 1 (Rank 1 - Worker)
    participant DNS as Headless DNS (jax-cpu-job)

    K8s->>DNS: Create Headless Service & Register Pod IPs
    K8s->>Rank0: Start Pod 0 (batch.kubernetes.io/job-completion-index=0)
    K8s->>Rank1: Start Pod 1 (batch.kubernetes.io/job-completion-index=1)
    
    Rank0->>Rank0: jax.distributed.initialize(COORDINATOR_ADDRESS=0.0.0.0:1234)
    Rank1->>DNS: Resolve jax-cpu-job-workers-0-0.jax-cpu-job.default.svc.cluster.local:1234
    DNS-->>Rank1: Return IP 10.4.0.5
    Rank1->>Rank0: Connect to Rank 0 Coordinator on Port 1234
    
    Note over Rank0,Rank1: All Ranks Connected! Cluster Ready.
    
    Rank0->>Rank1: jax.lax.psum(rank_val) All-Reduce
    Rank1->>Rank0: jax.lax.psum(rank_val) All-Reduce
    
    Note over Rank0,Rank1: Synchronized Sum Verified: 3.0 (2 Nodes) / 210.0 (20 Nodes)
```

---

## 🔬 Module 03b: Multi-Node CPU Toy Model Training Architecture

Module 03b executes distributed data-parallel (DDP) training of a 3-layer MLP classifier across CPU nodes using **Flax Linen** and **Optax Adam**.

```mermaid
flowchart TD
    subgraph DDP["SPMD Data Parallelism on CPUs"]
        subgraph Node0["Physical Node 0 (Rank 0)"]
            Devs0["4 Virtual CPU Devices"]
            Shard0["Local Batch Shard 0\n(128 samples)"]
            Loss0["Local Forward Loss & Grad"]
        end

        subgraph Node1["Physical Node 1 (Rank 1)"]
            Devs1["4 Virtual CPU Devices"]
            Shard1["Local Batch Shard 1\n(128 samples)"]
            Loss1["Local Forward Loss & Grad"]
        end

        Mesh["jax.sharding.Mesh: Mesh(devices, ('data',))"]
        Weights["Replicated Model Weights: NamedSharding(P())"]
        AllReduce["SPMD Gradient All-Reduce (Cross-Host TCP)"]
        Optimizer["Optax Adam Optimizer Step"]
    end

    Shard0 --> Loss0
    Shard1 --> Loss1
    Loss0 --> AllReduce
    Loss1 --> AllReduce
    Weights --> Loss0
    Weights --> Loss1
    AllReduce --> Optimizer
    Optimizer --> Weights
```

---

## 📘 Module 04: Multi-Node JAX GPU Training (NCCL Ring All-Reduce)

Module 04 orchestrates multi-node GPU training over NVIDIA NCCL interconnect across dedicated `g2-standard-8` (NVIDIA L4) nodes.

```mermaid
graph TD
    subgraph GPUNode1["GPU Node 1 (nvidia-l4)"]
        GPUPod0["Pod 0 (Rank 0)\n1x NVIDIA L4 GPU\nCOORDINATOR_ADDRESS: :1234"]
    end

    subgraph GPUNode2["GPU Node 2 (nvidia-l4)"]
        GPUPod1["Pod 1 (Rank 1)\n1x NVIDIA L4 GPU\nConnects to Rank 0"]
    end

    GPUPod0 <-->|NCCL Ring All-Reduce over VPC Network| GPUPod1
```

---

## 📗 Module 05: Multi-Host TPU Slice Training (ICI Interconnect)

Module 05 deploys a TPU v5e multi-host slice (`2x4` topology, 8 total TPU chips) connected via Inter-Chip Interconnect (ICI).

```mermaid
graph TD
    subgraph TPUSlice["TPU v5e Podslice (2x4 Topology)"]
        subgraph TPUHost0["TPU Host 0 (ct5lp-hightpu-4t)"]
            TPUPod0["Worker 0 (Rank 0)\n4x TPU v5e Chips\nContainer Ports: 8471, 8080"]
        end

        subgraph TPUHost1["TPU Host 1 (ct5lp-hightpu-4t)"]
            TPUPod1["Worker 1 (Rank 1)\n4x TPU v5e Chips\nContainer Ports: 8471, 8080"]
        end
    end

    TPUPod0 <-->|High-Speed ICI Interconnect (Port 8471)| TPUPod1
```

---

## 📙 Module 06: Cost Teardown & Resource Cleanup Flow

Module 06 cleans up all created GCP assets in strict reverse dependency order to prevent unexpected compute charges (`make tf-destroy` or `./scripts/06_cleanup.sh`).

```mermaid
flowchart TD
    Step1["1. Delete JobSets\n(kubectl delete jobset ...)"] --> Step2["2. Scale CPU Node Pool Down\n(Resize to 1 Node)"]
    Step2 --> Step3["3. Delete Accelerator Pools\n(GPU & TPU Node Pools)"]
    Step3 --> Step4["4. Delete Artifact Registry Repository\n(gcloud artifacts repositories delete)"]
    Step4 --> Step5["5. Delete GKE Cluster\n(gcloud container clusters delete)"]
    Step5 --> Step6["6. Delete Cloud NAT & Cloud Router\n(gcloud compute routers nats/routers delete)"]
    Step6 --> Step7["7. Delete Custom Subnet & VPC Network\n(gcloud compute networks subnets/networks delete)"]
```
