# JAX Multi-Node GKE Architecture & Lesson Breakdown

This document provides visual architecture diagrams and explanations for each module in the JAX Multi-Node training learning plan.

---

## 🏗️ High-Level System Architecture

```mermaid
graph TD
    subgraph GCP["Google Cloud Project (PROJECT_ID)"]
        subgraph VPC["Custom VPC Network (jax-network)"]
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
            ControlPlane["GKE Control Plane (JobSet Controller v0.6.0)"]
            HeadlessDNS["Headless DNS Service (jax-cpu-job)"]
            
            subgraph CPUNodes["Default CPU Node Pool (2 to 20 Nodes)"]
                Pod0["Pod 0 (Rank 0 - Coordinator)\nIP: 10.4.0.5\n4 Virtual Devices"]
                Pod1["Pod 1 (Rank 1 - Worker)\nIP: 10.4.0.6\n4 Virtual Devices"]
                PodN["Pod N-1 (Rank N-1 - Worker)\nIP: 10.4.x.y\n4 Virtual Devices"]
            end
        end
    end

    ControlPlane -->|Manages Lifecycle| HeadlessDNS
    ControlPlane -->|Gang Schedules| CPUNodes
    Pod0 <-->|TCP Port 1234 / Headless DNS| Pod1
    Pod0 <-->|TCP Port 1234 / Headless DNS| PodN
    Pod1 <-->|jax.lax.psum All-Reduce| PodN
    AR -->|Pull Images| CPUNodes
```

---

## 📘 Module 00: Environment & Custom VPC Subnet Architecture

Module 00 provisions an isolated, enterprise-compliant VPC network with dedicated secondary IP ranges for Kubernetes Pods and Services.

```mermaid
flowchart LR
    User["gcloud CLI / Notebook 00"] -->|1. Enable Cloud APIs| APIs["Container, ArtifactRegistry, CloudBuild APIs"]
    User -->|2. Create Network| VPC["Custom VPC (jax-network)"]
    VPC -->|3. Create Subnet| Subnet["Custom Subnet (jax-subnet: 10.0.0.0/20)"]
    Subnet --> Secondary1["Secondary Range: pods-range (10.4.0.0/14)"]
    Subnet --> Secondary2["Secondary Range: services-range (10.8.0.0/20)"]
```

---

## 📗 Module 01: GKE Cluster & JobSet Controller Setup

Module 01 creates a GKE Standard Cluster with Shielded VM security flags and installs the **Kubernetes JobSet Operator**.

```mermaid
graph TD
    subgraph GKECluster["GKE Cluster Provisioning (Module 01)"]
        Flags["Cluster Security Flags:\n- Workload Identity\n- Shielded Secure Boot\n- Shielded Integrity Monitoring\n- IP Aliasing"]
        
        JobSetCRD["Install JobSet Operator (v0.6.0)\nkubectl apply --server-side"]
        
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
        DockerCPU["Dockerfile.cpu\n+ jax_cpu_test.py"]
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
    Rank1->>DNS: Resolve jax-cpu-job-workers-0-0.jax-cpu-job:1234
    DNS-->>Rank1: Return IP 10.4.0.5
    Rank1->>Rank0: Connect to Rank 0 Coordinator on Port 1234
    
    Note over Rank0,Rank1: All Ranks Connected! Cluster Ready.
    
    Rank0->>Rank1: jax.lax.psum(rank_val) All-Reduce
    Rank1->>Rank0: jax.lax.psum(rank_val) All-Reduce
    
    Note over Rank0,Rank1: Synchronized Sum Verified: 3.0 (2 Nodes) / 210.0 (20 Nodes)
```

### 2. Pod Anti-Affinity Topology
```mermaid
graph LR
    subgraph Node1["Physical Node 1 (compute-vm-1)"]
        Pod0["Pod 0 (Rank 0)\n1.5 CPU / 2Gi Mem\n4 XLA Virtual Devices"]
    end
    
    subgraph Node2["Physical Node 2 (compute-vm-2)"]
        Pod1["Pod 1 (Rank 1)\n1.5 CPU / 2Gi Mem\n4 XLA Virtual Devices"]
    end

    Pod0 -.-|podAntiAffinity: topologyKey kubernetes.io/hostname| Pod1
```

---

## 📘 Module 04: Multi-Node JAX GPU Training (NCCL Ring All-Reduce)

Module 04 orchestrates multi-node GPU training over NVIDIA NCCL interconnect.

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

Module 05 deploys a TPU v5e multi-host slice (2x4 topology) connected via Inter-Chip Interconnect (ICI).

```mermaid
graph TD
    subgraph TPUSlice["TPU v5e Podslice (2x4 Topology)"]
        subgraph TPUHost0["TPU Host 0"]
            TPUPod0["Worker 0 (Rank 0)\n4x TPU v5e Chips\nContainer Ports: 8471, 8080"]
        end

        subgraph TPUHost1["TPU Host 1"]
            TPUPod1["Worker 1 (Rank 1)\n4x TPU v5e Chips\nContainer Ports: 8471, 8080"]
        end
    end

    TPUPod0 <-->|High-Speed ICI Interconnect| TPUPod1
```

---

## 📙 Module 06: Cost Teardown & Resource Cleanup Flow

Module 06 cleans up all created GCP assets in strict reverse dependency order to prevent unexpected compute charges.

```mermaid
flowchart TD
    Step1["1. Delete JobSets\n(kubectl delete jobset ...)"] --> Step2["2. Scale CPU Node Pool Down\n(Resize to 1 Node)"]
    Step2 --> Step3["3. Delete Accelerator Pools\n(GPU & TPU Node Pools)"]
    Step3 --> Step4["4. Delete Artifact Registry Repository\n(gcloud artifacts repositories delete)"]
    Step4 --> Step5["5. Delete GKE Cluster\n(gcloud container clusters delete)"]
    Step5 --> Step6["6. Delete Custom Subnet & VPC Network\n(gcloud compute networks subnets/networks delete)"]
```
