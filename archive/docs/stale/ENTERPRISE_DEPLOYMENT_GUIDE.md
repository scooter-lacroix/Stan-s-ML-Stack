# Stan's ML Stack - Enterprise Deployment Guide

## 🏢 Complete Enterprise Deployment Package

Welcome to the comprehensive enterprise deployment guide for Stan's ML Stack. This document provides complete instructions for deploying, managing, and operating the ML Stack in production enterprise environments.

---

## 🎯 Deployment Overview

Stan's ML Stack provides enterprise-grade deployment options for all environments, from development to production. Our deployment configurations are designed for scalability, security, and maintainability.

### Supported Deployment Methods

1. **Kubernetes Deployment** - Production-ready container orchestration
2. **Docker Compose** - Local development and small deployments
3. **Systemd Services** - Traditional Linux service management
4. **Windows Services** - Windows-based deployments
5. **macOS Services** - macOS launch agents
6. **Cloud Platforms** - Multi-cloud deployment support

### Deployment Environments

- **Development** - Local development and testing
- **Staging** - Pre-production validation
- **Production** - Live enterprise deployment
- **Disaster Recovery** - Backup and failover environments

---

## 🚀 Quick Start Deployment

### Prerequisites

**System Requirements:**
- AMD GPU with ROCm support (see [Requirements Guide](docs/user-manual/installation/requirements.md))
- 16GB+ RAM (32GB+ recommended)
- 100GB+ storage
- Kubernetes 1.25+ (for K8s deployment)
- Docker 20.10+ (for container deployment)

**Required Software:**
- `kubectl` (Kubernetes CLI)
- `helm` (Kubernetes package manager)
- `docker` (Container runtime)
- `git` (Version control)

### One-Command Kubernetes Deployment

```bash
# Deploy Stan's ML Stack to Kubernetes
curl -sSL https://raw.githubusercontent.com/scooter-lacroix/Stans_MLStack/main/scripts/deploy-kubernetes.sh | bash

# Verify deployment
kubectl get pods -l app.kubernetes.io/name=stans-ml-stack
```

### Docker Compose Deployment

```bash
# Clone repository
git clone https://github.com/scooter-lacroix/Stans_MLStack.git
cd Stans_MLStack

# Deploy with Docker Compose
cd deploy/docker/docker-compose
docker-compose up -d

# Verify deployment
docker-compose ps
```

---

## ☸️ Kubernetes Deployment

### Architecture Overview

Our Kubernetes deployment provides:
- **High Availability**: Multi-replica deployment with automatic failover
- **Scalability**: Horizontal Pod Autoscaling (HPA) support
- **Resource Management**: Comprehensive CPU, memory, and GPU management
- **Security**: Pod Security Standards and Network Policies
- **Monitoring**: Integrated Prometheus and Grafana monitoring
- **Persistence**: Persistent volume management for data and logs

### Installation Prerequisites

**Cluster Requirements:**
```bash
# Verify Kubernetes version
kubectl version --short

# Check available GPU nodes
kubectl get nodes --label-selector=accelerator=amd-gpu

# Verify storage classes
kubectl get storageclass
```

**GPU Operator Setup:**
```bash
# Install AMD GPU Operator (if not already installed)
helm repo add amd-gpu https://rocm.github.io/k8s-device-plugin
helm repo update

# Install GPU operator
helm install amd-gpu amd-gpu/amd-gpu-operator \
  --namespace gpu-operator \
  --create-namespace
```

### Step-by-Step Deployment

**1. Add Helm Repository:**
```bash
helm repo add stans-ml-stack https://stans-ml-stack.github.io/helm-charts
helm repo update
```

**2. Create Namespace:**
```bash
kubectl create namespace ml-stack
```

**3. Install ML Stack:**
```bash
helm install ml-stack stans-ml-stack/stans-ml-stack \
  --namespace ml-stack \
  --values deploy/kubernetes/helm-chart/stans-ml-stack/values.yaml
```

**4. Verify Installation:**
```bash
# Check pod status
kubectl get pods -n ml-stack

# Check services
kubectl get services -n ml-stack

# Verify GPU access
kubectl exec -n ml-stack deployment/ml-stack -- rocminfo
```

### Production Configuration

**Production Values Override:**
```yaml
# values-production.yaml
replicaCount: 3

resources:
  limits:
    cpu: 8
    memory: 32Gi
    amd.com/gpu: 1
  requests:
    cpu: 4
    memory: 16Gi
    amd.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

persistence:
  data:
    size: 1Ti
    storageClass: fast-ssd

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true

backup:
  enabled: true
  schedule: "0 2 * * *"
  retention:
    daily: 7
    weekly: 4
    monthly: 12
```

**Deploy with Production Configuration:**
```bash
helm upgrade ml-stack stans-ml-stack/stans-ml-stack \
  --namespace ml-stack \
  --values deploy/kubernetes/helm-chart/stans-ml-stack/values.yaml \
  --values deploy/kubernetes/helm-chart/stans-ml-stack/values-production.yaml
```

### Advanced Configuration

**Multi-GPU Configuration:**
```yaml
# Multi-GPU setup
gpu:
  enabled: true
  deviceCount: 4
  amd:
    memory:
      maxMemory: "96Gi"

# Node affinity for GPU nodes
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: accelerator
          operator: In
          values: ["amd-gpu"]
        - key: gpu-memory
          operator: Gt
          values: ["16Gi"]
```

**Distributed Training Configuration:**
```yaml
# Distributed training setup
distributedTraining:
  enabled: true
  replicas: 4
  resources:
    limits:
      cpu: 16
      memory: 64Gi
      amd.com/gpu: 1

  mpi:
    enabled: true
    replicas: 4

  networking:
    hostNetwork: true
    dnsPolicy: ClusterFirstWithHostNet
```

### Monitoring and Observability

**Prometheus Monitoring:**
```bash
# Access Prometheus
kubectl port-forward -n ml-stack svc/ml-stack-prometheus 9090:9090

# Access Grafana
kubectl port-forward -n ml-stack svc/ml-stack-grafana 3000:3000
```

**Log Aggregation:**
```bash
# View application logs
kubectl logs -n ml-stack -l app.kubernetes.io/name=stans-ml-stack -f

# View system logs
kubectl logs -n ml-stack -l app.kubernetes.io/name=stans-ml-stack --previous
```

### Backup and Disaster Recovery

**Automated Backup Setup:**
```yaml
backup:
  enabled: true
  schedule: "0 2 * * *"
  retention:
    daily: 7
    weekly: 4
    monthly: 12
  storage:
    type: s3
    s3:
      bucket: ml-stack-backups
      region: us-west-2
```

**Manual Backup:**
```bash
# Backup configuration
kubectl get configmap ml-stack-config -n ml-stack -o yaml > backup-config.yaml

# Backup persistent volumes
kubectl exec -n ml-stack deployment/ml-stack -- tar -czf /backup/data-$(date +%Y%m%d).tar.gz /data
```

---

## 🐳 Docker Compose Deployment

### Local Development Setup

**Prerequisites:**
```bash
# Verify Docker installation
docker --version
docker-compose --version

# Verify GPU support
docker run --rm --gpus all nvidia-smi
```

### Development Deployment

**Basic Setup:**
```bash
# Clone repository
git clone https://github.com/scooter-lacroix/Stans_MLStack.git
cd Stans_MLStack

# Set environment variables
cp deploy/docker/docker-compose/.env.example .env

# Edit environment variables
nano .env
```

**Environment Configuration (.env):**
```bash
# GPU Configuration
GPU_COUNT=1
GPU_MEMORY=24Gi

# Database Configuration
POSTGRES_PASSWORD=secure_password_change_me
REDIS_PASSWORD=redis_password_change_me

# Monitoring Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin_change_me

# Application Configuration
MLSTACK_ENV=development
MLSTACK_LOG_LEVEL=DEBUG
```

**Start Services:**
```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d ml-stack postgres redis

# Start with GPU support
docker-compose --profile gpu up -d
```

**Development Workflow:**
```bash
# View logs
docker-compose logs -f ml-stack

# Access services
docker-compose exec ml-stack bash

# Restart services
docker-compose restart ml-stack

# Scale services
docker-compose up -d --scale ml-stack=3
```

### Production Docker Deployment

**Production Docker Compose:**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ml-stack:
    image: bartholemewii/stans-ml-stack:latest
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]

    environment:
      - MLSTACK_ENV=production
      - MLSTACK_LOG_LEVEL=INFO

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
```

**Deploy Production Setup:**
```bash
# Deploy production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
```

---

## 🖥️ Systemd Service Deployment

### Linux Service Installation

**Create Service File:**
```bash
# Create systemd service
sudo nano /etc/systemd/system/stans-ml-stack.service
```

**Service Configuration:**
```ini
[Unit]
Description=Stan's ML Stack Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=ml-stack
Group=ml-stack
WorkingDirectory=/opt/stans-ml-stack
Environment=ROCM_PATH=/opt/rocm
Environment=HIP_VISIBLE_DEVICES=0
Environment=MLSTACK_ENV=production
ExecStart=/opt/stans-ml-stack/bin/ml-stack-server
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/stans-ml-stack /var/lib/stans-ml-stack

[Install]
WantedBy=multi-user.target
```

**Service Management:**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable stans-ml-stack
sudo systemctl start stans-ml-stack

# Check service status
sudo systemctl status stans-ml-stack

# View logs
sudo journalctl -u stans-ml-stack -f
```

### Service Configuration

**Environment Configuration:**
```bash
# Create environment file
sudo nano /etc/default/stans-ml-stack
```

```bash
# /etc/default/stans-ml-stack
ROCM_PATH=/opt/rocm
HIP_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
PYTORCH_ROCM_ARCH=GFX1100
HSA_OVERRIDE_GFX_VERSION=11.0.0

MLSTACK_ENV=production
MLSTACK_LOG_LEVEL=INFO
MLSTACK_DATA_DIR=/var/lib/stans-ml-stack/data
MLSTACK_LOG_DIR=/var/log/stans-ml-stack
```

**Service Logging:**
```bash
# Configure log rotation
sudo nano /etc/logrotate.d/stans-ml-stack
```

```
/var/log/stans-ml-stack/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ml-stack ml-stack
    postrotate
        systemctl reload stans-ml-stack
    endscript
}
```

---

## 🪟 Windows Service Deployment

### Windows Service Installation

**Prerequisites:**
- Windows 10/11 Pro or Enterprise
- WSL2 with Ubuntu 22.04
- Docker Desktop with WSL2 integration
- Administrative privileges

**WSL2 Setup:**
```powershell
# Enable WSL2
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu
wsl --install -d Ubuntu-22.04
```

**Docker Desktop Configuration:**
```powershell
# Enable WSL2 integration in Docker Desktop
# Settings > Resources > WSL Integration > Enable Ubuntu-22.04
```

### Windows Service Script

**Create Service Script:**
```powershell
# Create service installation script
New-Item -ItemType Directory -Force -Path "C:\Program Files\Stan's ML Stack"
```

**Service Configuration (service.ps1):**
```powershell
# C:\Program Files\Stan's ML Stack\service.ps1
param(
    [string]$Action = "start"
)

$ServiceName = "StansMLStack"
$ContainerName = "stans-ml-stack-windows"

switch ($Action) {
    "install" {
        # Create Windows service
        New-Service -Name $ServiceName -BinaryPathName "powershell.exe -File `"C:\Program Files\Stan's ML Stack\service.ps1`" -Action run" -DisplayName "Stan's ML Stack" -StartupType Automatic
        Write-Host "Service installed successfully"
    }

    "start" {
        # Start Docker container
        docker run -d `
            --name $ContainerName `
            --gpus all `
            -p 8080:8080 `
            -p 9090:9090 `
            -v C:\data:/data `
            -v C:\logs:/logs `
            -e MLSTACK_ENV=production `
            bartholemewii/stans-ml-stack:latest
        Write-Host "Container started"
    }

    "stop" {
        # Stop and remove container
        docker stop $ContainerName
        docker rm $ContainerName
        Write-Host "Container stopped"
    }

    "status" {
        # Check container status
        docker ps -a --filter name=$ContainerName
    }

    default {
        Write-Host "Usage: service.ps1 -Action [install|start|stop|status]"
    }
}
```

**Install Windows Service:**
```powershell
# Run as Administrator
cd "C:\Program Files\Stan's ML Stack"
powershell -ExecutionPolicy Bypass -File service.ps1 -Action install

# Start the service
Start-Service -Name $ServiceName
```

---

## 🍎 macOS Service Deployment

### macOS LaunchAgent Setup

**Prerequisites:**
- macOS 12+ (Monterey)
- Docker Desktop for Mac
- Homebrew package manager

**Installation via Homebrew:**
```bash
# Install dependencies
brew install python@3.11 docker
brew services start docker

# Create application directory
sudo mkdir -p /opt/stans-ml-stack
sudo chown $(whoami) /opt/stans-ml-stack
```

### LaunchAgent Configuration

**Create LaunchAgent:**
```bash
# Create LaunchAgent directory
mkdir -p ~/Library/LaunchAgents

# Create LaunchAgent file
nano ~/Library/LaunchAgents/com.stansmlstack.service.plist
```

**LaunchAgent Configuration:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.stansmlstack.service</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/docker</string>
        <string>run</string>
        <string>--rm</string>
        <string>--name</string>
        <string>stans-ml-stack</string>
        <string>-p</string>
        <string>8080:8080</string>
        <string>-p</string>
        <string>9090:9090</string>
        <string>-v</string>
        <string>/opt/stans-ml-stack/data:/data</string>
        <string>-v</string>
        <string>/opt/stans-ml-stack/logs:/logs</string>
        <string>-e</string>
        <string>MLSTACK_ENV=production</string>
        <string>bartholemewii/stans-ml-stack:latest</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/opt/stans-ml-stack/logs/service.log</string>

    <key>StandardErrorPath</key>
    <string>/opt/stans-ml-stack/logs/service.error.log</string>

    <key>WorkingDirectory</key>
    <string>/opt/stans-ml-stack</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
        <key>MLSTACK_ENV</key>
        <string>production</string>
    </dict>
</dict>
</plist>
```

**Load and Start Service:**
```bash
# Load the LaunchAgent
launchctl load ~/Library/LaunchAgents/com.stansmlstack.service.plist

# Start the service
launchctl start com.stansmlstack.service

# Check service status
launchctl list | grep com.stansmlstack.service
```

---

## ☁️ Cloud Platform Deployment

### AWS Deployment

**EKS Cluster Setup:**
```bash
# Create EKS cluster
eksctl create cluster \
  --name ml-stack-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --with-oidc \
  --ssh-access \
  --ssh-public-key ~/.ssh/id_rsa.pub
```

**Install NVIDIA Device Plugin:**
```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

**Deploy to EKS:**
```bash
# Add IAM OIDC provider
eksctl utils associate-iam-oidc-provider --region us-west-2 --cluster ml-stack-cluster --approve

# Deploy ML Stack
helm install ml-stack stans-ml-stack/stans-ml-stack \
  --namespace ml-stack \
  --create-namespace \
  --set cloud.provider=aws \
  --set cloud.region=us-west-2 \
  --set storageClass=gp3
```

### Google Cloud Platform Deployment

**GKE Cluster Setup:**
```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create ml-stack-cluster \
  --region us-central1 \
  --num-nodes=1 \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=3
```

**Install GPU Driver:**
```bash
# Install NVIDIA GPU driver on GKE
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

**Deploy to GKE:**
```bash
# Deploy ML Stack
helm install ml-stack stans-ml-stack/stans-ml-stack \
  --namespace ml-stack \
  --create-namespace \
  --set cloud.provider=gcp \
  --set cloud.region=us-central1 \
  --set storageClass=standard-rwo
```

### Microsoft Azure Deployment

**AKS Cluster Setup:**
```bash
# Create resource group
az group create --name ml-stack-rg --location eastus

# Create AKS cluster with GPU nodes
az aks create \
  --resource-group ml-stack-rg \
  --name ml-stack-cluster \
  --node-count 1 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 3 \
  --generate-ssh-keys
```

**Install NVIDIA Device Plugin:**
```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

**Deploy to AKS:**
```bash
# Deploy ML Stack
helm install ml-stack stans-ml-stack/stans-ml-stack \
  --namespace ml-stack \
  --create-namespace \
  --set cloud.provider=azure \
  --set cloud.region=eastus \
  --set storageClass=managed-premium
```

---

## 🔧 Configuration Management

### Environment Configuration

**Development Environment:**
```yaml
# values-dev.yaml
replicaCount: 1
resources:
  requests:
    cpu: 2
    memory: 8Gi
  limits:
    cpu: 4
    memory: 16Gi
monitoring:
  enabled: false
persistence:
  size: 50Gi
```

**Staging Environment:**
```yaml
# values-staging.yaml
replicaCount: 2
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 4
monitoring:
  enabled: true
persistence:
  size: 200Gi
backup:
  enabled: true
  schedule: "0 3 * * *"
```

**Production Environment:**
```yaml
# values-prod.yaml
replicaCount: 3
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
persistence:
  size: 1Ti
backup:
  enabled: true
  schedule: "0 2 * * *"
  retention:
    daily: 7
    weekly: 4
    monthly: 12
security:
  enabled: true
  podSecurityPolicy:
    enabled: true
```

### Secret Management

**Kubernetes Secrets:**
```bash
# Create database secrets
kubectl create secret generic ml-stack-db-secrets \
  --from-literal=postgres-password=$(openssl rand -base64 32) \
  --from-literal=redis-password=$(openssl rand -base64 32)

# Create application secrets
kubectl create secret generic ml-stack-app-secrets \
  --from-literal=secret-key=$(openssl rand -base64 32) \
  --from-literal=api-token=$(openssl rand -base64 32)
```

**Environment Variables:**
```yaml
# Configure secret references
env:
  - name: DATABASE_PASSWORD
    valueFrom:
      secretKeyRef:
        name: ml-stack-db-secrets
        key: postgres-password
  - name: REDIS_PASSWORD
    valueFrom:
      secretKeyRef:
        name: ml-stack-db-secrets
        key: redis-password
```

---

## 📊 Monitoring and Maintenance

### Health Monitoring

**Application Health Checks:**
```yaml
healthChecks:
  enabled: true
  livenessProbe:
    httpGet:
      path: /health
      port: 8080
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  readinessProbe:
    httpGet:
      path: /ready
      port: 8080
    initialDelaySeconds: 5
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3
```

**Monitoring Dashboard Access:**
```bash
# Port forward to access services
kubectl port-forward -n ml-stack svc/ml-stack-grafana 3000:3000
kubectl port-forward -n ml-stack svc/ml-stack-prometheus 9090:9090

# Default credentials
# Grafana: admin/admin (change on first login)
```

### Log Management

**Log Collection:**
```bash
# View application logs
kubectl logs -n ml-stack -l app.kubernetes.io/name=stans-ml-stack -f

# View logs from specific time range
kubectl logs -n ml-stack -l app.kubernetes.io/name=stans-ml-stack --since=1h

# View previous deployment logs
kubectl logs -n ml-stack -l app.kubernetes.io/name=stans-ml-stack --previous
```

**Log Aggregation Setup:**
```yaml
logging:
  enabled: true
  level: INFO
  format: json
  collection:
    fluentd:
      enabled: true
      host: fluentd.default.svc.cluster.local
      port: 24224
```

### Backup and Recovery

**Automated Backup Configuration:**
```bash
# Create backup schedule
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: ml-stack-backup
  namespace: ml-stack
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U postgres mlstack > /backup/backup-$(date +%Y%m%d).sql
              aws s3 cp /backup/backup-$(date +%Y%m%d).sql s3://ml-stack-backups/
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: ml-stack-db-secrets
                  key: postgres-password
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: ml-stack-backup
          restartPolicy: OnFailure
EOF
```

**Recovery Procedures:**
```bash
# Restore from backup
kubectl exec -i -n ml-stack deployment/postgres -- psql -U postgres -d mlstack < backup-20231028.sql

# Verify restore
kubectl exec -n ml-stack deployment/postgres -- psql -U postgres -d mlstack -c "\dt"
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

**GPU Not Detected:**
```bash
# Check GPU availability
kubectl describe nodes | grep gpu

# Check GPU device plugin
kubectl logs -n gpu-operator deployment/amd-gpu-device-plugin

# Verify GPU resources
kubectl top nodes
```

**Pod Startup Issues:**
```bash
# Check pod events
kubectl describe pod -n ml-stack <pod-name>

# Check pod logs
kubectl logs -n ml-stack <pod-name>

# Check resource usage
kubectl top pods -n ml-stack
```

**Performance Issues:**
```bash
# Check GPU utilization
kubectl exec -n ml-stack deployment/ml-stack -- rocm-smi

# Check resource limits
kubectl describe pod -n ml-stack <pod-name> | grep Limits

# Check network connectivity
kubectl exec -n ml-stack deployment/ml-stack -- ping postgres
```

**Storage Issues:**
```bash
# Check PV status
kubectl get pv

# Check PVC status
kubectl get pvc -n ml-stack

# Check storage class
kubectl get storageclass
```

### Debug Commands

**Comprehensive Health Check:**
```bash
#!/bin/bash
# health-check.sh

echo "=== Cluster Status ==="
kubectl cluster-info

echo -e "\n=== Node Status ==="
kubectl get nodes -o wide

echo -e "\n=== Pod Status ==="
kubectl get pods -n ml-stack -o wide

echo -e "\n=== Service Status ==="
kubectl get services -n ml-stack

echo -e "\n=== GPU Status ==="
kubectl describe nodes | grep -i gpu

echo -e "\n=== Resource Usage ==="
kubectl top pods -n ml-stack

echo -e "\n=== Recent Events ==="
kubectl get events -n ml-stack --sort-by=.metadata.creationTimestamp
```

---

## 📞 Support and Resources

### Documentation Resources
- **Complete User Manual**: [User Manual](docs/user-manual/README.md)
- **Installation Guide**: [Installation Guide](docs/user-manual/installation/)
- **Troubleshooting Guide**: [Troubleshooting Guide](docs/user-manual/troubleshooting/)
- **Performance Guide**: [Performance Guide](docs/user-manual/performance/)

### Community Support
- **GitHub Issues**: [Report Issues](https://github.com/scooter-lacroix/Stans_MLStack/issues)
- **GitHub Discussions**: [Community Forum](https://github.com/scooter-lacroix/Stans_MLStack/discussions)
- **Discord Server**: [Join Community](https://discord.gg/stans-ml-stack)

### Professional Support
- **Email**: scooterlacroix@gmail.com
- **Enterprise Support**: Available upon request
- **Consulting Services**: Custom deployment and optimization

---

## ✅ Deployment Checklist

### Pre-Deployment Checklist
- [ ] System requirements verified
- [ ] GPU drivers installed and configured
- [ ] Kubernetes cluster provisioned
- [ ] Storage classes configured
- [ ] Network policies reviewed
- [ ] Security policies configured
- [ ] Backup strategy defined
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Documentation reviewed

### Post-Deployment Checklist
- [ ] All pods running and healthy
- [ ] GPU resources accessible
- [ ] Services accessible via load balancer
- [ ] Monitoring dashboards working
- [ ] Log aggregation working
- [ ] Backup procedures tested
- [ ] Security scan completed
- [ ] Performance benchmarks run
- [ ] User access tested
- [ ] Documentation updated

---

## 🎯 Next Steps

After successful deployment:

1. **Configure Monitoring**: Set up alerts and dashboards
2. **Optimize Performance**: Tune GPU and system settings
3. **Implement Security**: Apply security best practices
4. **Establish Backup**: Configure automated backup procedures
5. **Train Users**: Provide user training and documentation
6. **Monitor Usage**: Track system performance and usage
7. **Plan Scaling**: Prepare for future scaling needs

---

*This enterprise deployment guide is part of the comprehensive documentation suite for Stan's ML Stack. For additional information, please refer to the [Complete Documentation Index](docs/ENTERPRISE_DOCUMENTATION_INDEX.md).*