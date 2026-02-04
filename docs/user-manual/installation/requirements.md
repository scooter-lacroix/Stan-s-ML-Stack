# System Requirements - Stan's ML Stack

## 📋 Hardware and Software Prerequisites

This guide outlines the complete system requirements for installing and running Stan's ML Stack across different platforms and deployment scenarios.

---

## 🎯 Overview

Stan's ML Stack is designed to work with AMD GPUs and provides comprehensive machine learning capabilities. The requirements vary based on your intended use case, from basic development to production deployment.

### Supported Platforms
- **Linux**: Ubuntu 22.04 LTS (primary), Ubuntu 20.04 LTS, RHEL 8+, CentOS 8+
- **Windows**: Windows 10/11 with WSL2 (limited support)
- **macOS**: macOS 12+ (CPU-only, limited GPU support)
- **Docker**: All platforms with Docker support
- **Cloud**: AWS, GCP, Azure with AMD GPU instances

---

## 💻 Hardware Requirements

### Minimum Hardware Requirements

**CPU:**
- **Architecture**: x86_64 (AMD64)
- **Cores**: 4+ physical cores
- **Speed**: 2.5 GHz+ base clock
- **Architecture**: AMD Ryzen 5/7 or Intel Core i5/i7 recommended

**GPU:**
- **Required**: AMD GPU with ROCm support
- **Series**: Radeon RX 5000 series or newer
- **VRAM**: 8GB+ dedicated VRAM
- **Architecture**: RDNA2 or newer recommended

**Memory:**
- **RAM**: 16GB DDR4/DDR5
- **Speed**: 3200MHz+ recommended
- **Type**: DDR4 or DDR5

**Storage:**
- **Space**: 50GB free disk space
- **Type**: SSD recommended (NVMe preferred)
- **Speed**: 500MB/s+ sequential read/write

**Network:**
- **Connection**: Ethernet or WiFi
- **Speed**: 100Mbps+ for downloads
- **Latency**: <100ms recommended

### Recommended Hardware Specifications

**High-Performance Setup:**
- **CPU**: AMD Ryzen 7/9 or Intel Core i7/i9 (8+ cores, 3.5GHz+)
- **GPU**: AMD Radeon RX 7900 XTX (24GB VRAM)
- **RAM**: 32GB+ DDR5 (4800MHz+)
- **Storage**: 1TB+ NVMe SSD (3000MB/s+)
- **Network**: 1Gbps+ Ethernet

**Multi-GPU Setup:**
- **GPU**: 2-4x AMD Radeon RX 7900 XTX/7800 XT
- **CPU**: AMD Ryzen 9 or Intel Core i9 (12+ cores)
- **RAM**: 64GB+ DDR5
- **Storage**: 2TB+ NVMe SSD
- **Power**: 1200W+ PSU (80+ Platinum)
- **Cooling**: Adequate GPU and case cooling

### Supported AMD GPUs

**Fully Supported (ROCm 6.x):**
- **Radeon RX 7900 XTX** (Navi 31, GFX1100)
- **Radeon RX 7900 XT** (Navi 31, GFX1100)
- **Radeon RX 7800 XT** (Navi 32, GFX1100)
- **Radeon RX 7700 XT** (Navi 32, GFX1100)
- **Radeon RX 6900 XT** (Navi 21, GFX1030)
- **Radeon RX 6800 XT** (Navi 21, GFX1030)
- **Radeon RX 6800** (Navi 21, GFX1030)

**Limited Support:**
- **Radeon RX 5700 XT** (Navi 10, GFX1010) - Experimental
- **Radeon RX 5700** (Navi 10, GFX1010) - Experimental
- **Radeon VII** (Vega 20, GFX903) - Limited

**GPU Architecture Support:**
- **GFX1100**: Navi 3x (RX 7000 series) - Full support
- **GFX1030**: Navi 2x (RX 6000 series) - Full support
- **GFX1010**: Navi 1x (RX 5000 series) - Experimental
- **GFX903**: Vega 20 - Limited support

### GPU Compatibility Matrix

| GPU Model | Architecture | VRAM | ROCm Support | Performance | Recommended |
|-----------|-------------|------|--------------|-------------|-------------|
| RX 7900 XTX | GFX1100 | 24GB | ✅ Full | ⭐⭐⭐⭐⭐ | ✅ Yes |
| RX 7900 XT | GFX1100 | 20GB | ✅ Full | ⭐⭐⭐⭐⭐ | ✅ Yes |
| RX 7800 XT | GFX1100 | 16GB | ✅ Full | ⭐⭐⭐⭐ | ✅ Yes |
| RX 7700 XT | GFX1100 | 12GB | ✅ Full | ⭐⭐⭐ | ✅ Yes |
| RX 6900 XT | GFX1030 | 16GB | ✅ Full | ⭐⭐⭐⭐ | ✅ Yes |
| RX 6800 XT | GFX1030 | 16GB | ✅ Full | ⭐⭐⭐⭐ | ✅ Yes |
| RX 6800 | GFX1030 | 16GB | ✅ Full | ⭐⭐⭐ | ✅ Yes |
| RX 5700 XT | GFX1010 | 8GB | ⚠️ Experimental | ⭐⭐ | ⚠️ Limited |

---

## 🖥️ Software Requirements

### Operating Systems

**Linux (Primary Support):**
- **Ubuntu 22.04 LTS** (Jammy Jellyfish) - Recommended
- **Ubuntu 20.04 LTS** (Focal Fossa) - Supported
- **RHEL 8.x** - Supported with modifications
- **CentOS 8.x** - Supported with modifications
- **Fedora 37+** - Experimental

**Windows:**
- **Windows 10 Pro/Enterprise** (Build 19044+) with WSL2
- **Windows 11 Pro/Enterprise** (Build 22000+) with WSL2
- **Requirements**: WSL2 Ubuntu 22.04 distribution

**macOS:**
- **macOS 12+ (Monterey)** - CPU-only support
- **macOS 13+ (Ventura)** - CPU-only support
- **Requirements**: Apple Silicon or Intel Mac with 16GB+ RAM

### Kernel Requirements

**Linux Kernel:**
- **Minimum**: 5.15.x (Ubuntu 22.04 default)
- **Recommended**: 5.19.x or newer
- **ROCm Compatibility**: Kernel must support ROCm drivers

**Required Kernel Modules:**
- `amdgpu` - AMD GPU driver
- `kfd` - HSA (Heterogeneous System Architecture) kernel driver
- `drm` - Direct Rendering Manager
- `i2c_algo_bit` - I2C bit-banging algorithm

### System Libraries

**Base System Libraries:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    pkg-config \
    libnuma-dev \
    libhsa-runtime-dev \
    librocblas-dev \
    miopen-hip-dev \
    rccl-rocm-dev \
    hipblas-dev \
    hipsparse-dev \
    hipcub-dev \
    rocrand-dev \
    rocfft-dev \
    rocthrust-dev \
    rocsolver-dev
```

**Development Tools:**
- **GCC**: 9.0+ (recommended 11.0+)
- **Git**: 2.25+
- **CMake**: 3.18+ (recommended 3.25+)
- **Make**: 4.2+
- **Python**: 3.10+ (recommended 3.11+)

### Python Environment

**Python Version:**
- **Minimum**: Python 3.10.0
- **Recommended**: Python 3.11.0+
- **Maximum**: Python 3.13.x (with workarounds)

**Package Manager:**
- **pip**: 23.0+ (recommended)
- **conda**: 4.12+ (alternative)
- **uv**: 0.1+ (recommended for performance)

**Core Python Dependencies:**
```bash
# Required base packages
pip install numpy>=1.21.0
pip install wheel setuptools
pip install typing-extensions

# ML Stack will install additional packages during setup
```

---

## 🔧 Platform-Specific Requirements

### Ubuntu 22.04 LTS (Recommended)

**System Preparation:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required repositories
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# Install development tools
sudo apt install -y build-essential cmake git wget curl

# Install Python 3.11 (if not default)
sudo apt install -y python3.11 python3.11-dev python3.11-pip
```

**User Configuration:**
```bash
# Add user to required groups
sudo usermod -a -G video,render $USER

# Configure environment (requires logout/login)
newgrp video
newgrp render
```

### Windows with WSL2

**Windows Requirements:**
- **Windows 10/11 Pro/Enterprise**
- **Virtualization**: Enabled in BIOS
- **WSL2**: Installed and configured
- **Memory**: 16GB+ system RAM recommended

**WSL2 Setup:**
```powershell
# Enable WSL2
wsl --install -d Ubuntu-22.04

# Configure WSL2 memory (optional)
# Create %USERPROFILE%\.wslconfig
[wsl2]
memory=32GB
swap=8GB
```

**Ubuntu WSL2 Configuration:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y build-essential cmake git wget curl python3-dev

# Configure user groups
sudo usermod -a -G video,render $USER
```

### macOS

**System Requirements:**
- **macOS 12+ (Monterey)**
- **Apple Silicon M1/M2** or **Intel Mac**
- **RAM**: 16GB+ recommended
- **Storage**: 100GB+ available space

**Installation via Homebrew:**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install python@3.11 git cmake

# Create virtual environment
python3.11 -m venv ~/mlstack-env
source ~/mlstack-env/bin/activate
```

---

## 🏢 Enterprise Requirements

### Production Environment

**Hardware Redundancy:**
- **Power**: Dual power supplies (N+1 redundancy)
- **Storage**: RAID 1+0 for system, RAID 5 for data
- **Network**: Dual network interfaces (teaming)
- **Cooling**: Redundant cooling systems
- **UPS**: Uninterruptible power supply

**Network Requirements:**
- **Bandwidth**: 10Gbps+ for multi-node clusters
- **Latency**: <1ms for internode communication
- **Firewall**: Configured for required ports
- **DNS**: Proper hostname resolution
- **NTP**: Time synchronization

**Security Requirements:**
- **Access Control**: LDAP/Active Directory integration
- **Authentication**: Multi-factor authentication
- **Authorization**: Role-based access control
- **Audit Logging**: Comprehensive audit trails
- **Encryption**: Data at rest and in transit

### Cloud Platform Requirements

**AWS:**
- **Instance**: g4dn.xlarge or larger (AMD equivalent when available)
- **Storage**: EBS gp3 (3000 IOPS, 125 MB/s)
- **Network**: Enhanced networking enabled
- **IAM**: Appropriate IAM roles and policies
- **VPC**: Proper VPC configuration

**Google Cloud:**
- **Instance**: n2-standard-8 or larger
- **Storage**: Persistent Disk SSD
- **Network**: VPC with proper firewall rules
- **Service Account**: Required permissions
- **Monitoring**: Cloud monitoring integration

**Azure:**
- **Instance**: Standard_NC6s_v3 or larger
- **Storage**: Premium SSD
- **Network**: Virtual Network with NSG
- **Identity**: Managed Identity configuration
- **Monitoring**: Azure Monitor integration

---

## 📊 Storage Requirements

### Disk Space Breakdown

**Base Installation:**
- **ROCm**: 8-10GB
- **PyTorch**: 3-5GB
- **ONNX Runtime**: 2-3GB
- **ML Stack**: 1-2GB
- **Dependencies**: 5-8GB
- **Total Base**: ~20-30GB

**Development Environment:**
- **Development Tools**: 5-10GB
- **Datasets**: 10-100GB+ (varies)
- **Models**: 5-50GB+ (varies)
- **Cache**: 5-20GB
- **Total Development**: ~50-200GB

**Production Environment:**
- **System**: 50GB
- **Applications**: 20GB
- **Data**: 500GB+ (varies)
- **Models**: 100GB+ (varies)
- **Logs**: 50GB+ (varies)
- **Total Production**: ~1TB+

### Storage Performance Requirements

**System Disk:**
- **Type**: NVMe SSD (recommended)
- **Speed**: 3000MB/s+ sequential
- **IOPS**: 100,000+ IOPS
- **Latency**: <0.1ms

**Data Storage:**
- **Type**: NVMe SSD or Enterprise SATA SSD
- **Speed**: 1000MB/s+ sequential
- **IOPS**: 50,000+ IOPS
- **Capacity**: 1TB+ (depends on use case)

**Network Storage (Optional):**
- **Protocol**: NFS v4.2 or SMB 3.1+
- **Speed**: 10Gbps+ network
- **Latency**: <10ms
- **Capacity**: 10TB+ (for large datasets)

---

## 🔍 Validation Checklist

### Pre-Installation Validation

**Hardware Validation:**
- [ ] AMD GPU detected in system
- [ ] Sufficient RAM available
- [ ] Adequate storage space
- [ ] Proper power supply capacity
- [ ] Cooling system adequate

**Software Validation:**
- [ ] Supported OS version installed
- [ ] Kernel version compatible
- [ ] Required system libraries available
- [ ] Python 3.10+ installed
- [ ] User permissions adequate

**Network Validation:**
- [ ] Internet connectivity available
- [ ] DNS resolution working
- [ ] Required ports open
- [ ] Sufficient bandwidth
- [ ] Low latency connection

**User Environment Validation:**
- [ ] User in video/render groups
- [ ] Sufficient user permissions
- [ ] Home directory accessible
- [ ] Environment variables not conflicting
- [ ] No conflicting software installed

### Post-Installation Validation

**Basic Functionality:**
- [ ] ROCm drivers loaded correctly
- [ ] GPU accessible to user
- [ ] PyTorch with ROCm working
- [ ] Basic tensor operations successful
- [ ] Memory allocation working

**Performance Validation:**
- [ ] GPU utilization normal
- [ ] Memory usage appropriate
- [ ] Network communication working
- [ ] Storage performance adequate
- [ ] Temperature within limits

---

## 🚨 Common Issues and Solutions

### Hardware Compatibility Issues

**GPU Not Detected:**
```bash
# Check if GPU is detected
lspci | grep -i amd
lspci | grep -i vga

# Check if amdgpu driver is loaded
lsmod | grep amdgpu
dmesg | grep amdgpu
```

**Solution:**
1. Verify GPU is supported
2. Check motherboard compatibility
3. Update system BIOS/UEFI
4. Ensure proper power connections

### Software Compatibility Issues

**ROCm Installation Fails:**
```bash
# Check ROCm compatibility
rocminfo
rocm-smi

# Check kernel modules
lsmod | grep -E "(amdgpu|kfd|drm)"
```

**Solution:**
1. Verify kernel version compatibility
2. Check required system libraries
3. Ensure proper user permissions
4. Update system packages

### Performance Issues

**Poor GPU Performance:**
```bash
# Check GPU utilization
rocm-smi

# Check GPU memory
rocm-smi --showmemuse

# Check GPU clock speeds
rocm-smi --showclocks
```

**Solution:**
1. Verify GPU power settings
2. Check thermal throttling
3. Update GPU drivers
4. Optimize system settings

---

## 📞 Support Resources

### Self-Service Diagnostics
```bash
# Run comprehensive system check
curl -sSL https://raw.githubusercontent.com/scooter-lacroix/Stans_MLStack/main/scripts/diagnostic_check.sh | bash

# Generate system report
./scripts/generate_system_report.sh
```

### Community Support
- **GitHub Issues**: [Report issues](https://github.com/scooter-lacroix/Stan-s-ML-Stack/issues)
- **GitHub Discussions**: [Community forum](https://github.com/scooter-lacroix/Stan-s-ML-Stack/discussions)
- **Documentation**: [Complete docs](https://github.com/scooter-lacroix/Stan-s-ML-Stack/wiki)

### Professional Support
- **Email**: scooterlacroix@gmail.com
- **Consulting**: Custom deployment assistance
- **Training**: Team training available

---

## 📝 Additional Resources

### Documentation Links
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [AMD GPU Compatibility](https://www.amd.com/en/support)

### Performance Benchmarks
- [GPU Benchmarks](https://www.phoronix.com/)
- [ML Performance Guides](../performance/benchmarking.md)
- [Optimization Tips](../performance/gpu-optimization.md)

---

## ✅ Next Steps

After confirming your system meets these requirements:

1. **Proceed to Installation**: [Linux Installation Guide](linux.md)
2. **Choose Installation Method**: [Installation Methods Overview](../README.md)
3. **Prepare Environment**: [Environment Setup Guide](../../configuration/environment.md)
4. **Plan Configuration**: [Configuration Overview](../../configuration/README.md)

---

*This requirements document is part of the comprehensive user manual for Stan's ML Stack. For installation instructions, please proceed to the [Installation Guide](linux.md).*