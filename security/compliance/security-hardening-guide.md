# Enterprise Security Hardening Guide - Stan's ML Stack

## 🔒 Comprehensive Security Configuration

This guide provides comprehensive security hardening procedures for Stan's ML Stack in enterprise environments, covering network security, access control, data protection, and compliance requirements.

---

## 🎯 Security Overview

Stan's ML Stack implements defense-in-depth security architecture with multiple layers of protection:

**Security Layers:**
- **Network Security**: Network policies, firewalls, encryption
- **Application Security**: Code scanning, vulnerability management, secure coding
- **Data Security**: Encryption at rest and in transit, access controls
- **Infrastructure Security**: Container security, host security, runtime protection
- **Identity and Access Management**: Authentication, authorization, audit logging
- **Compliance**: Industry standards adherence, audit trails, reporting

---

## 🏗️ Security Architecture

### Threat Model

**Primary Threats:**
- Unauthorized access to GPU resources and ML models
- Data exfiltration of sensitive training data
- Model poisoning and adversarial attacks
- Resource exhaustion and denial of service
- Supply chain attacks in ML dependencies
- Privilege escalation and lateral movement

**Security Controls:**
- Zero-trust network architecture
- Multi-factor authentication (MFA)
- Role-based access control (RBAC)
- End-to-end encryption
- Comprehensive audit logging
- Regular security scanning and patching

### Security Zones

**Production Zone:**
- Highest security controls
- Restricted access
- Continuous monitoring
- Regular compliance audits

**Staging Zone:**
- Production-equivalent security
- Isolated from production
- Testing security controls
- Pre-deployment validation

**Development Zone:**
- Developer-friendly security
- Code scanning and analysis
- Secure development practices
- Isolated from sensitive data

---

## 🔐 Network Security

### Kubernetes Network Policies

**Default Deny Policy:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: ml-stack
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

**Application Network Policy:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-stack-netpol
  namespace: ml-stack
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: stans-ml-stack
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to:
    - namespaceSelector:
        matchLabels:
          name: ml-stack
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 80   # HTTP (for package downloads)
```

### Service Mesh Integration

**Istio Configuration:**
```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: ml-stack
spec:
  mtls:
    mode: STRICT

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: ml-stack-authz
  namespace: ml-stack
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: stans-ml-stack
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/ingress-nginx/sa/ingress-nginx"]
  - to:
    - operation:
        methods: ["GET", "POST", "PUT", "DELETE"]
```

### TLS Configuration

**Ingress TLS:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-stack-ingress
  namespace: ml-stack
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
    nginx.ingress.kubernetes.io/ssl-ciphers: "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - ml-stack.company.com
    secretName: ml-stack-tls
  rules:
  - host: ml-stack.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-stack
            port:
              number: 8080
```

**Certificate Management:**
```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: ml-stack-cert
  namespace: ml-stack
spec:
  secretName: ml-stack-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - ml-stack.company.com
  - api.ml-stack.company.com
  - grafana.ml-stack.company.com
```

---

## 🔑 Identity and Access Management

### RBAC Configuration

**Service Account:**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-stack
  namespace: ml-stack
  annotations:
    iam.amazonaws.com/role: arn:aws:iam::123456789012:role/ml-stack-role

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-stack-role
  namespace: ml-stack
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ml-stack-binding
  namespace: ml-stack
subjects:
- kind: ServiceAccount
  name: ml-stack
  namespace: ml-stack
roleRef:
  kind: Role
  name: ml-stack-role
  apiGroup: rbac.authorization.k8s.io
```

### OIDC Integration

**OIDC Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: oidc-config
  namespace: ml-stack
data:
  oidc-config.yaml: |
    oidc:
      issuer_url: "https://login.company.com"
      client_id: "ml-stack-client"
      client_secret: "ml-stack-secret"
      redirect_uri: "https://ml-stack.company.com/oauth/callback"
      scopes: ["openid", "profile", "email"]
      groups_claim: "groups"
```

### Authentication Policies

**Multi-Factor Authentication:**
```yaml
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: ml-stack-req-auth
  namespace: ml-stack
spec:
  jwtRules:
  - issuer: "https://login.company.com"
    jwksUri: "https://login.company.com/.well-known/jwks.json"
    forwardOriginalToken: true

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: ml-stack-mfa
  namespace: ml-stack
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: stans-ml-stack
  rules:
  - when:
    - key: request.auth.claims[amr]
      values: ["mfa"]
```

---

## 🛡️ Container Security

### Pod Security Standards

**Pod Security Policy:**
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: ml-stack-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
```

**Security Context:**
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault

containers:
- name: ml-stack
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
      - ALL
    # Required for GPU access
    add:
    - SYS_ADMIN
  resources:
    requests:
      cpu: "4"
      memory: "16Gi"
      amd.com/gpu: "1"
    limits:
      cpu: "8"
      memory: "32Gi"
      amd.com/gpu: "1"
```

### Image Security

**Trivy Image Scanning:**
```yaml
apiVersion: aquasecurity.github.io/v1alpha1
kind: Policy
metadata:
  name: ml-stack-policy
  namespace: ml-stack
spec:
  rules:
  - name: check-cves
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "Container image has CVE vulnerabilities"
      pattern:
        any:
        - spec:
            containers:
            - name: "*"
              image:
                pattern: "*:*"
                validate:
                  message: "Image must be scanned for vulnerabilities"
                  cet:
                    required: true
```

**Admission Controller:**
```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: ml-stack-validator
webhooks:
- name: ml-stack-validator.company.com
  clientConfig:
    service:
      name: ml-stack-validator
      namespace: ml-stack
      path: "/validate"
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  admissionReviewVersions: ["v1", "v1beta1"]
  sideEffects: None
```

---

## 🔒 Data Protection

### Encryption at Rest

**Kubernetes Secrets Encryption:**
```yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: <base64-encoded-32-byte-key>
    - identity: {}
```

**Persistent Volume Encryption:**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: encrypted-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  encrypted: "true"
  kmsKeyId: arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012
allowVolumeExpansion: true
```

**Database Encryption:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-encryption
  namespace: ml-stack
data:
  postgresql.conf: |
    ssl = on
    ssl_cert_file = '/var/lib/postgresql/server.crt'
    ssl_key_file = '/var/lib/postgresql/server.key'
    ssl_ca_file = '/var/lib/postgresql/ca.crt'

    # Enable transparent data encryption
    shared_preload_libraries = 'pgcrypto'

    # Log all SSL connections
    log_connections = on
    log_disconnections = on
```

### Encryption in Transit

**Service Mesh mTLS:**
```yaml
apiVersion: security.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ml-stack-destination
  namespace: ml-stack
spec:
  host: ml-stack.ml-stack.svc.cluster.local
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
    portLevelSettings:
    - port:
        number: 5432
      tls:
        mode: ISTIO_MUTUAL
    - port:
        number: 6379
      tls:
        mode: ISTIO_MUTUAL
```

**API Gateway Security:**
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: ml-stack-gateway
  namespace: ml-stack
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: ml-stack-tls
    hosts:
    - ml-stack.company.com
```

---

## 🔍 Security Monitoring and Logging

### Audit Logging

**Kubernetes Audit Policy:**
```yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Metadata
  namespaces: ["ml-stack"]
  resources:
  - group: ""
    resources: ["secrets", "configmaps"]
  - group: "apps"
    resources: ["deployments", "replicasets"]
- level: Request
  namespaces: ["ml-stack"]
  resources:
  - group: ""
    resources: ["pods"]
  - group: "batch"
    resources: ["jobs", "cronjobs"]
```

**Falco Rules:**
```yaml
- rule: Unexpected Process in ML Stack Container
  desc: Detect unexpected process execution in ML Stack containers
  condition: >
    spawned_process and
    container.image contains "stans-ml-stack" and
    not proc.name in (python, bash, sh, rocminfo, rocm-smi)
  output: >
    Unexpected process detected in ML Stack container
    (user=%user.name command=%proc.cmdline container=%container.name)
  priority: WARNING
  tags: [process, container, security]

- rule: GPU Access Suspicious Activity
  desc: Detect suspicious GPU access patterns
  condition: >
    spawned_process and
    proc.name in (python, torch) and
    container.image contains "stans-ml-stack" and
    proc.args contains "network" and
    proc.args contains "download"
  output: >
    Suspicious GPU activity detected
    (user=%user.name command=%proc.cmdline container=%container.name)
  priority: HIGH
  tags: [gpu, security, ml-stack]
```

### Security Monitoring

**Prometheus Security Rules:**
```yaml
groups:
- name: security.rules
  rules:
  - alert: UnauthorizedAccessAttempts
    expr: rate(http_requests_total{status=~"4.."}[5m]) > 10
    for: 5m
    labels:
      severity: warning
      category: security
    annotations:
      summary: "High rate of unauthorized access attempts"
      description: "{{ $value }} unauthorized requests per second detected."

  - alert: SuspiciousProcessActivity
    expr: increase(falco_events_total{priority="HIGH"}[5m]) > 0
    for: 1m
    labels:
      severity: critical
      category: security
    annotations:
      summary: "Suspicious process activity detected"
      description: "High priority security event detected: {{ $labels.rule }}."

  - alert: GPUAnomalousUsage
    expr: rocm_gpu_utilization_percent > 95
    for: 15m
    labels:
      severity: warning
      category: security
    annotations:
      summary: "Anomalous GPU usage detected"
      description: "GPU {{ $labels.gpu_id }} utilization at {{ $value }}% for extended period."
```

### Log Aggregation

**Fluentd Security Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-security-config
  namespace: logging
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*ml-stack*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>

    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>

    <filter kubernetes.**>
      @type grep
      <regexp>
        key $.kubernetes.namespace_name
        pattern ^ml-stack$
      </regexp>
    </filter>

    <filter kubernetes.**>
      @type record_transformer
      <record>
        hostname ${hostname}
        environment production
        service ml-stack
      </record>
    </filter>

    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name ml-stack-security
      type_name _doc
    </match>
```

---

## 🛡️ Vulnerability Management

### Container Scanning

**Trivy Integration:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: trivy-scan
  namespace: security
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trivy
            image: aquasec/trivy:latest
            command:
            - /bin/sh
            - -c
            - |
              trivy image --format json --output /reports/ml-stack-$(date +%Y%m%d).json bartholemewii/stans-ml-stack:latest
              aws s3 cp /reports/ml-stack-$(date +%Y%m%d).json s3://security-reports/container-scans/
            env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key-id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-access-key
            volumeMounts:
            - name: reports
              mountPath: /reports
          volumes:
          - name: reports
            emptyDir: {}
          restartPolicy: OnFailure
```

**OWASP Dependency Check:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dependency-check
  namespace: security
spec:
  schedule: "0 3 * * 0"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: dependency-check
            image: owasp/dependency-check:latest
            command:
            - /bin/sh
            - -c
            - |
              /usr/share/dependency-check/bin/dependency-check.sh \
                --project "ML Stack" \
                --scan /src \
                --format JSON \
                --out /reports/dependency-check-$(date +%Y%m%d).json
              aws s3 cp /reports/dependency-check-$(date +%Y%m%d).json s3://security-reports/dependency-check/
            volumeMounts:
            - name: source-code
              mountPath: /src
            - name: reports
              mountPath: /reports
          volumes:
          - name: source-code
            gitRepo:
              repository: https://github.com/scooter-lacroix/Stans_MLStack.git
              revision: main
          - name: reports
            emptyDir: {}
          restartPolicy: OnFailure
```

### Runtime Security

**Syscall Monitoring:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: seccomp-config
  namespace: ml-stack
data:
  ml-stack.json: |
    {
      "defaultAction": "SCMP_ACT_ERRNO",
      "architectures": [
        "SCMP_ARCH_X86_64"
      ],
      "syscalls": [
        {
          "names": [
            "accept",
            "accept4",
            "access",
            "adjtimex",
            "alarm",
            "bind",
            "brk",
            "capget",
            "capset",
            "chdir",
            "chmod",
            "chown",
            "chown32",
            "clock_getres",
            "clock_gettime",
            "clock_nanosleep",
            "close",
            "connect",
            "copy_file_range",
            "creat",
            "dup",
            "dup2",
            "dup3",
            "epoll_create",
            "epoll_create1",
            "epoll_ctl",
            "epoll_ctl_old",
            "epoll_pwait",
            "epoll_wait",
            "epoll_wait_old",
            "eventfd",
            "eventfd2",
            "execve",
            "execveat",
            "exit",
            "exit_group",
            "faccessat",
            "fadvise64",
            "fadvise64_64",
            "fallocate",
            "fanotify_mark",
            "fchdir",
            "fchmod",
            "fchmodat",
            "fchown",
            "fchown32",
            "fchownat",
            "fcntl",
            "fcntl64",
            "fdatasync",
            "fgetxattr",
            "flistxattr",
            "flock",
            "fork",
            "fremovexattr",
            "fsetxattr",
            "fstat",
            "fstat64",
            "fstatat64",
            "fstatfs",
            "fstatfs64",
            "fsync",
            "ftruncate",
            "ftruncate64",
            "futex",
            "getcwd",
            "getdents",
            "getdents64",
            "getegid",
            "getegid32",
            "geteuid",
            "geteuid32",
            "getgid",
            "getgid32",
            "getgroups",
            "getgroups32",
            "getitimer",
            "getpeername",
            "getpgid",
            "getpgrp",
            "getpid",
            "getppid",
            "getpriority",
            "getrandom",
            "getresgid",
            "getresgid32",
            "getresuid",
            "getresuid32",
            "getrlimit",
            "get_robust_list",
            "getrusage",
            "getsid",
            "getsockname",
            "getsockopt",
            "get_thread_area",
            "gettid",
            "gettimeofday",
            "getuid",
            "getuid32",
            "getxattr",
            "inotify_add_watch",
            "inotify_init",
            "inotify_init1",
            "inotify_rm_watch",
            "io_cancel",
            "ioctl",
            "io_destroy",
            "io_getevents",
            "ioperm",
            "iopl",
            "io_setup",
            "io_submit",
            "ipc",
            "kill",
            "lchown",
            "lchown32",
            "lgetxattr",
            "link",
            "linkat",
            "listen",
            "listxattr",
            "llistxattr",
            "lremovexattr",
            "lseek",
            "lsetxattr",
            "lstat",
            "lstat64",
            "madvise",
            "memfd_create",
            "mincore",
            "mkdir",
            "mkdirat",
            "mknod",
            "mknodat",
            "mlock",
            "mlock2",
            "mlockall",
            "mmap",
            "mmap2",
            "mprotect",
            "mq_getsetattr",
            "mq_notify",
            "mq_open",
            "mq_timedreceive",
            "mq_timedsend",
            "mq_unlink",
            "mremap",
            "msgctl",
            "msgget",
            "msgrcv",
            "msgsnd",
            "msync",
            "munlock",
            "munlockall",
            "munmap",
            "nanosleep",
            "newfstatat",
            "open",
            "openat",
            "pause",
            "pipe",
            "pipe2",
            "poll",
            "ppoll",
            "prctl",
            "pread64",
            "prlimit64",
            "pselect6",
            "ptrace",
            "pwrite64",
            "read",
            "readahead",
            "readlink",
            "readlinkat",
            "readv",
            "recv",
            "recvfrom",
            "recvmmsg",
            "recvmsg",
            "remap_file_pages",
            "removexattr",
            "rename",
            "renameat",
            "renameat2",
            "restart_syscall",
            "rmdir",
            "rt_sigaction",
            "rt_sigpending",
            "rt_sigprocmask",
            "rt_sigqueueinfo",
            "rt_sigreturn",
            "rt_sigsuspend",
            "rt_sigtimedwait",
            "rt_tgsigqueueinfo",
            "sched_getaffinity",
            "sched_getattr",
            "sched_getparam",
            "sched_get_priority_max",
            "sched_get_priority_min",
            "sched_getscheduler",
            "sched_rr_get_interval",
            "sched_setaffinity",
            "sched_setattr",
            "sched_setparam",
            "sched_setscheduler",
            "sched_yield",
            "seccomp",
            "select",
            "semctl",
            "semget",
            "semop",
            "semtimedop",
            "send",
            "sendfile",
            "sendfile64",
            "sendmmsg",
            "sendmsg",
            "sendto",
            "setfsgid",
            "setfsgid32",
            "setfsuid",
            "setfsuid32",
            "setgid",
            "setgid32",
            "setgroups",
            "setgroups32",
            "setitimer",
            "setpgid",
            "setpriority",
            "setregid",
            "setregid32",
            "setresgid",
            "setresgid32",
            "setresuid",
            "setresuid32",
            "setreuid",
            "setreuid32",
            "setrlimit",
            "set_robust_list",
            "setsid",
            "setsockopt",
            "set_thread_area",
            "set_tid_address",
            "setuid",
            "setuid32",
            "setxattr",
            "shmat",
            "shmctl",
            "shmdt",
            "shmget",
            "shutdown",
            "sigaltstack",
            "signalfd",
            "signalfd4",
            "sigreturn",
            "socket",
            "socketcall",
            "socketpair",
            "splice",
            "stat",
            "stat64",
            "statfs",
            "statfs64",
            "statx",
            "symlink",
            "symlinkat",
            "sync",
            "sync_file_range",
            "syncfs",
            "sysinfo",
            "tee",
            "tgkill",
            "time",
            "timer_create",
            "timer_delete",
            "timerfd_create",
            "timerfd_gettime",
            "timerfd_settime",
            "timer_getoverrun",
            "timer_gettime",
            "timer_settime",
            "times",
            "tkill",
            "truncate",
            "truncate64",
            "ugetrlimit",
            "umask",
            "uname",
            "unlink",
            "unlinkat",
            "utime",
            "utimensat",
            "utimes",
            "vfork",
            "vmsplice",
            "wait4",
            "waitid",
            "waitpid",
            "write",
            "writev"
          ],
          "action": "SCMP_ACT_ALLOW"
        },
        {
          "names": [
            "personality"
          ],
          "action": "SCMP_ACT_ALLOW",
          "args": [
            {
              "index": 0,
              "value": 0x0,
              "valueTwo": 0x0,
              "op": "SCMP_CMP_EQ"
            }
          ]
        }
      ]
    }
```

---

## 📋 Compliance Frameworks

### SOC 2 Type II Compliance

**Control Implementation:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: soc2-controls
  namespace: compliance
data:
  controls.yaml: |
    soc2:
      security:
        - access_control:
            description: "Multi-factor authentication required for all users"
            implementation: "OIDC with MFA integration"
        - encryption:
            description: "Data encrypted at rest and in transit"
            implementation: "KMS for data at rest, TLS 1.3 for transit"
        - audit_logging:
            description: "Comprehensive audit trails for all access"
            implementation: "Falco + CloudTrail audit logging"

      availability:
        - backup:
            description: "Automated daily backups with 30-day retention"
            implementation: "CronJob backups to S3 with versioning"
        - monitoring:
            description: "24/7 monitoring with alerting"
            implementation: "Prometheus + AlertManager + PagerDuty"
        - disaster_recovery:
            description: "Multi-region disaster recovery capability"
            implementation: "Cross-region backup and restoration procedures"

      processing_integrity:
        - change_management:
            description: "Formal change management process"
            implementation: "GitOps with required approvals"
        - data_validation:
            description: "Input validation and data integrity checks"
            implementation: "Application-level validation and checksums"

      confidentiality:
        - data_classification:
            description: "Data classified and handled appropriately"
            implementation: "Labels and access controls based on classification"
        - least_privilege:
            description: "Principle of least privilege enforced"
            implementation: "RBAC with minimal required permissions"

      privacy:
        - data_minimization:
            description: "Only collect necessary data"
            implementation: "Data collection policies and retention schedules"
        - consent_management:
            description: "User consent obtained and managed"
            implementation: "Consent management system and audit trails"
```

### GDPR Compliance

**Data Protection Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gdpr-compliance
  namespace: compliance
data:
  gdpr-config.yaml: |
    gdpr:
      data_protection:
        - lawful_basis: "Legitimate interest for ML model training"
        - purpose_limitation: "Only use data for specified ML purposes"
        - data_minimization: "Collect only necessary training data"
        - accuracy: "Regular data quality checks and validation"
        - storage_limitation: "Retain data only for necessary period"
        - integrity_confidentiality: "Encryption and access controls"
        - accountability: "Comprehensive audit trails and documentation"

      subject_rights:
        - right_to_access: "API endpoints for data access requests"
        - right_to_rectification: "Data correction and update procedures"
        - right_to_erasure: "Data deletion and anonymization processes"
        - right_to_portability: "Data export in standard formats"
        - right_to_object: "Automated decision-making controls"
        - right_to_restriction: "Processing limitation capabilities"

      technical_measures:
        - encryption_at_rest: "AES-256 encryption for stored data"
        - encryption_in_transit: "TLS 1.3 for network communication"
        - access_controls: "Multi-factor authentication and RBAC"
        - audit_logging: "Comprehensive access and modification logs"
        - pseudonymization: "Data pseudonymization where possible"
        - data_breach_procedures: "72-hour breach notification process"
```

### HIPAA Compliance

**Healthcare Data Protection:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hipaa-compliance
  namespace: compliance
data:
  hipaa-config.yaml: |
    hipaa:
      administrative_safeguards:
        - security_officer: "Designated HIPAA Security Officer"
        - workforce_training: "Regular security awareness training"
        - access_management: "Formal access request and approval process"
        - incident_response: "Security incident response procedures"
        - contingency_planning: "Business continuity and disaster recovery"

      physical_safeguards:
        - facility_access: "Controlled access to data centers"
        - workstation_security: "Screen locks and device encryption"
        - device_disposal: "Secure disposal and media destruction"

      technical_safeguards:
        - access_control: "Unique user authentication and authorization"
        - audit_controls: "Comprehensive audit logging and monitoring"
        - integrity: "Data integrity controls and validation"
        - transmission_security: "Encryption for data transmission"

      breach_notification:
        - risk_assessment: "Breach risk assessment procedures"
        - notification_timeline: "60-day notification requirement"
        - documentation: "Comprehensive breach documentation"
        - remediation: "Breach response and remediation procedures"
```

---

## 🔧 Security Automation

### Automated Security Scanning

**GitHub Actions Security Workflow:**
```yaml
name: Security Scan Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Run Trivy Vulnerability Scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy Results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'ml-stack'
          path: '.'
          format: 'HTML'
          out: 'dependency-check-report'

      - name: Run Semgrep Security Scan
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/cwe-top-25
            p/owasp-top-ten

      - name: Run Gosec Security Scanner
        uses: securecodewarrior/github-action-gosec@master
        with:
          args: './...'

      - name: Security Scorecard Analysis
        uses: ossf/scorecard-action@v2.3.1
        with:
          results_file: results.sarif
          results_format: sarif

      - name: Upload Scorecard Results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: results.sarif
```

### Policy Enforcement

**OPA Gatekeeper Policies:**
```yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: ml-stack-must-have-labels
spec:
  enforcementAction: deny
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["ml-stack"]
  parameters:
    labels:
      - key: "app.kubernetes.io/name"
        allowedRegex: "stans-ml-stack"
      - key: "security.company.com/owner"
        allowedRegex: ".*"

---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sDisallowAnonymous
metadata:
  name: ml-stack-no-anonymous-access
spec:
  enforcementAction: deny
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["ServiceAccount"]
    namespaces: ["ml-stack"]

---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sPSPPrivilegedContainer
metadata:
  name: ml-stack-no-privileged-containers
spec:
  enforcementAction: deny
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["ml-stack"]
  parameters:
    exemptImages: []
```

### Compliance Reporting

**Automated Compliance Reports:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: compliance-report
  namespace: compliance
spec:
  schedule: "0 6 * * 1"  # Weekly on Monday at 6 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: compliance-reporter
            image: company/compliance-reporter:latest
            command:
            - /bin/sh
            - -c
            - |
              # Generate SOC 2 report
              python generate_soc2_report.py --output /reports/soc2-$(date +%Y%m%d).pdf

              # Generate GDPR report
              python generate_gdpr_report.py --output /reports/gdpr-$(date +%Y%m%d).pdf

              # Generate HIPAA report
              python generate_hipaa_report.py --output /reports/hipaa-$(date +%Y%m%d).pdf

              # Upload to compliance dashboard
              aws s3 cp /reports/ s3://compliance-reports/ --recursive

              # Send notification
              curl -X POST "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
                -H 'Content-type: application/json' \
                --data '{"text":"Compliance reports generated for week $(date +%Y-%U)"}'
            env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key-id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-access-key
            volumeMounts:
            - name: reports
              mountPath: /reports
          volumes:
          - name: reports
            emptyDir: {}
          restartPolicy: OnFailure
```

---

## 📊 Security Metrics and KPIs

### Security Dashboard Metrics

**Key Security Indicators:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-metrics
  namespace: monitoring
data:
  security-kpis.yaml: |
    security_metrics:
      vulnerability_management:
        - critical_vulnerabilities: "count of critical CVEs"
        - vulnerability_remediation_time: "avg time to patch vulnerabilities"
        - scan_coverage: "percentage of images scanned"

      access_control:
        - failed_login_attempts: "rate of failed authentication attempts"
        - privileged_access_requests: "count of privilege escalation requests"
        - mfa_adoption_rate: "percentage of users with MFA enabled"

      threat_detection:
        - security_alerts: "count of security alerts generated"
        - false_positive_rate: "percentage of false positive alerts"
        - detection_time: "avg time to detect security incidents"

      compliance:
        - policy_compliance_rate: "percentage of resources compliant"
        - audit_findings: "count of audit findings"
        - remediation_time: "avg time to address compliance issues"

      infrastructure_security:
        - secure_configuration_score: "percentage of securely configured resources"
        - encryption_coverage: "percentage of encrypted resources"
        - network_segmentation: "percentage of properly segmented networks"
```

### Security Reporting

**Automated Security Reports:**
```bash
#!/bin/bash
# security-report.sh

echo "=== Stan's ML Stack Security Report ==="
echo "Generated: $(date)"
echo ""

echo "=== Vulnerability Summary ==="
trivy image --format json bartholemewii/stans-ml-stack:latest | jq '.Results[0].Vulnerabilities | group_by(.Severity) | map({severity: .[0].Severity, count: length})'

echo ""
echo "=== Compliance Status ==="
kubectl get cm -n compliance soc2-controls -o yaml | yq '.data.controls.yaml'

echo ""
echo "=== Recent Security Events ==="
kubectl logs -n monitoring -l app=falco --since=24h | grep "priority.*HIGH\|priority.*CRITICAL"

echo ""
echo "=== Access Control Summary ==="
kubectl get roles,rolebindings -n ml-stack | wc -l

echo ""
echo "=== Network Policy Status ==="
kubectl get networkpolicies -n ml-stack

echo ""
echo "=== Pod Security Standards ==="
kubectl get pods -n ml-stack -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.securityContext.runAsNonRoot}{"\n"}{end}'

echo ""
echo "=== End of Report ==="
```

---

## 🚨 Incident Response

### Security Incident Response Playbook

**Incident Classification:**
```yaml
incident_classification:
  critical:
    - data_breach: "Unauthorized access to sensitive data"
    - ransomware: "Ransomware infection detected"
    - privilege_escalation: "Unauthorized privilege escalation"
    - persistent_threat: "Advanced persistent threat detected"

  high:
    - malware_detection: "Malware detected on ML infrastructure"
    - dos_attack: "Denial of service attack in progress"
    - suspicious_activity: "Suspicious activity patterns detected"
    - policy_violation: "Security policy violations detected"

  medium:
    - vulnerability_disclosure: "New vulnerability disclosed"
    - configuration_drift: "Security configuration drift detected"
    - access_anomaly: "Unusual access patterns detected"

  low:
    - security_scan_failure: "Security scan failures"
    - minor_policy_violation: "Minor security policy violations"
    - outdated_dependencies: "Outdated dependencies detected"
```

**Response Procedures:**
```yaml
incident_response:
  detection:
    - automated_monitoring: "Continuous monitoring and alerting"
    - security_scanning: "Regular vulnerability and configuration scanning"
    - log_analysis: "Security log analysis and correlation"
    - user_reporting: "Security incident reporting mechanisms"

  analysis:
    - impact_assessment: "Assess impact and scope of incident"
    - root_cause_analysis: "Identify root cause of security incident"
    - evidence_collection: "Collect and preserve forensic evidence"
    - threat_intelligence: "Gather threat intelligence on attack vectors"

  containment:
    - isolate_affected: "Isolate affected systems and networks"
    - disable_compromised: "Disable compromised accounts and credentials"
    - block_malicious: "Block malicious IP addresses and domains"
    - implement_controls: "Implement additional security controls"

  eradication:
    - remove_malware: "Remove malware and malicious artifacts"
    - patch_vulnerabilities: "Patch security vulnerabilities exploited"
    - rebuild_systems: "Rebuild compromised systems from known-good state"
    - update_security: "Update security controls and configurations"

  recovery:
    - restore_services: "Restore services from clean backups"
    - validate_security: "Validate security of restored systems"
    - monitor_activity: "Enhanced monitoring of recovered systems"
    - update_documentation: "Update documentation and procedures"

  lessons_learned:
    - incident_review: "Conduct post-incident review"
    - identify_improvements: "Identify security improvements"
    - update_procedures: "Update incident response procedures"
    - security_training: "Provide security awareness training"
```

---

## 📞 Security Support

### Security Contacts

**Security Team:**
- **Security Engineer**: security@company.com
- **Security Operations**: soc@company.com
- **Compliance Officer**: compliance@company.com
- **Incident Response**: incident@company.com

**External Support:**
- **Security Vendor**: vendor@security-company.com
- **Legal Counsel**: legal@company.com
- **PR/Communications**: communications@company.com

### Security Resources

**Documentation:**
- **Security Policies**: https://wiki.company.com/security/policies
- **Incident Response**: https://wiki.company.com/security/incident-response
- **Compliance Framework**: https://wiki.company.com/security/compliance
- **Security Tools**: https://wiki.company.com/security/tools

**Training:**
- **Security Awareness**: https://training.company.com/security-awareness
- **Secure Development**: https://training.company.com/secure-development
- **Incident Response**: https://training.company.com/incident-response

---

## ✅ Security Hardening Checklist

### Pre-Deployment Security
- [ ] Container image security scanning completed
- [ ] Vulnerability assessment performed
- [ ] Security policies reviewed and approved
- [ ] Network security zones configured
- [ ] Access control implemented
- [ ] Encryption configured
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery tested
- [ ] Compliance requirements validated
- [ ] Security incident response plan updated

### Post-Deployment Security
- [ ] Security controls verified operational
- [ ] Monitoring dashboards configured
- [ ] Log aggregation working
- [ ] Alert notifications tested
- [ ] Access controls validated
- [ ] Security scanning automated
- [ ] Compliance reporting configured
- [ ] Incident response procedures tested
- [ ] Security training completed
- [ ] Documentation updated

---

## 🎯 Continuous Security Improvement

### Security Roadmap
- **Phase 1**: Implement basic security controls and monitoring
- **Phase 2**: Enhance threat detection and response capabilities
- **Phase 3**: Automate security operations and compliance
- **Phase 4**: Implement advanced security analytics and AI/ML security
- **Phase 5**: Achieve security excellence and thought leadership

### Security Metrics
- **Security Posture Score**: Overall security effectiveness metric
- **Mean Time to Detect (MTTD)**: Average time to detect security incidents
- **Mean Time to Respond (MTTR)**: Average time to respond to security incidents
- **Vulnerability Remediation Time**: Average time to patch vulnerabilities
- **Compliance Score**: Overall compliance with security frameworks

---

*This security hardening guide is part of the comprehensive enterprise documentation suite for Stan's ML Stack. For additional security information, please refer to the [Complete Documentation Index](../ENTERPRISE_DOCUMENTATION_INDEX.md).*