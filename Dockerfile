FROM vastai/pytorch:latest

# Ensure GPU is always visible regardless of launch flags
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# System packages: git, SSH server, Node.js/npm for Claude Code
RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# GPU verification script — base image already ships PyTorch/CUDA
COPY test_pytorch_gpu.py /workspace/test_pytorch_gpu.py

# SSH runtime dir (required by sshd)
RUN mkdir -p /run/sshd

# SSH key injection script — reads SSH_PUBLIC_KEY env var at startup
COPY setup-ssh-key.sh /opt/setup-ssh-key.sh
RUN chmod +x /opt/setup-ssh-key.sh

# Supervisor config to ensure SSH daemon starts on boot
COPY supervisord-extras.conf /etc/supervisor/conf.d/ssh-extras.conf

# Expose ports:
#   22    — SSH (VS Code Remote SSH, terminal)
#   6006  — TensorBoard
#   18080 — Jupyter Lab (base image default)
EXPOSE 22
EXPOSE 6006
EXPOSE 18080

# Make TensorBoard accessible via the vast.ai Instance Portal
ENV PORTAL_CONFIG="localhost:6006:6006:/tensorboard:TensorBoard"
