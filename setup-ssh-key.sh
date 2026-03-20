#!/bin/bash
# Inject SSH_PUBLIC_KEY env var into root's authorized_keys at container startup.
# On vast.ai this is handled by the platform; this script covers local Docker runs.

if [[ -n "${SSH_PUBLIC_KEY}" ]]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "${SSH_PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    sort -u /root/.ssh/authorized_keys -o /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "SSH public key injected."
else
    echo "WARNING: SSH_PUBLIC_KEY not set — password auth only."
fi
