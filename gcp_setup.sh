#!/usr/bin/env bash
set -euo pipefail

# Oracle Linux setup script
# - Installs Python 3.12 + devel, Node.js 24 module, misc deps
# - Installs CMake 3.31.1 and Neovim latest
# - Updates ~/.bashrc (idempotent) and sources it
# - Configures npm prefix + installs @openai/codex
# - Tightens ~/.ssh permissions
# - Writes ~/.gitconfig

log() { printf "\n\033[1;32m==> %s\033[0m\n" "$*"; }

require_cmd() {
  local c="$1"
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $c" >&2
    exit 1
  fi
}

append_if_missing() {
  local file="$1"
  local line="$2"
  mkdir -p "$(dirname "$file")"
  touch "$file"
  if ! grep -Fqx "$line" "$file"; then
    echo "$line" >> "$file"
  fi
}

write_gitconfig() {
  local file="$HOME/.gitconfig"
  log "Writing $file"
  cat > "$file" <<'EOF'
[user]
	name = leon modal
	email = leon@modal.com
	signingkey = /home/modal/.ssh/id_ed25519.pub
[gpg]
	format = ssh
[commit]
	gpgsign = true
EOF
}

install_cmake() {
  local ver="3.31.1"
  local sh="cmake-${ver}-linux-x86_64.sh"
  local url="https://github.com/Kitware/CMake/releases/download/v${ver}/${sh}"

  log "Installing CMake ${ver} to /usr/local"
  require_cmd wget

  if command -v cmake >/dev/null 2>&1; then
    local current
    current="$(cmake --version | head -n1 | awk '{print $3}' || true)"
    if [[ "${current}" == "${ver}" ]]; then
      log "CMake ${ver} already installed (cmake --version = ${current}); skipping"
      return
    fi
  fi

  wget -q --show-progress -O "$sh" "$url"
  sudo bash "$sh" --prefix=/usr/local --exclude-subdir --skip-license
  rm -f "$sh"
}

install_neovim() {
  local tar="nvim-linux-x86_64.tar.gz"
  local url="https://github.com/neovim/neovim/releases/latest/download/${tar}"

  log "Installing Neovim (latest) to /opt/nvim-linux-x86_64"
  require_cmd curl
  require_cmd tar

  curl -L -o "$tar" "$url"
  sudo rm -rf /opt/nvim-linux-x86_64
  sudo tar -C /opt -xzf "$tar"
  rm -f "$tar"
}

configure_bashrc() {
  local bashrc="$HOME/.bashrc"
  log "Updating ${bashrc}"

  if ! grep -Fq "### modal-dev-setup BEGIN" "$bashrc" 2>/dev/null; then
    cat >> "$bashrc" <<'EOF'

### modal-dev-setup BEGIN
export PATH="$PATH:/opt/nvim-linux-x86_64/bin"
export HF_HOME="/tmp"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
### modal-dev-setup END
EOF
  else
    append_if_missing "$bashrc" 'export PATH="$PATH:/opt/nvim-linux-x86_64/bin"'
    append_if_missing "$bashrc" 'export HF_HOME="/tmp"'
    append_if_missing "$bashrc" 'export PATH="/usr/local/cuda/bin:$PATH"'
    append_if_missing "$bashrc" 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"'
    append_if_missing "$bashrc" 'export CUDA_HOME="/usr/local/cuda"'
  fi

  log "Not sourcing ~/.bashrc from this script (avoids set -u issues)."
  log "Run this in your shell to apply changes now: source ~/.bashrc"
}

install_packages() {
  log "Installing packages via dnf"
  require_cmd dnf

  sudo dnf -y install python3.12 python3.12-devel
  sudo dnf -y module install nodejs:24
  sudo dnf -y install ninja-build numactl-devel ripgrep
}

install_uv() {
  log "Installing uv"
  require_cmd curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
}

install_claude() {
  log "Installing Claude Code"
  require_cmd curl
  curl -fsSL https://claude.ai/install.sh | bash
}

configure_node() {
  log "Configuring npm prefix and installing @openai/codex"
  require_cmd node
  require_cmd npm

  # The user wrote "nodejs config", but on Oracle Linux this is typically npm.
  # This matches the intent: set global prefix under ~/.local and install global packages.
  npm config set prefix "$HOME/.local"

  mkdir -p "$HOME/.local/bin"
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *)
      log "Note: ~/.local/bin is not on PATH for this session; consider adding it to your shell config."
      ;;
  esac

  npm i -g @openai/codex
}

fix_ssh_perms() {
  log "Setting ~/.ssh permissions"
  if [[ -d "$HOME/.ssh" ]]; then
    chmod 700 "$HOME/.ssh"
    # Only chmod keys that exist; avoid failing if none match.
    shopt -s nullglob
    local keys=("$HOME/.ssh"/id*)
    shopt -u nullglob
    if (( ${#keys[@]} > 0 )); then
      chmod 600 "${keys[@]}"
    else
      log "No ~/.ssh/id* files found; skipping chmod 600 ~/.ssh/id*"
    fi
  else
    log "~/.ssh does not exist; skipping ssh permission changes"
  fi
}

main() {
  log "Starting Oracle Linux dev setup"

  install_packages
  install_cmake
  install_neovim
  configure_bashrc
  configure_node
  fix_ssh_perms
  write_gitconfig
  install_uv
  install_claude

  log "Done."
  log "If you want the new env vars in your current terminal, run: source ~/.bashrc"
}

main "$@"
