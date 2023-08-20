#!/usr/bin/env bash
VENV_NAME="lore"
echo "Setting up environment in .${VENV_NAME} ..."
if [ ! -d ".${VENV_NAME}" ]; then \
  virtualenv --system-site-packages .${VENV_NAME}
fi
export BUILD_SPLIT_CUDA=ON
source "./.${VENV_NAME}/bin/activate"
echo "Welcome to lore the publicist dev env"
echo "You are in a virtual env. Please Run "pip install . --user" to install non-nix dependencies"
