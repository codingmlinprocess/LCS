#!/bin/bash
# Setup Anaconda + LCS conda env
# TODO: modify WORK_DIR if your repo path differs

GITHUB_TOKEN=""
HUGGINGFACE_TOKEN=""
WORK_DIR="${WORK_DIR:-/mnt/localssd/workspace/lcs}"

# setup anaconda ###############################################################################
echo "Downloading Anaconda..."
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh

echo "Installing Anaconda..."
bash Anaconda3-2025.06-0-Linux-x86_64.sh -b -p $HOME/anaconda3

echo "Initializing conda..."
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"

echo "Verifying conda..."
conda --version


# setup github ###############################################################################
echo "Installing GitHub CLI..."
if command -v apt-get &> /dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install gh -y
elif command -v yum &> /dev/null; then
    sudo dnf install 'dnf-command(config-manager)'
    sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
    sudo dnf install gh -y
fi

echo "Setting up SSH key..."
mkdir -p ~/.ssh
cd ~/.ssh
ssh-keygen -t rsa -C 'lcs-setup' -f ~/.ssh/id_rsa -N ""

echo "Starting ssh-agent..."
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

echo "Logging into GitHub CLI..."
echo "$GITHUB_TOKEN" | gh auth login --with-token

echo "Adding SSH key to GitHub..."
gh ssh-key add ~/.ssh/id_rsa.pub --title "setup-script-key"

echo "Testing GitHub connection..."
ssh-keyscan -H github.com >> ~/.ssh/known_hosts
ssh -T git@github.com

# setup repo ###############################################################################
echo "Cloning repository..."
mkdir -p "$(dirname "${WORK_DIR}")"

if [ -d "${WORK_DIR}/.git" ]; then
    echo "Repo already exists at ${WORK_DIR}, skipping clone"
    cd "${WORK_DIR}"
    echo "Updating repo..."
    git pull || echo "Update failed, please check manually"
else
    echo "Cloning https://github.com/codingmlinprocess/LCS.git ..."
    git clone https://github.com/codingmlinprocess/LCS.git "${WORK_DIR}"
    cd "${WORK_DIR}"
fi

# setup simlingo conda environment ###############################################################################
echo "Accepting conda ToS..."
$HOME/anaconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/anaconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "Enabling conda..."
$HOME/anaconda3/bin/conda init bash
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"

echo "Creating simlingo env..."
if $HOME/anaconda3/bin/conda env list | grep -q "^simlingo "; then
    echo "Env simlingo already exists, skipping"
else
    cd "${WORK_DIR}"
    if [ -f "environment_simplified.yaml" ]; then
        echo "Creating env from environment_simplified.yaml..."
        $HOME/anaconda3/bin/conda env create -f environment_simplified.yaml
    else
        echo "WARNING: environment_simplified.yaml not found, creating python=3.8 env..."
        $HOME/anaconda3/bin/conda create -n simlingo python=3.8 -y
    fi
fi

echo "Entering project dir..."
cd "${WORK_DIR}"

echo "Activating simlingo..."
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate simlingo

if [ "$CONDA_DEFAULT_ENV" != "simlingo" ]; then
    echo "WARNING: activate may have failed, trying fallback..."
    source activate simlingo || {
        echo "ERROR: could not activate simlingo"
        exit 1
    }
fi

echo "Current conda env: $CONDA_DEFAULT_ENV"

echo "Purging pip cache..."
pip cache purge

echo "Setting environment variables..."
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

if ! grep -q "export PYTHONPATH.*workspace/lcs\|export PYTHONPATH.*LCS" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# LCS environment" >> ~/.bashrc
    echo "export PYTHONPATH=\"${WORK_DIR}:\${PYTHONPATH:-}\"" >> ~/.bashrc
    echo "export TOKENIZERS_PARALLELISM=false" >> ~/.bashrc
    echo "export OMP_NUM_THREADS=8" >> ~/.bashrc
    if ! grep -q "conda initialize" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# >>> conda initialize >>>" >> ~/.bashrc
        echo "eval \"\$($HOME/anaconda3/bin/conda shell.bash hook)\"" >> ~/.bashrc
        echo "# <<< conda initialize <<<" >> ~/.bashrc
    fi
fi

echo "Setup done!"
echo "  - Conda env: $CONDA_DEFAULT_ENV"
echo "  - Python: $(which python)"
echo "  - WORK_DIR: ${WORK_DIR}"
echo "  - PYTHONPATH: $PYTHONPATH"
echo ""
echo "In a new terminal, run: source ~/.bashrc"
