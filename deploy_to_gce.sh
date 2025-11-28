#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
ZONE="us-central1-a"
INSTANCE_NAME="stock-prediction-vm"
MACHINE_TYPE="e2-medium" # 2 vCPU, 4GB RAM

if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project selected. Run 'gcloud config set project <PROJECT_ID>' first."
    exit 1
fi

echo "Deploying to Project: $PROJECT_ID in Zone: $ZONE"

# 1. Create VM Instance
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
    echo "Instance $INSTANCE_NAME already exists. Skipping creation."
else
    echo "Creating VM instance $INSTANCE_NAME..."
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=20GB
    
    echo "Waiting for instance to initialize..."
    sleep 30
fi

# 2. Upload Code
echo "Uploading code to instance..."
# Exclude venv, .git, __pycache__ to speed up upload
# Create a temporary tarball
tar --exclude='venv' --exclude='__pycache__' --exclude='.git' -czf stock_prediction.tar.gz .

# Upload tarball
gcloud compute scp stock_prediction.tar.gz $INSTANCE_NAME:~/ --zone=$ZONE

# Cleanup local tarball
rm stock_prediction.tar.gz

# 3. Execute Installation and Run
echo "Executing installation and run script on instance..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    tar -xzf stock_prediction.tar.gz
    chmod +x install_and_run.sh
    ./install_and_run.sh
"

echo "Deployment and execution completed."
echo "To SSH into the instance: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
