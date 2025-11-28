#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1" # Change as needed
REPO_NAME="stock-prediction-repo"
IMAGE_NAME="stock-prediction-app"
JOB_NAME="stock-prediction-job"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project selected. Run 'gcloud config set project <PROJECT_ID>' first."
    exit 1
fi

echo "Deploying to Project: $PROJECT_ID in Region: $REGION"

# 1. Enable required services
echo "Enabling required services..."
gcloud services enable artifactregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com

# 2. Create Artifact Registry repository if it doesn't exist
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION &>/dev/null; then
    echo "Creating Artifact Registry repository..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Repository for Stock Prediction App"
fi

# 3. Build and Push Docker Image
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"
echo "Building and Pushing image to $IMAGE_URI..."
# Using Cloud Build for easier authentication and building
gcloud builds submit --tag $IMAGE_URI .

# 4. Create/Update Cloud Run Job
echo "Deploying Cloud Run Job..."
gcloud run jobs deploy $JOB_NAME \
    --image $IMAGE_URI \
    --region $REGION \
    --tasks 1 \
    --max-retries 0 \
    --memory 4Gi \
    --cpu 2 \
    --task-timeout 60m

# 5. Execute the Job
echo "Executing Cloud Run Job..."
gcloud run jobs execute $JOB_NAME --region $REGION

echo "Job execution started. Monitor logs with:"
echo "gcloud run jobs logs tail $JOB_NAME --region $REGION"
