#!/bin/bash

apt update -y
apt install -y docker.io
systemctl start docker
systemctl enable docker
docker pull keretdodor/yolo5
docker run -d --restart always \
  -e AWS_REGION="${AWS_REGION}" \
  -e DYNAMODB_TABLE="${DYNAMODB_TABLE}" \
  -e S3_BUCKET="${S3_BUCKET}" \
  -e SQS_QUEUE_URL="${SQS_QUEUE_URL}" \
  -e ALIAS_RECORD="${ALIAS_RECORD}" \
  keretdodor/yolo5