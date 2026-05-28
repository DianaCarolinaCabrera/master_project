#!/usr/bin/env bash
# Full AWS deployment: S3 bucket, Lambda function, API Gateway.
# Prerequisites: AWS CLI configured, model trained, Lambda package built.
# Run from project root: bash deploy/deploy.sh

set -euo pipefail

# Load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Required variables (with defaults)
AWS_REGION="${AWS_REGION:-us-east-1}"
MODEL_BUCKET="${MODEL_BUCKET:-microcredit-demo-models}"
CONFIG_KEY="${CONFIG_KEY:-models/config.yaml}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-credit-risk-inference}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-LambdaMLRole}"
API_NAME="${API_NAME:-CreditRiskAPI}"
API_STAGE="${API_STAGE:-prod}"
ZIP_PATH="build/credit_inference.zip"
MODEL_PATH="data/models/credit_model.joblib"
CONFIG_PATH="data/models/config.yaml"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

echo "======================================"
echo " Despliegue: ML Arch Recommender Demo"
echo " Region:  $AWS_REGION"
echo " Account: $ACCOUNT_ID"
echo "======================================"
echo ""

# ── PASO 1: Crear bucket S3 ────────────────────────────────────────────────
echo "[1/6] Creando bucket S3: $MODEL_BUCKET"
if aws s3 ls "s3://$MODEL_BUCKET" --region "$AWS_REGION" > /dev/null 2>&1; then
  echo "      Bucket ya existe, omitiendo creación."
else
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3 mb "s3://$MODEL_BUCKET" --region "$AWS_REGION"
  else
    aws s3api create-bucket \
      --bucket "$MODEL_BUCKET" \
      --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION"
  fi
  echo "      Bucket creado."
fi

# ── PASO 2: Subir config.yaml y modelo ────────────────────────────────────
echo "[2/6] Subiendo config.yaml y modelo a S3..."
if [ ! -f "$CONFIG_PATH" ]; then
  echo "      ERROR: config.yaml no encontrado en $CONFIG_PATH"
  echo "      Ejecuta primero: poetry run train-model"
  exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "      ERROR: modelo no encontrado en $MODEL_PATH"
  echo "      Ejecuta primero: poetry run train-model"
  exit 1
fi
aws s3 cp "$CONFIG_PATH" "s3://$MODEL_BUCKET/$CONFIG_KEY"
echo "      Config subido: s3://$MODEL_BUCKET/$CONFIG_KEY"
MODEL_FILE=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c['paths']['model_file'])")
aws s3 cp "$MODEL_PATH" "s3://$MODEL_BUCKET/$MODEL_FILE"
echo "      Modelo subido: s3://$MODEL_BUCKET/$MODEL_FILE"

# ── PASO 3: Crear rol IAM ──────────────────────────────────────────────────
echo "[3/6] Configurando rol IAM: $LAMBDA_ROLE_NAME"
if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" > /dev/null 2>&1; then
  echo "      Rol ya existe."
else
  aws iam create-role \
    --role-name "$LAMBDA_ROLE_NAME" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"lambda.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }]
    }' > /dev/null

  aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

  aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

  echo "      Esperando propagación del rol (15s)..."
  sleep 15
  echo "      Rol creado y políticas adjuntadas."
fi

# ── PASO 4: Crear/actualizar Lambda ───────────────────────────────────────
echo "[4/6] Desplegando función Lambda: $LAMBDA_FUNCTION_NAME"
if [ ! -f "$ZIP_PATH" ]; then
  echo "      ERROR: paquete no encontrado en $ZIP_PATH"
  echo "      Ejecuta primero: bash deploy/build_lambda.sh"
  exit 1
fi

LAMBDA_EXISTS=$(aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" 2>&1 || true)

ENV_VARS="Variables={MODEL_BUCKET=$MODEL_BUCKET,CONFIG_KEY=$CONFIG_KEY}"

if echo "$LAMBDA_EXISTS" | grep -q "Function"; then
  echo "      Actualizando código de la función..."
  aws lambda update-function-code \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_PATH" \
    --region "$AWS_REGION" > /dev/null

  aws lambda update-function-configuration \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --environment "$ENV_VARS" \
    --region "$AWS_REGION" > /dev/null
  echo "      Función actualizada."
else
  aws lambda create-function \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --runtime python3.11 \
    --handler lambda_function.lambda_handler \
    --role "$ROLE_ARN" \
    --zip-file "fileb://$ZIP_PATH" \
    --timeout 30 \
    --memory-size 512 \
    --environment "$ENV_VARS" \
    --region "$AWS_REGION" > /dev/null
  echo "      Función creada."
fi

LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" \
  --query 'Configuration.FunctionArn' --output text --region "$AWS_REGION")

# ── PASO 5: API Gateway ────────────────────────────────────────────────────
echo "[5/6] Configurando API Gateway: $API_NAME"

API_ID=$(aws apigateway get-rest-apis \
  --query "items[?name=='$API_NAME'].id" \
  --output text --region "$AWS_REGION")

if [ -z "$API_ID" ]; then
  API_ID=$(aws apigateway create-rest-api \
    --name "$API_NAME" \
    --description "Demo API - Tesis MLOps" \
    --query 'id' --output text --region "$AWS_REGION")
  echo "      API creada: $API_ID"
fi

ROOT_ID=$(aws apigateway get-resources \
  --rest-api-id "$API_ID" \
  --query 'items[?path==`/`].id' \
  --output text --region "$AWS_REGION")

RESOURCE_ID=$(aws apigateway get-resources \
  --rest-api-id "$API_ID" \
  --query 'items[?path==`/predict`].id' \
  --output text --region "$AWS_REGION")

if [ -z "$RESOURCE_ID" ]; then
  RESOURCE_ID=$(aws apigateway create-resource \
    --rest-api-id "$API_ID" \
    --parent-id "$ROOT_ID" \
    --path-part predict \
    --query 'id' --output text --region "$AWS_REGION")
fi

# Create POST method (ignore error if exists)
aws apigateway put-method \
  --rest-api-id "$API_ID" \
  --resource-id "$RESOURCE_ID" \
  --http-method POST \
  --authorization-type NONE \
  --region "$AWS_REGION" > /dev/null 2>&1 || true

# Lambda integration
aws apigateway put-integration \
  --rest-api-id "$API_ID" \
  --resource-id "$RESOURCE_ID" \
  --http-method POST \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri "arn:aws:apigateway:$AWS_REGION:lambda:path/2015-03-31/functions/$LAMBDA_ARN/invocations" \
  --region "$AWS_REGION" > /dev/null

# Lambda permission for API Gateway
aws lambda add-permission \
  --function-name "$LAMBDA_FUNCTION_NAME" \
  --statement-id "apigateway-prod-$(date +%s)" \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:$AWS_REGION:$ACCOUNT_ID:$API_ID/*/POST/predict" \
  --region "$AWS_REGION" > /dev/null 2>&1 || true

# Deploy
aws apigateway create-deployment \
  --rest-api-id "$API_ID" \
  --stage-name "$API_STAGE" \
  --region "$AWS_REGION" > /dev/null

ENDPOINT="https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/${API_STAGE}/predict"

# ── PASO 6: Resumen ────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo " DESPLIEGUE COMPLETADO"
echo "======================================"
echo ""
echo " Endpoint: $ENDPOINT"
echo ""
echo " Prueba rápida:"
echo " curl -X POST '$ENDPOINT' \\"
echo "   -H 'Content-Type: application/json' \\"
echo "   -d '{\"<feature_1>\":valor,\"<feature_2>\":valor,...}'"
echo ""
echo " Las features exactas dependen de config.yaml (campo 'features:')"
echo ""
echo " Actualiza API_ENDPOINT en tu .env:"
echo " echo 'API_ENDPOINT=$ENDPOINT' >> .env"
