#!/usr/bin/env bash
# Builds the Lambda deployment package (.zip) with all Python dependencies.
# Run from the project root: bash deploy/build_lambda.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/lambda_package"
ZIP_PATH="$ROOT_DIR/build/credit_inference.zip"
HANDLER_SRC="$ROOT_DIR/src/ml_arch_recommender/lambda_function/handler.py"

echo "==> Limpiando build anterior..."
rm -rf "$BUILD_DIR" "$ZIP_PATH"
mkdir -p "$BUILD_DIR"

echo "==> Instalando dependencias en el paquete..."
pip install \
  scikit-learn \
  joblib \
  numpy \
  boto3 \
  pyyaml \
  --target "$BUILD_DIR" \
  --quiet

echo "==> Copiando handler de Lambda..."
cp "$HANDLER_SRC" "$BUILD_DIR/lambda_function.py"

echo "==> Comprimiendo paquete..."
cd "$BUILD_DIR"
zip -r "$ZIP_PATH" . -x "*.pyc" -x "*/__pycache__/*" > /dev/null
cd "$ROOT_DIR"

SIZE_KB=$(du -k "$ZIP_PATH" | cut -f1)
echo "==> Paquete listo: $ZIP_PATH (${SIZE_KB} KB)"
echo ""
echo "Siguiente paso: bash deploy/deploy.sh"
