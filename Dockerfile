# Wayfinder - reference image build.
#
# The final stage MUST start from the mandated ROCm base (checked by layer identity).
# Never squash or flatten this image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/site:/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/models/Qwen3-VL-8B-Instruct \
    WAYFINDER_BACKEND=transformers \
    WAYFINDER_INDEX_DIR=/app/index

# 1) Model weights (~17 GB bf16). Fetch them on the build host first:
#        hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
#    Nothing is downloaded at evaluation time: the container has no network then.
COPY models/Qwen3-VL-8B-Instruct /models/Qwen3-VL-8B-Instruct

# 2) Python packages that the base does not ship. --no-deps keeps the ROCm torch untouched.
COPY requirements-image.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --target /opt/site -r /app/requirements.txt

# 3) Application code (standard library only, plus the model server).
COPY app/app.py app/server.py /app/
COPY wayfinder /app/wayfinder
RUN mkdir -p /app/output /app/corpus /app/index \
 && python3 -c "import sys; sys.path.insert(0, '/app'); import wayfinder.entry, wayfinder.server"

# The supervisor starts the resident model server, restarts it on crash and never exits,
# so the harness can `docker exec ... python3 /app/app.py --query-id q01 --query ...`.
CMD ["python3", "/app/server.py"]
