# Face-Swap-Worker (05.10.2026) — FaceFusion 3.9.1 headless, onnxruntime-gpu (cuda@12) auf CUDA 12.8 (laeuft auch auf Blackwell sm_120).
# Modelle (hyperswap_1c_256, gfpgan_1.4, Detektor/Landmarker/Masken, Inhaltsfilter) werden beim Bauen per CPU-Probelauf geladen -> kein Download beim Kaltstart.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git ffmpeg curl ca-certificates libgl1 libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && pip install --upgrade pip setuptools wheel
RUN git clone --depth 1 --branch 3.9.1 https://github.com/facefusion/facefusion.git /opt/facefusion
WORKDIR /opt/facefusion
RUN python3 install.py cuda@12 --skip-conda && pip install runpod requests
# 07.10.: Auf RunPod-MIG-Scheiben meldet nvidia-smi den Speicher als "Insufficient Permissions" -> FaceFusion stuerzte beim int() ab.
RUN grep -q "'value': int(value)," facefusion/execution.py \
 && sed -i "s/'value': int(value),/'value': int(value) if value.isdigit() else 0,/" facefusion/execution.py \
 && grep -q "value.isdigit()" facefusion/execution.py
# Probelauf auf CPU: laedt genau die Modelle, die der Handler nutzt (schlaegt der Befehl fehl, scheitert der Build sichtbar)
RUN mkdir -p /tmp/ex && curl -sL -o /tmp/ex/source.jpg https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/source.jpg \
 && curl -sL -o /tmp/ex/target.mp4 https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/target-240p.mp4 \
 && python3 facefusion.py headless-run -s /tmp/ex/source.jpg -t /tmp/ex/target.mp4 -o /tmp/ex/out.mp4 \
      --processors face_swapper face_enhancer --face-swapper-model hyperswap_1c_256 --face-swapper-pixel-boost 512x512 \
      --face-enhancer-model gfpgan_1.4 --face-enhancer-blend 35 --face-selector-mode one --face-mask-types box occlusion \
      --execution-providers cpu --trim-frame-start 0 --trim-frame-end 8 --output-video-encoder libx264 --log-level info \
 && ls -la /opt/facefusion/.assets/models && test -s /tmp/ex/out.mp4 && rm -rf /tmp/ex
COPY handler.py /opt/facefusion/handler.py
CMD ["python3", "-u", "handler.py"]
