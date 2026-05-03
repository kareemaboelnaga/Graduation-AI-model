FROM tensorflow/tensorflow:2.15.0

WORKDIR /ai

# -----------------------------
# System dependencies (OpenCV + rendering + ML libs)
# -----------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Upgrade pip tooling (important for TF images)
# -----------------------------
RUN pip install --upgrade pip setuptools wheel

# -----------------------------
# Copy requirements first (better Docker caching)
# -----------------------------
COPY ./AI/requirements.txt .

# -----------------------------
# Install Python dependencies
# FIX: ignore system-installed packages like blinker
# FIX: prevents uninstall errors in TF base image
# -----------------------------
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# -----------------------------
# Copy application code
# -----------------------------
COPY ./AI .

# -----------------------------
# Expose Flask port
# -----------------------------
EXPOSE 5000

# -----------------------------
# Run application
# -----------------------------
CMD ["python", "Model.py"]