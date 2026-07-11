FROM python:3.11-slim

WORKDIR /app

# Set environment variables to fix permissions
ENV HOME=/tmp
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV HF_HOME=/tmp/.cache/huggingface

# Copy requirements and install dependencies.
# Install CPU-only torch first (Spaces are CPU); avoids pulling multi-GB CUDA
# wheels that overflow the build disk.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/.streamlit && \
    mkdir -p /tmp/.cache && \
    chmod -R 777 /tmp

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]