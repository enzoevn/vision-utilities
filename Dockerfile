FROM nvcr.io/nvidia/pytorch:24.05-py3 

# Create user
RUN groupadd -g 1000 enzo && \
    useradd -u 1000 -g enzo enzo

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Install requirements.txt
WORKDIR /home/enzo
COPY requirements.txt /home/enzo
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python
RUN pip install opencv-python

USER enzo
