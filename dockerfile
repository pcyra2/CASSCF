FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV CUDA_HOME="/usr/local/cuda" 
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64::${LD_LIBRARY_PATH}" 

RUN echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bash.bashrc
RUN echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bash.bashrc
ENV OMP_NUM_THREADS="12"
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

RUN pip3 install  --prefer-binary pyscf && \
    pip3 install numpy scipy openfermion cupy-cuda12x qiskit qiskit-nature qiskit-algorithms ase
    

RUN pip3 install cutensor-cu12  &&\
    pip3 install nvidia-cublas-cu12 &&\
    pip3 install nvidia-pyindex  &&\
    pip3 install pytest



WORKDIR /app
ADD casscf ./casscf
ADD setup.py .

RUN pip install -e . 

# CMD ["lspci"]
WORKDIR /app/data
# ENTRYPOINT ["/bin/bash", "-l", "-c"]
CMD ["/bin/bash"]
