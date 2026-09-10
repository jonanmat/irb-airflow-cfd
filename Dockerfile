FROM opencfd/openfoam-default:2312
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FOAM_BASHRC=/usr/lib/openfoam/openfoam2312/etc/bashrc \
    IRB_JOBS_DIR=/tmp/irb-cfd-jobs \
    IRB_MAX_RUNTIME_SECONDS=1800 \
    IRB_MAX_CONCURRENT_JOBS=1 \
    IRB_MAX_ESTIMATED_CELLS=350000
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt
COPY server.py /app/server.py
COPY public /app/public
ENTRYPOINT []
EXPOSE 10000
CMD ["bash", "-lc", "source /usr/lib/openfoam/openfoam2312/etc/bashrc && exec python3 -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
