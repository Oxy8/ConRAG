FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch \
    && python -m pip install -e .

COPY docker/compose-entrypoint.sh /usr/local/bin/conrag-compose
RUN chmod +x /usr/local/bin/conrag-compose

ENTRYPOINT ["conrag-compose"]
CMD ["run"]
