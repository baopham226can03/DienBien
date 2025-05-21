FROM python:3.11-slim

WORKDIR /code

# Cài các dependencies cần thiết và dọn dẹp cache apt trong cùng một layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Nâng cấp pip
RUN pip install --no-cache-dir --upgrade pip

# Sao chép chỉ requirements.txt trước để tận dụng cache Docker
COPY ./requirements.txt /code/requirements.txt

# Dùng PyPI chính thức thay vì Tsinghua mirror
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt

# Sao chép phần còn lại của code
COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
