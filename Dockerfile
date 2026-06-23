FROM python:3.13-slim

# システムライブラリのインストール
# - ffmpeg           : 動画保存用
# - libgl1, libglib2 : OpenCVの動作に必須のグラフィック系ライブラリ
# - iputils-ping, arp-scan: スタッフWi-Fi検知用のpingおよびMACアドレススキャンツール
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    iputils-ping \
    arp-scan \
    && rm -rf /var/lib/apt/lists/*

# パッケージマネージャー uv のバイナリをコピーしてインストールを高速化
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 依存パッケージ定義のコピーとインストール
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r pyproject.toml

# ソースコードをコピー
COPY . .

# データ保存ディレクトリの作成と権限設定
RUN mkdir -p /app/data && chmod -R 777 /app/data

# デフォルトの起動プログラム（docker-composeで上書き可能にします）
CMD ["python", "occupancy_detector.py"]
