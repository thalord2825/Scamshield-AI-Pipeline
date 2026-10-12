# ScamShield-AI-Pipeline

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-2--Tier%20Cascaded%20Router-orange.svg)]()
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-blueviolet.svg)](https://git-lfs.com/)

> **Microservice & Standalone AI Inference Engine** chuyên dụng phát hiện tin nhắn lừa đảo tiếng Việt thời gian thực (SMS, OTT, Zalo, Telegram, Facebook) dựa trên kiến trúc phân tầng **2-Tier Cascaded AI Router**.

---

## 🏛️ Kiến trúc 2-Tier Cascaded Router

```
                      [ Tin nhắn người dùng ]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   TIER 1 (Fast-Path)  │
                     │ ViSoBERT ONNX on CPU  │
                     │  Latency: < 45ms      │
                     │  Chi phí: $0.00       │
                     └───────────┬───────────┘
                                 │
                   Độ tự tin c >= τ (0.90)?
                         /               \
                   CÓ  /                   \  KHÔNG (Phân vân)
                     ▼                       ▼
          ┌─────────────────────┐  ┌───────────────────────────────────┐
          │  PHẢN HỒI TỨC THÌ   │  │        TIER 2 (Deep Forensic)     │
          │  (80-85% lưu lượng) │  │   Hội đồng Pháp y 5 Tác tử LLM    │
          │  Không tốn phí Cloud │  │ (Gemini, Claude, GPT, DeepSeek)  │
          └─────────────────────┘  │   - Borda Consensus               │
                                   │   - Anti-Alarm-Fatigue Defender   │
                                   │   - Early-Exit (4-1 / 5-0)        │
                                   └─────────────────┬─────────────────┘
                                                     │
                                                     ▼
                                          [ Phán quyết chi tiết ]
```

1. **Tier 1 (Fast-Path Local CPU)**:
   - Mô hình nền tảng: Fine-tuned **ViSoBERT** xuất sang **ONNX Runtime INT8/FP32**.
   - Kỹ thuật chuẩn hóa: **Temperature Scaling** ($T^*=1.0$) và **Cost-Sensitive WBCE** ($\alpha=5.0$) chống lọt lừa đảo (False Negative).
   - Bộ lọc Heuristics tiếng Việt bắt nhanh các mẫu Brand Spoofing, tên miền lạ (`.xyz`, `.top`), bẫy OTP.
2. **Tier 2 (Slow-Path Cloud Forensic Council)**:
   - Kích hoạt khi độ tự tin Tier 1 $< \tau = 0.90$ (khoảng 15–20% lưu lượng khó, mơ hồ).
   - Hội đồng 5 Agent chuyên môn hóa:
     1. `Cyber Threat & Malicious URL Inspector` (Chuyên gia kỹ thuật IOC & link)
     2. `Social Engineering & Urgency Profiler` (Chuyên gia tâm lý cưỡng ép, tạo gấp gáp)
     3. `Vietnamese Linguistic & Teencode Analyst` (Chuyên gia teencode, lóng tiếng Việt)
     4. `Banking & Financial Protocol Verifier` (Chuyên gia quy chuẩn ngân hàng)
     5. `Public Defender / Anti-Alarm-Fatigue Advocate` (Chuyên gia phản biện chống báo động giả)
   - Tối ưu hiệu năng: Cơ chế **Early-Exit** (dừng sớm khi đạt đa số áp đảo 4-1 hoặc 5-0), **Circuit Breaker** (tự ngắt và chuyển model dự phòng khi lỗi 429).

---

## 📁 Cấu trúc Thư mục

```text
Scamshield-AI-Pipeline/
├── .env.example              # Mẫu biến môi trường API keys
├── .gitattributes            # Cấu hình Git LFS (track *.onnx, *.safetensors)
├── .gitignore                # Bỏ qua bytecode, venv, secrets
├── Dockerfile                # Build container độc lập
├── docker-compose.yml        # Chạy microservice với 1 lệnh
├── requirements.txt          # Danh sách thư viện Python
├── README.md                 # Tài liệu hướng dẫn
├── demo_scamshield_mvp.py    # Demo tương tác dòng lệnh
├── run_demo.bat              # Shortcut chạy demo trên Windows
│
├── api/                      # REST API phục vụ Backend & App
│   ├── app.py                # FastAPI endpoints (/api/v1/scan, /health)
│   └── schemas.py            # Pydantic data contracts (chuẩn hóa I/O)
│
├── config/                   # Cấu hình độc lập (Zero-hardcoding)
│   ├── council_registry.yaml # Cấu hình 5 Chuyên gia Hội đồng Pháp y
│   └── gemini_models.yaml    # Cấu hình Dynamic Model Registry
│
├── models/                   # Trọng số & Tokenizer Tier 1 (Quản lý qua Git LFS)
│   ├── visobert_tier1.onnx   # ONNX Model (~390MB)
│   ├── scamshield_config.json
│   ├── tokenizer.json
│   └── config.json
│
├── pipeline/                 # Core AI Engine
│   ├── two_tier_router.py    # Orchestrator định tuyến 2 tầng
│   ├── local_classifier.py   # ViSoBERT Local Classifier
│   ├── forensic_council.py   # Multi-Agent Forensic Council
│   └── model_registry.py     # Dynamic Google Gemini Registry
│
├── scripts/                  # Scripts benchmark & fine-tuning
│   └── evaluate_council_benchmark.py
│
└── tests/                    # Kiểm thử tự động
    └── test_api.py
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy

### 1. Chuẩn bị Môi trường

```bash
# Clone repository
git clone https://github.com/thalord2825/Scamshield-AI-Pipeline.git
cd Scamshield-AI-Pipeline

# Kích hoạt Git LFS để kéo file mô hình lớn (*.onnx)
git lfs install
git lfs pull
```

### 2. Cài đặt Python Dependencies

```bash
# Khuyến nghị sử dụng Python 3.10 - 3.12
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Cấu hình Biến Môi trường

Sao chép `.env.example` thành `.env`:
```bash
cp .env.example .env
```
Điền `GEMINI_API_KEY` (và các API key tùy chọn khác nếu dùng đa nhà cung cấp cho Tier 2).

### 4. Khởi chạy Microservice REST API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation tự động có sẵn tại:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

### 5. Chạy bằng Docker / Docker Compose

```bash
# Chạy với Docker Compose
docker compose up -d --build

# Kiểm tra log
docker compose logs -f
```

---

## 📡 API Endpoints

### 1. Kiểm tra sức khỏe hệ thống: `GET /health`
```json
{
  "status": "healthy",
  "version": "1.1.0",
  "service": "ScamShield AI Pipeline Engine",
  "tier1_ready": true,
  "tier2_ready": true
}
```

### 2. Quét tin nhắn lừa đảo: `POST /api/v1/scan`

**Request:**
```json
{
  "content": "Vietcombank: Tai khoan cua ban tam khoa do bat thuong. Dang nhap https://vcb-xacthuc.xyz de mo khoa.",
  "check_type": "TEXT"
}
```

**Response (Chuẩn hóa cho Spring Boot & Frontend):**
```json
{
  "status": "success",
  "data": {
    "scan_id": "SC-20261012-A8F12B",
    "timestamp": "2026-10-12T15:15:20.123456Z",
    "tier_executed": 1,
    "verdict": {
      "risk_level": "DANGEROUS",
      "risk_score": 100.0,
      "scam_category": "Detected by Tier 1 Heuristic/Pattern",
      "confidence": 0.9997,
      "summary": "Xử lý trực tiếp bởi Tier 1 (PhoBERT Local Fast-Path) với độ tự tin 100.0% >= 90%."
    },
    "radar_metrics": {
      "urgency": 85,
      "authority_impersonation": 95,
      "financial_lure": 90,
      "suspicious_channel": 85,
      "data_harvesting": 90
    },
    "actionable_recommendations": [
      "Tuyệt đối không nhấp vào bất kỳ đường link nào trong tin nhắn.",
      "Không cung cấp mật khẩu ngân hàng, mã OTP, số CCCD hoặc thông tin sinh trắc học.",
      "Thông báo cho người thân, chặn số liên lạc và gửi báo cáo lừa đảo lên hệ thống."
    ]
  }
}
```

---

## 🧪 Chạy Demo & Kiểm thử

- **Chạy console demo kiểm tra 10 benchmark test cases:**
  ```bash
  python demo_scamshield_mvp.py
  ```
- **Chạy demo chế độ tương tác nhập tin nhắn trực tiếp:**
  ```bash
  python demo_scamshield_mvp.py --interactive
  ```

---

## 🛡️ License & Team
Phát triển bởi đội ngũ ScamShield-VN Core AI Engineering Team.
Mọi đóng góp tuân thủ quy chuẩn mã nguồn mở và bảo mật dữ liệu an toàn thông tin.
