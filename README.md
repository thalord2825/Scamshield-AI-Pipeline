# ScamShield-AI-Pipeline

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-2--Tier%20Cascaded%20Router-orange.svg)]()
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-blueviolet.svg)](https://git-lfs.com/)

> **Microservice và bộ máy suy luận AI độc lập** chuyên phát hiện tin nhắn lừa đảo tiếng Việt theo thời gian thực trên SMS, OTT, Zalo, Telegram và Facebook, sử dụng kiến trúc định tuyến AI phân tầng hai cấp.

---

## 🏛️ Kiến trúc bộ định tuyến AI hai cấp

```
                      [ Tin nhắn người dùng ]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   TIER 1 (Fast-Path)  │
                     │ ViSoBERT ONNX / PyTorch│
                     │ Local inference        │
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

1. **Cấp 1 (phân loại cục bộ)**:
   - Bộ phân loại **ViSoBERT** đã tinh chỉnh chạy bằng ONNX Runtime trên CPU khi có tệp ONNX hợp lệ; nếu không, hệ thống thử dùng checkpoint PyTorch. Tệp ONNX hiện được xuất ở độ chính xác thông thường (FP32), chưa được lượng tử hóa INT8.
   - Khi huấn luyện, mô hình dùng Cross Entropy có trọng số lớp, với trọng số lớp lừa đảo là **5.0**. Nhiệt độ hiệu chỉnh được học trên tập xác thực; giá trị lưu trong báo cáo huấn luyện hiện tại là **T = 1.4015**. Bộ phân loại áp dụng các ngưỡng 0.35 và 0.85 để đưa ra một trong ba kết quả: HAM, REVIEW hoặc SCAM.
   - Không có bộ lọc từ khóa riêng để quyết định Brand Spoofing, tên miền đáng ngờ hay bẫy OTP. Một số heuristics tạo chỉ số radar phục vụ trình bày kết quả; chúng không thay đổi nhãn phân loại của ViSoBERT.
2. **Cấp 2 (hội đồng phân tích chuyên sâu trên đám mây)**:
   - Kích hoạt khi độ tự tin Tier 1 $< \tau = 0.90$ (khoảng 15–20% lưu lượng khó, mơ hồ).
   - Hội đồng 5 Agent chuyên môn hóa:
     1. `Cyber Threat & Malicious URL Inspector` (Chuyên gia kỹ thuật IOC & link)
     2. `Social Engineering & Urgency Profiler` (Chuyên gia tâm lý cưỡng ép, tạo gấp gáp)
     3. `Vietnamese Linguistic & Teencode Analyst` (Chuyên gia teencode, lóng tiếng Việt)
     4. `Banking & Financial Protocol Verifier` (Chuyên gia quy chuẩn ngân hàng)
     5. `Public Defender / Anti-Alarm-Fatigue Advocate` (Chuyên gia phản biện chống báo động giả)
   - Tối ưu hiệu năng: Cơ chế **Early-Exit** (dừng sớm khi đạt đa số áp đảo 4-1 hoặc 5-0), **Circuit Breaker** (tự ngắt và chuyển sang mô hình dự phòng khi gặp lỗi 429).

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
│   ├── visobert_classifier.py# ViSoBERT Production Classifier (ONNX CPU Fast-Path + PyTorch)
│   ├── forensic_council.py   # 5-Agent Multi-LLM Forensic Council (Maker-Checker & Defender Veto)
│   └── model_registry.py     # Dynamic Cloud Model Registry
│
├── scripts/                  # Scripts benchmark, training & ONNX export
│   ├── train_visobert.py     # Tinh chỉnh ViSoBERT với trọng số lớp SCAM = 5.0
│   ├── export_visobert_onnx.py # Xuất ONNX Runtime opset 17 (FP32)
│   └── evaluate_council_benchmark.py
│
└── tests/                    # Kiểm thử tự động
    └── test_api.py
```

---

## 🚀 Hướng dẫn cài đặt và chạy

### 1. Chuẩn bị môi trường

```bash
# Sao chép kho mã nguồn
git clone https://github.com/thalord2825/Scamshield-AI-Pipeline.git
cd Scamshield-AI-Pipeline

# Kích hoạt Git LFS để tải các tệp mô hình lớn (*.onnx)
git lfs install
git lfs pull
```

### 2. Cài đặt thư viện Python

```bash
# Khuyến nghị dùng Python 3.10–3.12
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường

Tạo tệp `.env` từ mẫu `.env.example`:
```bash
cp .env.example .env
```
Điền `GEMINI_API_KEY` và các khóa API tùy chọn nếu sử dụng nhiều nhà cung cấp cho Cấp 2.

### 4. Khởi chạy dịch vụ REST API

Endpoint quét lưu kết quả hoàn tất vào PostgreSQL. Khi chạy cục bộ, hãy khởi động
các dịch vụ bằng Compose để có sẵn cơ sở dữ liệu, hoặc đặt `DATABASE_URL` trỏ đến
một máy chủ PostgreSQL có thể truy cập. Hai endpoint `POST /api/v1/scan` và
`POST /api/v1/visobert/predict` nhận tùy chọn `core_check_id` là số nguyên dương.
Truyền `id` tương ứng trong bảng `check_requests` của Core để các lần thử lại cập
nhật cùng một tác vụ. Nếu không truyền, dịch vụ tự tạo mã tham chiếu nội bộ và
ghi mỗi yêu cầu thành một tác vụ riêng. Hãy truyền mã Core khi cần loại bỏ các
tác vụ trùng lặp do thử lại.

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Tài liệu API được tạo tự động tại:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

### 5. Chạy bằng Docker hoặc Docker Compose

```bash
# Khởi động các dịch vụ bằng Docker Compose
docker compose up -d --build

# Theo dõi nhật ký hoạt động
docker compose logs -f
```

Compose khởi động PostgreSQL có bật pgvector trước dịch vụ AI. Trong lần khởi tạo
cơ sở dữ liệu đầu tiên, Postgres chạy `db/init/001_ai_service.sql`, tệp DDL tương
ứng với `ScamShield_App/docs/database/ai-service.dbml`. Lược đồ tạo các bảng tác
vụ AI, mô hình, suy luận và RAG. Các tập lệnh khởi tạo chỉ chạy khi volume dữ liệu
Postgres còn trống. Nếu volume đã có cơ sở dữ liệu, hãy áp dụng SQL một lần bằng
lệnh `psql -f db/init/001_ai_service.sql`.

Trước khi triển khai ngoài môi trường phát triển cục bộ, hãy đặt `POSTGRES_PASSWORD` trong tệp `.env`.

---

## 📡 Các endpoint API

### 1. Kiểm tra trạng thái hệ thống: `GET /health`
```json
{
  "status": "healthy",
  "version": "1.1.0",
  "service": "ScamShield AI Pipeline Engine",
  "tier1_ready": true,
  "tier2_ready": true,
  "database_ready": true
}
```

### 2. Quét tin nhắn lừa đảo: `POST /api/v1/scan`

**Yêu cầu:**
```json
{
  "content": "Vietcombank: Tai khoan cua ban tam khoa do bat thuong. Dang nhap https://vcb-xacthuc.xyz de mo khoa.",
  "check_type": "TEXT"
}
```

**Phản hồi (định dạng chuẩn hóa cho Spring Boot và giao diện người dùng):**
```json
{
  "status": "success",
  "data": {
    "scan_id": "SC-20261018-140FCE",
    "timestamp": "2026-10-18T14:17:23.126590+00:00",
    "tier_executed": 1,
    "verdict": {
      "risk_level": "DANGEROUS",
      "risk_score": 99.6,
      "scam_category": "ViSoBERT text classifier",
      "confidence": 0.996,
      "summary": "ViSoBERT classified this message as SCAM with calibrated scam probability 0.9960."
    },
    "radar_metrics": {
      "urgency": 15,
      "authority_impersonation": 95,
      "financial_lure": 15,
      "suspicious_channel": 85,
      "data_harvesting": 10
    },
    "council_breakdown": null,
    "actionable_recommendations": [
      "Tuyệt đối không nhấp vào bất kỳ đường link nào trong tin nhắn.",
      "Không cung cấp mật khẩu ngân hàng, mã OTP, số CCCD hoặc thông tin sinh trắc học.",
      "Thông báo cho người thân, chặn số liên lạc và gửi báo cáo lừa đảo lên hệ thống."
    ],
    "telemetry": {
      "routing_path": "TIER_1_DIRECT_FASTPATH",
      "routing_reason": "ViSoBERT P(SCAM)=0.9960 falls outside REVIEW band [0.35, 0.85].",
      "total_latency_ms": 651.79,
      "total_cost_usd": 0,
      "tier1": {
        "label": "SCAM",
        "class_id": 1,
        "confidence": 0.9960368275642395,
        "scam_prob": 0.9960368275642395,
        "threshold_low": 0.35,
        "threshold_high": 0.85,
        "ham_prob": 0.004,
        "latency_ms": 651.26,
        "cost_usd": 0,
        "is_sure": true,
        "status": "CONFIDENT",
        "matched_cues": [],
        "reason": "ViSoBERT P(SCAM)=0.9960; REVIEW band=[0.35, 0.85]."
      },
      "final_label": "SCAM"
    }
  },
  "error": null
}
```

---

## 🧪 Huấn luyện, chạy demo và kiểm thử

### Tinh chỉnh ViSoBERT cho Cấp 1

Tệp dữ liệu huấn luyện là `data/raw/vietnamese_sms_merged_all.csv` (gồm các cột
`message,label`; có thể có thêm `message_id,source`). Tệp kiểm thử độc lập là
`data/raw/vietnamese_sms_test.csv` (gồm `message,label`). Nhãn `0=HAM` biểu thị
tin nhắn bình thường, `1=SCAM` biểu thị tin nhắn lừa đảo. Tập kiểm thử được cung
cấp có một số nội dung trùng với tệp huấn luyện; trước khi tạo tập xác thực phân
tầng, `scripts/train_visobert.py` loại các tin nhắn kiểm thử khỏi dữ liệu huấn
luyện. Quá trình sẽ dừng nếu phát hiện nhãn mâu thuẫn.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-training.txt
.venv/Scripts/python.exe scripts/train_visobert.py
```

Trong lần chạy đầu tiên, tập lệnh tải checkpoint chính thức `uitnlp/visobert` theo
phiên bản `model.safetensors` đã ghim. Trên Windows có trình điều khiển NVIDIA
tương thích, hãy cài bản PyTorch hỗ trợ CUDA từ kho chính thức của PyTorch trước
khi huấn luyện; gói pip mặc định có thể chỉ hỗ trợ CPU. Tệp kiểm thử có 597 dòng
nhưng chỉ gồm 566 tin nhắn khác nhau, vì vậy số mẫu kiểm thử được báo cáo là 566.

Tập lệnh tinh chỉnh đầu phân loại hai lớp với trọng số lớp lừa đảo bằng 5, chọn
checkpoint có macro-F1 trên tập xác thực cao nhất, hiệu chỉnh nhiệt độ từ logits
xác thực rồi đánh giá một lần trên tập kiểm thử độc lập. Các tệp mô hình,
tokenizer và `training_report.json` được lưu trong `models/visobert_finetuned/`.
Dùng các tùy chọn `--epochs`, `--batch-size`, `--max-length` và `--output-dir` để
điều chỉnh lần chạy. Checkpoint `.safetensors` được quản lý bằng Git LFS theo cấu
hình của kho mã nguồn.

### Xuất bộ phân loại đã huấn luyện sang ONNX

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-export.txt
.venv/Scripts/python.exe scripts/export_visobert_onnx.py
```

Lệnh này tạo `models/visobert_finetuned/visobert_classifier.onnx` và
`onnx_export_report.json`. Công cụ xuất kiểm tra đồ thị ONNX và so sánh logits
trên CPU của ONNX Runtime với PyTorch bằng một số tin nhắn mẫu. Thông tin hiệu
chỉnh và ngưỡng quyết định vẫn nằm trong `training_report.json`. API quét hiện
dùng checkpoint PyTorch; việc xuất sang ONNX chưa làm thay đổi bộ máy chạy.

- **Chạy bản demo dòng lệnh với 10 trường hợp kiểm thử chuẩn:**
  ```bash
  python demo_scamshield_mvp.py
  ```
- **Chạy demo tương tác để nhập tin nhắn trực tiếp:**
  ```bash
  python demo_scamshield_mvp.py --interactive
  ```

---

## 🛡️ Giấy phép và đội ngũ
Phát triển bởi đội ngũ ScamShield-VN Core AI Engineering Team.
Mọi đóng góp cần tuân thủ quy chuẩn mã nguồn mở và bảo đảm an toàn thông tin, bảo mật dữ liệu.
