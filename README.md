# CoTMix-DINOv3-HAR

![CI-CD](https://github.com/Mducki/CoTMix-DinoV3-HAR/actions/workflows/ci-cd.yml/badge.svg)

Nhận dạng hoạt động của con người (Human Activity Recognition - HAR) từ dữ liệu cảm biến gia tốc và con quay hồi chuyển (bộ dữ liệu **KU-HAR**, 18 lớp hoạt động), bằng phương pháp thích nghi miền không giám sát **CoTMix** với backbone **DINOv3**.

Repo gồm hai phần chính:

| Phần | Nội dung |
|---|---|
| [Phần 1](#phần-1-thay-đổi-so-với-cotmix-gốc) | Những gì đã sửa so với [CoTMix gốc](https://github.com/emadeldeen24/CoTMix), thay backbone CNN bằng DINOv |
| [Phần 2](#phần-2-cicd-và-docker) | Quy trình CI/CD: tự động kiểm tra, build và đẩy Docker image lên Docker Hub |

---

# Phần 1. Thay đổi so với CoTMix gốc

## 1.1. Nguồn gốc

- **CoTMix** (Eldele et al., *Contrastive Domain Adaptation for Time-Series via Temporal Mixup*, IEEE Transactions on Artificial Intelligence, 2024, doi: 10.1109/TAI.2023.3293473): https://github.com/emadeldeen24/CoTMix. Mã nguồn và dữ liệu thí nghiệm của CoTMix gốc được lấy từ framework AdaTime.
- **DINOv3** (Meta AI): https://github.com/facebookresearch/dinov3. Thư mục `dinov3/` trong repo này là mã nguồn backbone.

Repo này là một bản tích hợp tùy chỉnh. Toàn bộ ý tưởng thuật toán (temporal mixup và các hàm loss) thuộc về CoTMix gốc; phần đóng góp ở đây là thay backbone, thêm pipeline dữ liệu và huấn luyện cho KU-HAR, và thêm giao diện dự đoán.

## 1.2. Ý tưởng chính: thay backbone CNN bằng DINOv3

CoTMix gốc dùng một backbone CNN nhỏ cho chuỗi thời gian. Ở đây backbone được thay bằng **DINOv3 ViT-S/16**, một mô hình thị giác đã được huấn luyện trước (pretrained), sau đó fine-tune trên KU-HAR. Không cần pretrain DINOv3 từ đầu.

Vì DINOv3 là backbone cho **ảnh** còn dữ liệu đầu vào là **tín hiệu cảm biến**, cần một lớp chuyển đổi (adapter):

```
Tín hiệu cảm biến [B, 6, T]      (6 kênh: 3 trục gia tốc + 3 trục gyro, T = 300 mẫu)
        |
        v
Adapter: xếp tín hiệu thành ảnh 2D (6 cảm biến x thời gian), mở rộng thành 3 kênh,
         resize về 224 x 224
        |
        v
DINOv3 ViT-S/16  ->  lấy vector đặc trưng x_norm_clstoken
        |
        v
classifier (dropout 0.1)  ->  18 lớp hoạt động
```

Lưu ý khoa học: DINOv3 gốc không được thiết kế cho tín hiệu cảm biến. Cách chuyển tín hiệu thành ảnh ở trên là một adapter do dự án đề xuất, **không phải** công thức đầu vào chính thức của DINOv3.

## 1.3. Những thứ được giữ nguyên từ CoTMix

| Thành phần | Trạng thái |
|---|---|
| Chữ ký `CoTMix.update(src_x, src_y, trg_x)` | Giữ nguyên. Hàm **không nhận nhãn của miền đích**, đúng tinh thần unsupervised domain adaptation |
| Temporal mixup theo chiều thời gian | Giữ nguyên |
| Bốn thành phần loss | Giữ nguyên: cross-entropy trên source, conditional entropy trên target, supervised contrastive trên source, NT-Xent chéo giữa hai view trên target |

Giá trị mặc định trong `algorithms.py`:

| Tham số | Mặc định |
|---|---|
| Optimizer | AdamW, `learning_rate = 1e-5`, `weight_decay = 1e-4` |
| `mix_ratio` | 0.5 |
| `temporal_shift` | 10 |
| Trọng số loss | `src_cls = 1.0`, `trg_entropy = 0.1`, `src_supCon = 0.1`, `trg_cont = 0.1` |
| Nhiệt độ (temperature) contrastive | 0.2 |
| Cắt gradient (`clip_grad_norm_`) | 5.0 |

## 1.4. File được thay đổi

| File | Thay đổi |
|---|---|
| `algorithms.py` | Lớp `CoTMix` dùng `DINOv3TimeSeriesEncoder` thay cho backbone CNN. Giữ nguyên logic loss và temporal mixup |
| `models/models.py` | Thêm `DINOv3TimeSeriesEncoder` (adapter tín hiệu thành ảnh 2D 3 kênh, lấy `x_norm_clstoken` từ DINOv3). Giữ tên `classifier` để tương thích với luồng gốc |

## 1.5. File thêm mới

| File / thư mục | Vai trò |
|---|---|
| `dinov3/` | Mã nguồn backbone DINOv3 |
| `kuhar_data.py` | Đọc dữ liệu thô của KU-HAR, nội suy về 100 Hz, chia cửa sổ 300 điểm, đọc file subsample 1803 cột, chia miền target theo từng file recording |
| `train_kuhar.py` | Trainer riêng cho KU-HAR. Nhãn target không được truyền vào bước update. Lưu checkpoint `.pth` |
| `app.py` | Giao diện Streamlit: tải file CSV cảm biến lên và nhận dự đoán hoạt động |
| `requirements_kuhar.txt` | Thư viện bổ sung cho huấn luyện (cài thêm PyTorch bản CUDA theo hướng dẫn chính thức) |
| `requirements.txt` | Thư viện cho ứng dụng và Docker image (torch bản CPU) |
| `README_VI.md` | Hướng dẫn chi tiết cài đặt môi trường, tải weights, sắp xếp dữ liệu, chạy huấn luyện |
| `MODEL_NOTES_VI.md` | Ghi chú các thay đổi so với CoTMix gốc |
| `overlay/`, `.devcontainer/` | Hỗ trợ môi trường phát triển |

So với CoTMix gốc, repo này **không kèm** `main.py`, `trainer.py`, `dataloader.py` và các thư mục `configs/`, `data/`, `misc/`.

## 1.6. Vì sao không sửa `main.py` và `trainer.py` gốc

Trainer gốc phụ thuộc vào cấu hình dataset của AdaTime và các file dữ liệu `.pt` đã chuẩn bị sẵn theo định dạng riêng. Thay vì làm hỏng pipeline cho các dataset cũ, dự án thêm `train_kuhar.py` với bộ nạp dữ liệu riêng cho KU-HAR. Muốn tích hợp ngược vào trainer gốc, cần bổ sung cấu hình dataset và cấu hình backbone tương ứng.

## 1.7. Cách chia miền source và target

- CoTMix là phương pháp **unsupervised domain adaptation**: batch nguồn có nhãn `(src_x, src_y)`, batch đích chỉ đưa tín hiệu `trg_x` vào `update`. Nhãn target không dùng để tính loss.
- Dữ liệu `Time_domain_subsamples` của KU-HAR không có mã người tham gia (participant ID) rõ ràng. Chia ngẫu nhiên các đoạn này thành source/target chỉ để thử pipeline, không chứng minh domain shift thực sự, và các đoạn liên quan có thể lọt vào cả hai miền.
- Muốn thí nghiệm UDA nghiêm túc, dùng `Raw_time_domain_data` và chia thành hai thư mục source/target theo participant hoặc recording ID. Nếu dataset không có thông tin này, cần ghi rõ giới hạn trong báo cáo.


# Phần 2. CI/CD và Docker

## 2.1. Tổng quan

Mỗi lần có code mới đẩy lên GitHub, **GitHub Actions** tự động kiểm tra code, build Docker image, chạy thử, rồi đẩy image lên **Docker Hub** ở chế độ công khai. Nhờ đó bất kỳ ai cũng có thể chạy ứng dụng chỉ bằng hai lệnh Docker, không cần cài môi trường Python.

```
git push  -->  GitHub Actions
                  |
                  +-- Job 1: ci               kiểm tra cú pháp, thử import module
                  +-- Job 2: docker-test      build image, chạy thử container
                  +-- Job 3: build-and-push   đăng nhập Docker Hub, đẩy image
                                   |
                                   v
                          Docker Hub (public)
                                   |
                                   v
                       docker pull / docker run
```

Mỗi job chỉ chạy khi job trước thành công, nên image lỗi không bao giờ lên Docker Hub.

## 2.2. Các file liên quan

| File | Vai trò |
|---|---|
| `.github/workflows/ci-cd.yml` | Định nghĩa toàn bộ pipeline |
| `Dockerfile` | Công thức đóng gói ứng dụng thành image |
| `.dockerignore` | Danh sách thứ không đưa vào image (`.git`, `.env`, file model `*.pth`, `secrets.toml`...) |

## 2.3. Khi nào pipeline chạy

| Sự kiện | Job chạy | Kết quả trên Docker Hub |
|---|---|---|
| Push lên nhánh `main` | `ci`, `docker-test`, `build-and-push` | Tag `latest` và `sha-xxxxxxx` |
| Push tag dạng `v1.0.0` | Cả ba job | Thêm tag `1.0.0` |
| Mở Pull Request vào `main` | Chỉ `ci` và `docker-test` | Không đẩy gì lên |

## 2.4. Chi tiết ba job

**Job `ci`: kiểm tra code**

- Cài Python 3.11 và các thư viện trong `requirements.txt`.
- Chạy `flake8` để bắt lỗi nghiêm trọng: lỗi cú pháp (`E9`), so sánh/assert sai (`F63`), lỗi câu lệnh (`F7`) và tên biến chưa định nghĩa (`F82`).
- Chạy `python -c "import algorithms, kuhar_data"` để phát hiện thiếu thư viện hoặc lỗi khi nạp module.

**Job `docker-test`: build và chạy thử**

- Build image từ `Dockerfile` nhưng chưa đẩy đi đâu.
- Chạy container, rồi gọi `http://localhost:8501/_stcore/health` tối đa 30 lần, mỗi lần cách 3 giây (chờ tối đa 90 giây). Ứng dụng phản hồi thì thành công; hết thời gian thì in log container và báo lỗi.
- Job này chỉ kiểm tra image build được và Streamlit khởi động được. **Phần dự đoán không được test tự động** vì cần `HF_TOKEN` để tải model.

**Job `build-and-push`: đẩy image**

- Bỏ qua khi sự kiện là Pull Request.
- Đăng nhập Docker Hub bằng hai secret `DOCKERHUB_USERNAME` và `DOCKERHUB_TOKEN`.
- Sinh tag tự động và đẩy image lên Docker Hub.


## Giấy phép và ghi công

Vui lòng xem giấy phép của từng dự án gốc trước khi sử dụng hoặc phân phối lại:

- CoTMix: https://github.com/emadeldeen24/CoTMix
- DINOv3: https://github.com/facebookresearch/dinov3 (cần chấp nhận điều khoản truy cập model weights theo hướng dẫn chính thức)
- HAR dataset: https://www.kaggle.com/datasets/mducki/human-activity-recognition
