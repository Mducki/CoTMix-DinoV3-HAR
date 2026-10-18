# CoTMix-DINOv3-KU-HAR (Windows + NVIDIA GPU)

Dự án này là một bản tích hợp tùy chỉnh dựa trên repository CoTMix:
https://github.com/emadeldeen24/CoTMix
và DINOv3:
https://github.com/facebookresearch/dinov3

## Lưu ý khoa học quan trọng

- CoTMix là phương pháp unsupervised domain adaptation (UDA): batch nguồn có nhãn `(src_x, src_y)`, batch đích chỉ đưa tín hiệu `trg_x` vào hàm update. Nhãn target không được dùng để tính loss.
- KU-HAR `Time_domain_subsamples` không có participant ID rõ ràng theo định dạng được mô tả trong README dataset. Chia ngẫu nhiên các đoạn subsample thành source/target chỉ là thử pipeline UDA, không chứng minh domain shift thực sự và có nguy cơ các đoạn liên quan lọt vào hai miền.
- Để làm thí nghiệm UDA nghiêm túc, dùng `Raw_time_domain_data`, sắp xếp các bản ghi thành hai thư mục source/target theo participant hoặc recording ID. Nếu dataset không cung cấp participant mapping, ghi rõ giới hạn đó trong báo cáo.
- DINOv3 gốc là backbone ảnh. Mô hình này biến tín hiệu `[B, 6, T]` thành ảnh xám 2D (6 cảm biến × thời gian), resize thành 224×224, rồi dùng DINOv3 ViT-S/16. Đây là adapter đề xuất cho tín hiệu, không phải input recipe chính thức của DINOv3.

## 1. Cấu trúc dự án

Sau khi clone CoTMix, sao chép/ghi đè các file trong gói này theo đúng vị trí:

- `algorithms.py` -> thay file gốc `CoTMix/algorithms.py`
- `models/models.py` -> thay file gốc `CoTMix/models/models.py`
- `kuhar_data.py` -> file mới tại thư mục gốc
- `train_kuhar.py` -> file mới tại thư mục gốc
- `app.py` -> file mới tại thư mục gốc
- `requirements_kuhar.txt` -> file mới tại thư mục gốc

`main.py`, `trainer.py`, `dataloader.py` gốc không cần dùng cho pipeline này vì chúng dựa trên cấu hình dataset AdaTime và file `.pt` đã được chuẩn hóa theo định dạng riêng. Pipeline mới vẫn giữ ý tưởng/loss update của `CoTMix` nhưng có data loader và trainer riêng cho KU-HAR.

## 2. Cài môi trường Windows

Khuyến nghị Python 3.10 hoặc 3.11. Tránh bắt đầu bằng Python 3.14 vì nhiều thư viện ML có thể chưa tương thích đầy đủ.

Mở PowerShell:

```powershell
git clone https://github.com/emadeldeen24/CoTMix.git
cd CoTMix
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch CUDA theo lệnh hiện hành trên trang chính thức:
https://pytorch.org/get-started/locally/
Sau đó:

```powershell
pip install -r requirements_kuhar.txt
git clone https://github.com/facebookresearch/dinov3.git
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Nếu `torch.cuda.is_available()` trả `False`, sửa PyTorch/CUDA trước khi train.

## 3. Tải DINOv3 pretrained weights

DINOv3 yêu cầu chấp nhận điều khoản truy cập model weights theo hướng dẫn chính thức:
https://github.com/facebookresearch/dinov3

Chọn backbone `dinov3_vits16` và tải checkpoint pretrained tương ứng. Đặt file tại, ví dụ:

`CoTMix/checkpoints/dinov3_vits16_pretrain.pth`

Tên file thật có thể khác. Truyền đúng đường dẫn bằng `--dinov3-weights`.

Không cần chạy pretraining DINOv3 từ đầu. Chương trình fine-tune backbone pretrained cho tác vụ KU-HAR.

## 4. Cách sắp xếp dữ liệu Raw để tránh rò rỉ

Tạo thư mục theo lớp hoạt động:

```text
CoTMix/
  data/
    raw_domains/
      source/
        Stand/
          recording_001.csv
          recording_002.csv
        Sit/
          recording_003.csv
        Talk-sit/
        Talk-stand/
        Stand-sit/
        Lay/
        Lay-stand/
        Pick/
        Jump/
        Push-up/
        Sit-up/
        Walk/
        Walk-backward/
        Walk-circle/
        Run/
        Stair-up/
        Stair-down/
        Table-tennis/
      target/
        Stand/
          recording_101.csv
          recording_102.csv
        Sit/
        Talk-sit/
        Talk-stand/
        Stand-sit/
        Lay/
        Lay-stand/
        Pick/
        Jump/
        Push-up/
        Sit-up/
        Walk/
        Walk-backward/
        Walk-circle/
        Run/
        Stair-up/
        Stair-down/
        Table-tennis/
```

Mỗi file CSV phải là **một bản ghi gốc** của một hoạt động. Đưa các bản ghi khác nhau vào source và target; không copy cùng file sang cả hai miền. Nếu có participant ID, tốt nhất đưa toàn bộ bản ghi của một người chỉ vào một miền. Tên thư mục lớp phải khớp một trong 18 tên trên.

Raw CSV có thể có 8 cột theo mô tả KU-HAR: thời gian accelerometer, 3 trục gia tốc, thời gian gyroscope, 3 trục gyro. Code cũng hỗ trợ CSV chỉ có 6 cột cảm biến. Dữ liệu được nội suy về 100 Hz và cắt cửa sổ 300 điểm.

Trong mỗi lớp của thư mục `target`, cần ít nhất 2 file recording để chia file thành target-train không nhãn và target-test có nhãn. Nếu thiếu, hãy bổ sung recording hoặc dùng `--mode subsamples` để kiểm tra pipeline, nhưng không diễn giải đó là đánh giá UDA theo participant.

## 5. Chạy train với Raw recordings (khuyến nghị)

Từ thư mục gốc `CoTMix`:

```powershell
python train_kuhar.py `
  --mode raw `
  --source-dir data/raw_domains/source `
  --target-dir data/raw_domains/target `
  --dinov3-repo dinov3 `
  --dinov3-weights checkpoints/dinov3_vits16_pretrain.pth `
  --epochs 20 `
  --batch-size 8 `
  --lr 0.00001 `
  --output checkpoints/cotmix_dinov3_kuhar.pth
```

Nếu VRAM thấp, thử `--batch-size 4`. Nếu GPU đủ VRAM, có thể thử 8 hoặc 16. Lúc đầu nên chạy 1 epoch để kiểm tra data và shape:

```powershell
python train_kuhar.py --mode raw --source-dir data/raw_domains/source --target-dir data/raw_domains/target --dinov3-repo dinov3 --dinov3-weights checkpoints/dinov3_vits16_pretrain.pth --epochs 1 --batch-size 4 --output checkpoints/test.pth
```

Checkpoint được lưu ở `checkpoints/cotmix_dinov3_kuhar.pth` (hoặc đường dẫn `--output`), gồm `model_state_dict`, optimizer, nhãn lớp và thống kê chuẩn hóa.
