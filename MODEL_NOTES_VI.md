# Thay đổi so với CoTMix gốc

## File được thay
1. `algorithms.py`
   - Giữ `CoTMix.update(src_x, src_y, trg_x)` không nhận target labels.
   - Giữ temporal mixup theo chiều thời gian.
   - Giữ bốn thành phần loss: source CE, target conditional entropy, source supervised contrastive, target cross-view NT-Xent.
   - Dùng `DINOv3TimeSeriesEncoder` thay backbone CNN.

2. `models/models.py`
   - Thêm `DINOv3TimeSeriesEncoder`.
   - Thêm adapter tín hiệu thành ảnh 2D 3 kênh và lấy `x_norm_clstoken` từ DINOv3.
   - Giữ tên `classifier` để dễ tương thích với flow gốc.

## File mới
- `kuhar_data.py`: đọc raw recordings, nội suy về 100 Hz, chia cửa sổ 300 điểm, đọc subsamples 1803 cột, chia target theo recording file.
- `train_kuhar.py`: trainer độc lập cho KU-HAR, target labels không được truyền vào update, lưu checkpoint .pth.
- `app.py`: giao diện Streamlit dự đoán từ CSV.
- `requirements_kuhar.txt`: dependencies bổ sung.
- `README_VI.md`: hướng dẫn setup, chạy train và deploy giao diện.

## Tại sao không sửa `trainer.py`/`main.py` gốc?
Trainer gốc phụ thuộc cấu hình dataset AdaTime và dữ liệu `.pt` được chuẩn bị trước. Thay vì làm hỏng pipeline các dataset cũ, bộ này thêm `train_kuhar.py` dùng loader riêng cho KU-HAR. Nếu cần tích hợp ngược vào trainer gốc, phải bổ sung dataset config và backbone config riêng tương ứng.
