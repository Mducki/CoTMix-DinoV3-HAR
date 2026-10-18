import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch

from algorithms import CoTMix
from kuhar_data import CLASS_NAMES, read_raw_recording, recording_to_windows, SensorDataset


st.set_page_config(page_title="KU-HAR Activity Recognition", page_icon="🏃", layout="wide")
st.title("KU-HAR Human Activity Recognition")
st.caption("CoTMix + DINOv3 | Dự đoán hoạt động từ Accelerometer và Gyroscope")

with st.sidebar:
    st.header("Cấu hình")
    checkpoint_path = st.text_input(
        "Đường dẫn checkpoint .pth",
        value="checkpoints/cotmix_dinov3_kuhar.pth"
    )
    dinov3_repo = st.text_input("Thư mục mã DINOv3", value="dinov3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Thiết bị: **{device}**")
    if device.type == "cuda":
        st.write(torch.cuda.get_device_name(0))

@st.cache_resource
def load_model(checkpoint_path, dinov3_repo, device_name):
    device_local = torch.device(device_name)
    ckpt = torch.load(checkpoint_path, map_location=device_local, weights_only=False)
    hp = ckpt.get("hparams", {})
    model = CoTMix(
        configs=None,
        hparams=hp,
        device=device_local,
        dinov3_repo=dinov3_repo,
        dinov3_weights=None,
        pretrained=False,
        image_size=int(ckpt.get("image_size", 224)),
        num_classes=len(ckpt.get("class_names", CLASS_NAMES)),
    ).to(device_local)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = np.asarray(ckpt["normalization_mean"], dtype=np.float32)
    std = np.asarray(ckpt["normalization_std"], dtype=np.float32)
    names = ckpt.get("class_names", CLASS_NAMES)
    return model, mean, std, names, device_local, ckpt


uploaded = st.file_uploader(
    "Tải CSV dữ liệu cảm biến",
    type=["csv"],
    help="Hỗ trợ raw CSV 8 cột (timestamp + 6 sensor axes), CSV 6 cột, hoặc subsamples KU-HAR 1803 cột."
)

st.info(
    "Raw CSV: mỗi bản ghi được chia thành cửa sổ 3 giây (300 điểm ở 100 Hz). "
    "Time_domain_subsamples: một hàng chứa 1800 giá trị tín hiệu và cột nhãn cuối không được dùng để dự đoán."
)

if uploaded is not None:
    if not os.path.isfile(checkpoint_path):
        st.error(f"Không tìm thấy checkpoint: {checkpoint_path}")
        st.stop()
    if not os.path.isdir(dinov3_repo):
        st.error(f"Không tìm thấy thư mục DINOv3: {dinov3_repo}")
        st.stop()

    try:
        model, mean, std, class_names, device_local, ckpt = load_model(
            checkpoint_path, dinov3_repo, str(device)
        )
        uploaded.seek(0)
        df = pd.read_csv(uploaded, header=None, engine="python", on_bad_lines="skip")
        df_numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
        if df_numeric.shape[1] >= 1801:
            row = df_numeric.iloc[0, :1800].to_numpy(dtype=np.float32)
            X = row.reshape(1, 6, 300)
            data_kind = "KU-HAR subsample"
        else:
            uploaded.seek(0)
            X = recording_to_windows(uploaded)
            data_kind = "Raw recording"

        ds = SensorDataset(X, None, mean, std)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        rows = []
        with torch.no_grad():
            for i, x in enumerate(loader):
                x = x.to(device_local)
                logits = model.classifier(model.feature_extractor(x))
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
                for j in range(len(pred)):
                    rows.append({
                        "Cửa sổ": len(rows) + 1,
                        "Hoạt động dự đoán": class_names[int(pred[j].item())],
                        "Độ tin cậy (%)": round(float(conf[j].item()) * 100, 2),
                    })
        result_df = pd.DataFrame(rows)
        st.subheader("Kết quả")
        st.write(f"Loại dữ liệu: **{data_kind}**")
        st.dataframe(result_df, use_container_width=True)
        counts = result_df["Hoạt động dự đoán"].value_counts()
        st.metric("Hoạt động phổ biến nhất", counts.index[0])
        st.download_button(
            "Tải kết quả CSV",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="kuhar_predictions.csv",
            mime="text/csv",
        )
        st.caption(
            f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')} | "
            f"Best target macro-F1 khi train: {ckpt.get('best_target_macro_f1', 'N/A')}"
        )
    except Exception as exc:
        st.exception(exc)
else:
    st.write("Tải lên một file CSV để xem dự đoán.")
