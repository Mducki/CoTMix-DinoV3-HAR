from huggingface_hub import hf_hub_download
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch
import traceback

from algorithms import CoTMix
from kuhar_data import CLASS_NAMES, recording_to_windows, SensorDataset

st.set_page_config(page_title="Dự đoán hành vi thông minh", page_icon="🧍", layout="wide")

st.markdown(r"""
<style>
:root { color-scheme: dark; }
.stApp { background: #05080d; color: #edf3ff; }
.block-container { max-width: 1500px; padding-top: 1.8rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
.hero { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:1.4rem; }
.hero h1 { color:#f4f7ff; font-size:2rem; font-weight:800; margin:0; }
.sub { color:#9baec4; margin-top:.35rem; font-size:.92rem; }
.status { background:#102d29; border:1px solid #155b50; color:#42e0b5; padding: .7rem 1rem; border-radius:9px; font-size:.85rem; white-space:nowrap; }
.section-label { color:#94a8bf; font-size:.78rem; font-weight:800; letter-spacing:.04em; margin: .6rem 0 .7rem; }
.panel { background:#101f33; border:1px solid #203a57; border-radius:24px; padding:1.35rem; margin-bottom:1rem; }
.result-panel { background:#0d1a2b; border:1px solid #203a57; border-radius:64px; padding:2rem; min-height: 450px; }
.result-inner { background:#101f33; border:1px solid #203a57; border-radius:28px; padding:1.6rem; }
.small-label { color:#9aacc0; font-size:.8rem; }
.prediction { font-size:2.1rem; font-weight:850; color:#f3f7ff; margin-top:.35rem; }
.confidence { font-size:1.9rem; font-weight:850; color:#28d5c0; }
.muted { color:#9aacc0; font-size:.85rem; }
div.stButton > button { border-radius:11px; min-height:3rem; font-weight:700; border:1px solid #254360; }
div.stButton > button[kind="primary"] { background:#27cdbb; color:#04131b; border:0; }
[data-testid="stFileUploader"] { background:#101f33; border:1px dashed #36516e; border-radius:15px; padding:1rem; }
hr { border-color:#20364f; }
footer { visibility:hidden; }
@media (max-width: 800px) { .hero {display:block;} .status {display:inline-block; margin-top:1rem;} .result-panel {border-radius:28px;padding:1rem;} }
</style>
""", unsafe_allow_html=True)

if "sensor_connected" not in st.session_state:
    st.session_state.sensor_connected = False
if "prediction_started" not in st.session_state:
    st.session_state.prediction_started = False

status_text = "Hệ thống đang hoạt động" if not st.session_state.sensor_connected else "Cảm biến đã kết nối"
st.markdown(f'''<div class="hero"><div><h1>Dự đoán hành vi thông minh</h1><div class="sub">Giám sát dữ liệu cảm biến • Phân tích hành vi từ mô hình CoTMix + DINOv3</div></div><div class="status">● &nbsp;{status_text}</div></div>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_model(checkpoint_path,
    dinov3_repo,
    device_name,
    dinov3_weights_path,):
    device_local = torch.device(device_name)
    ckpt = torch.load(checkpoint_path, map_location=device_local, weights_only=False)
    hp = ckpt.get("hparams", {})
    model = CoTMix(
        configs=None, hparams=hp, device=device_local, dinov3_repo=dinov3_repo,
        dinov3_weights=dinov3_weights_path, pretrained=False,
        image_size=int(ckpt.get("image_size", 224)),
        num_classes=len(ckpt.get("class_names", CLASS_NAMES)),
    ).to(device_local)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = np.asarray(ckpt["normalization_mean"], dtype=np.float32)
    std = np.asarray(ckpt["normalization_std"], dtype=np.float32)
    names = ckpt.get("class_names", CLASS_NAMES)
    return model, mean, std, names, device_local

def download_model_files():
    hf_token = st.secrets.get("HF_TOKEN", "")
    repo_id = st.secrets.get(
        "HF_MODEL_REPO",
        "mducki/cotmix-dinov3-kuhar"
    )

    if not hf_token:
        raise ValueError(
            "Chua cau hinh HF_TOKEN trong Streamlit Secrets."
        )

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename="cotmix_dinov3_kuhar.pth",
        token=hf_token
    )

    dinov3_weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="dinov3_vits16_pretrain.pth",
        repo_type="model",
        token=hf_token,
    )

    return checkpoint_path, dinov3_weights_path
left, right = st.columns([0.34, 0.66], gap="large")
with left:
    st.markdown('<div class="section-label">NGUỒN DỮ LIỆU</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><h3 style="margin:0;color:#f2f6ff">Tải tệp CSV</h3><p class="muted">Nạp dữ liệu cảm biến để dự đoán hoạt động</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Chọn tệp CSV", type=["csv"], label_visibility="collapsed", help="CSV dữ liệu cảm biến; raw recording nên có 6 trục gia tốc và con quay hồi chuyển.")
    st.markdown('<div class="panel"><h3 style="margin:0;color:#f2f6ff">Cảm biến trực tiếp</h3><p class="muted">Kết nối thiết bị cảm biến thời gian thực</p></div>', unsafe_allow_html=True)
    if st.button("●  Kết nối cảm biến", use_container_width=True):
        st.session_state.sensor_connected = False
        st.error("Không kết nối được cảm biến. Chưa có thiết bị/API cảm biến trực tiếp được cấu hình. Bạn vẫn có thể tải tệp CSV để dự đoán.")
    if not st.session_state.sensor_connected:
        st.caption("Trạng thái: Chưa kết nối")
    dinov3_repo = os.getenv("DINOV3_REPO", "dinov3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"Thiết bị suy luận: {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    start = st.button("▶  Bắt đầu dự đoán", type="primary", use_container_width=True, disabled=uploaded is None)
    stop = st.button("■  Dừng / xóa kết quả", use_container_width=True)
    if stop:
        st.session_state.prediction_started = False
        st.rerun()

with right:
    st.markdown('<div class="section-label">KẾT QUẢ DỰ ĐOÁN</div>', unsafe_allow_html=True)

    if start and uploaded is not None:
        st.session_state.prediction_started = True

        try:
            # Tai checkpoint tu Hugging Face truoc khi nap mo hinh
            checkpoint_path, dinov3_weights_path = download_model_files()

            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Khong tim thay checkpoint: {checkpoint_path}"
                )

            if not os.path.isdir(dinov3_repo):
                raise FileNotFoundError(
                    f"Khong tim thay thu muc ma DINOv3: {dinov3_repo}"
                )

            model, mean, std, class_names, device_local = load_model(
                checkpoint_path,
                dinov3_repo,
                str(device),
                dinov3_weights_path,
            )

            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False
            ) as tmp:
                tmp.write(uploaded.getvalue())
                temp_path = tmp.name

            try:
                X = recording_to_windows(temp_path)
            finally:
                os.unlink(temp_path)

            if len(X) == 0:
                st.error(
                    "Khong tim thay cua so du lieu hop le. "
                    "Hay kiem tra dinh dang CSV va cac cot cam bien."
                )
            else:
                ds = SensorDataset(X, None, mean, std)
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=8, shuffle=False
                )
                rows = []

                with torch.no_grad():
                    for x in loader:
                        x = x.to(device_local)
                        logits = model.classifier(
                            model.feature_extractor(x)
                        )
                        probs = torch.softmax(logits, dim=1)

                        for prob in probs:
                            top = torch.topk(
                                prob, k=min(3, len(class_names))
                            )
                            pred_idx = int(top.indices[0].item())

                            rows.append({
                                "Cua so": len(rows) + 1,
                                "Hoat dong du doan": class_names[pred_idx],
                                "Do tin cay (%)": round(
                                    float(top.values[0].item()) * 100, 2
                                ),
                            })

                st.session_state.results = rows

        except Exception as exc:
            st.error(f"Khong the du doan: {exc}")
            st.code(traceback.format_exc())

    if stop:
        st.session_state.pop("results", None)
    results = st.session_state.get("results", [])
    
    if results:
        first = results[0]

        # Lay ket qua theo dung ten cot da tao
        predicted_activity = first["Hoat dong du doan"]
        confidence = first["Do tin cay (%)"]

        st.markdown(
            f'''
            <div class="result-panel">
                <div class="result-inner">
                    <div class="small-label">
                        Hành vi dự đoán • Cửa sổ đầu tiên
                    </div>
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap">
                        <div class="prediction">
                            {predicted_activity.upper()}
                        </div>
                        <div>
                            <div class="small-label">Độ tin cậy</div>
                            <div class="confidence">{confidence:.1f}%</div>
                        </div>
                    </div>
                    <div style="background:#1d3048;border-radius:99px;height:12px;margin-top:1.2rem">
                        <div style="height:12px;width:{confidence}%;background:#28d5c0;border-radius:99px"></div>
                    </div>
                    <p class="muted" style="margin-top:.6rem">
                        Confidence score
                    </p>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        st.markdown("#### Các kết quả dự đoán")
        results_df = pd.DataFrame(results)
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Tải kết quả CSV",
            results_df.to_csv(index=False).encode("utf-8-sig"),
            "ket_qua_du_doan.csv",
            "text/csv",
        )
    else:
        st.markdown('''<div class="result-panel"><div class="small-label">Hành vi dự đoán</div><div class="result-inner" style="margin-top:1rem;text-align:center;padding:3.2rem 1rem"><div style="font-size:3rem">🧍</div><div class="prediction" style="font-size:1.4rem">Chưa có kết quả</div><p class="muted">Tải tệp CSV ở bên trái, sau đó nhấn “Bắt đầu dự đoán”.</p></div><p class="muted" style="margin-top:1.4rem">Kết quả chỉ xuất hiện sau khi mô hình đã nạp thành công và xử lý dữ liệu.</p></div>''', unsafe_allow_html=True)

st.markdown('<p style="text-align:right;color:#71869f;font-size:.75rem;margin-top:1rem">AI Model • CoTMix + DINOv3 • Dữ liệu được xử lý trên máy chủ triển khai</p>', unsafe_allow_html=True)
