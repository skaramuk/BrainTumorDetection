"""
frontend/app.py — NeuroScan AI, ince Streamlit istemcisi.

Bu dosya artik hicbir sekilde torch/model.py/predict.py/gradcam.py import
etmez ve modeli yerel olarak yuklemez. Tum inference islemleri
api_client.py uzerinden FastAPI backend'ine HTTP istegi olarak gonderilir.

Calistirma (repo kokunden, backend ayrica calisiyor olmali):
    python -m streamlit run frontend/app.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import base64
import io

import pandas as pd
import streamlit as st
from PIL import Image

import config
import api_client
import pdf_report

# =========================================================
# SAYFA YAPILANDIRMASI
# =========================================================
st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STATE YÖNETİMİ
# =========================================================
if 'current_result' not in st.session_state:
    st.session_state['current_result'] = None

if 'uploaded_image_metadata' not in st.session_state:
    st.session_state['uploaded_image_metadata'] = None

if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0

if 'gradcam_result_id' not in st.session_state:
    st.session_state['gradcam_result_id'] = None

if 'gradcam_data' not in st.session_state:
    st.session_state['gradcam_data'] = None

if 'gradcam_error' not in st.session_state:
    st.session_state['gradcam_error'] = None

# =========================================================
# ÖZEL TASARIM (CSS)
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1300px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero-container {
        padding: 2.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        color: white;
        border: 1px solid #1e293b;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        color: white;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.6;
    }
    .dashboard-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }
    .status-card-high {
        background: rgba(34, 197, 94, 0.1);
        border-left: 5px solid #22c55e;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-card-warning {
        background: rgba(245, 158, 11, 0.1);
        border-left: 5px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #f8fafc;
    }
    .metric-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        height: 100%;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .metric-value-high { color: #4ade80; }
    .metric-value-warn { color: #fbbf24; }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================


@st.cache_data(show_spinner=False)
def get_dataset_overview():
    counts = {"Train": 0, "Validation": 0, "Test": 0}
    dirs = {"Train": config.TRAIN_DIR, "Validation": config.VAL_DIR, "Test": config.TEST_DIR}
    class_names = config.CLASS_NAMES

    details = {}
    for split, d in dirs.items():
        details[split] = {}
        if os.path.exists(d):
            total_split = 0
            for cls in class_names:
                cls_dir = os.path.join(d, cls)
                if os.path.exists(cls_dir):
                    count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    details[split][cls] = count
                    total_split += count
            counts[split] = total_split

    total = sum(counts.values())
    return {"total": total, "splits": counts, "details": details, "num_classes": len(class_names)}


# =========================================================
# BACKEND SAĞLIK KONTROLÜ
# =========================================================
try:
    health_info = api_client.health()
    backend_up = True
    backend_error_msg = None
except api_client.BackendError as e:
    health_info = {"status": "error", "model_loaded": False, "device": "-"}
    backend_up = False
    backend_error_msg = str(e)

# =========================================================
# SIDEBAR / NAVİGASYON
# =========================================================
with st.sidebar:
    st.markdown("""
        <h2 style='margin-bottom: 0px;'>NeuroScan AI</h2>
        <p style='color:#94a3b8; font-size: 0.85rem; margin-top: 0px; margin-bottom: 20px;'>Brain MRI Intelligence Platform</p>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Menü",
        ["MRI Analizi", "Analiz Geçmişi", "Model Performansı", "Sistem Hakkında"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")
    if backend_up and health_info["model_loaded"]:
        st.markdown("🟢 **AI Model:** Ready")
        st.markdown("🟢 **Inference:** Available")
    elif backend_up:
        st.markdown("🟡 **Backend:** Up, model not loaded")
        st.markdown("🔴 **Inference:** Unavailable")
    else:
        st.markdown("🔴 **Backend:** Unreachable")
        st.markdown("🔴 **Inference:** Unavailable")
    st.markdown(f"💻 **Device:** `{health_info['device']}`")


# =========================================================
# SAYFALAR
# =========================================================

# --- SAYFA 1: MRI ANALİZİ ---
if page == "MRI Analizi":

    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">NeuroScan AI</div>
            <div class="hero-subtitle">
                Derin öğrenme tabanlı beyin MRI görüntüsü sınıflandırma ve karar destek prototipi.
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not backend_up:
        st.error(f"Backend API'sine ulaşılamıyor ({backend_error_msg}). "
                 f"`python -m uvicorn backend.main:app --reload` çalıştığından emin olun.")
        st.stop()

    if not health_info["model_loaded"]:
        st.error("AI modeli backend tarafında yüklenemedi. Sistem yöneticisiyle iletişime geçin.")
        st.stop()

    mode = st.radio("Analiz Modu", ["Tek Görüntü", "Toplu Analiz"], horizontal=True)
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1.5rem; border-color: #334155;'>", unsafe_allow_html=True)

    if mode == "Tek Görüntü":
        left_col, right_col = st.columns([1, 1.2], gap="large")

        image = None
        image_bytes = None

        with left_col:
            st.markdown('<div class="section-header">MRI Görüntüsü Yükle</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Yüklenecek MRI dosyasını seçin",
                type=["jpg", "jpeg", "png"],
                key=f"uploader_{st.session_state['uploader_key']}",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                try:
                    image_bytes = uploaded_file.getvalue()
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    st.session_state['uploaded_image_metadata'] = {
                        'filename': uploaded_file.name,
                        'size': image.size,
                        'format': image.format or uploaded_file.name.split('.')[-1].upper(),
                        'mode': image.mode
                    }

                    st.image(image, use_container_width=True)

                    meta = st.session_state['uploaded_image_metadata']
                    st.markdown(f"""
                    <div style="background-color: #1e293b; padding: 1rem; border-radius: 8px; margin-top: 1rem; font-size: 0.9rem; border: 1px solid #334155;">
                        <b>Dosya Adı:</b> {meta['filename']}<br>
                        <b>Çözünürlük:</b> {meta['size'][0]}x{meta['size'][1]}<br>
                        <b>Format:</b> {meta['format']}<br>
                        <b>Renk Modu:</b> {meta['mode']}<br>
                        <span style="color: #4ade80; font-weight: bold; display: inline-block; margin-top: 0.5rem;">✓ Analize Hazır</span>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Görüntü okuma hatası: {e}")
                    image = None
                    image_bytes = None

        with right_col:
            st.markdown('<div class="section-header">AI Analizi</div>', unsafe_allow_html=True)

            if uploaded_file is None:
                st.markdown("""
                <div class="dashboard-card" style="text-align: center; padding: 3rem 1rem;">
                    <h3 style="color: #94a3b8; margin-bottom: 0.5rem;">Analiz bekleniyor</h3>
                    <p style="color: #64748b; font-size: 0.95rem;">
                        Lütfen sol taraftan bir MRI görüntüsü yükleyin.<br>
                        Yükleme tamamlandıktan sonra analiz butonu aktifleşecektir.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif image is None:
                st.warning("Geçerli bir görüntü yüklenemediği için analiz yapılamıyor. "
                           "Lütfen geçerli bir JPG/PNG dosyası yükleyin.")
            else:
                col_btn1, col_btn2 = st.columns([3, 1])
                with col_btn1:
                    if st.button("🧠 MRI Görüntüsünü Analiz Et", type="primary", use_container_width=True, disabled=st.session_state['current_result'] is not None):
                        with st.spinner("Inference işlemi devam ediyor..."):
                            try:
                                api_result = api_client.predict(image_bytes, uploaded_file.name)
                            except api_client.BackendError as e:
                                st.error(f"Tahmin sırasında hata oluştu: {e}")
                                api_result = None

                        if api_result is not None:
                            sorted_probs = [
                                (p["class_name"], p["probability"]) for p in api_result["probabilities"]
                            ]
                            uncertain = api_result["status"] == "Review Required"

                            current_res = {
                                "id": api_result["id"],
                                "predicted_class": api_result["predicted_class"],
                                "confidence": api_result["confidence"],
                                "probabilities": sorted_probs,
                                "top1_prob": api_result["confidence"],
                                "top2_prob": api_result["top2_probability"],
                                "top2_class": api_result["top2_class"],
                                "margin": api_result["margin"],
                                "uncertain": uncertain,
                                "status": api_result["status"],
                                "date": api_result["created_at"],
                                "filename": uploaded_file.name,
                                "image_bytes": image_bytes,
                            }

                            st.session_state['current_result'] = current_res
                            st.rerun()

                with col_btn2:
                    if st.session_state['current_result'] is not None:
                        if st.button("Yeni Analiz", use_container_width=True):
                            st.session_state['current_result'] = None
                            st.session_state['uploader_key'] += 1
                            st.rerun()

                res = st.session_state['current_result']

                if res is not None:
                    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

                    if res["uncertain"]:
                        st.markdown(f"""
                        <div class="status-card-warning">
                            <div class="status-title">Prediction: {res['predicted_class'].title()}</div>
                            <div style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #f8fafc;">Confidence: %{res['confidence']*100:.2f}</div>
                            <div style="font-size: 1rem; font-weight: bold; color: #fbbf24;">Status: Review Required</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-card-high">
                            <div class="status-title">Prediction: {res['predicted_class'].title()}</div>
                            <div style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #f8fafc;">Confidence: %{res['confidence']*100:.2f}</div>
                            <div style="font-size: 1rem; font-weight: bold; color: #4ade80;">Status: High Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">Prediction</div>
                            <div class="metric-value">{res['predicted_class'].title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        val_color = "metric-value-warn" if res["uncertain"] else "metric-value-high"
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value {val_color}">%{res['confidence']*100:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        val_color = "metric-value-warn" if res["uncertain"] else "metric-value-high"
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">Status</div>
                            <div class="metric-value {val_color}" style="font-size: 1.1rem;">{res['status']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)

                    for cls_name, prob in res["probabilities"]:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.markdown(f"<span style='font-weight: 500; font-size: 0.95rem; color: #e2e8f0;'>{cls_name.title()}</span>", unsafe_allow_html=True)
                        with col2:
                            st.progress(float(prob), text=f"%{float(prob)*100:.2f}")

                    if res["uncertain"]:
                        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                        with st.expander("Tahmin neden belirsiz?"):
                            st.write(f"- **En yüksek sınıf:** {res['predicted_class'].title()} (%{res['top1_prob']*100:.2f})")
                            st.write(f"- **İkinci sınıf:** {res['top2_class'].title()} (%{res['top2_prob']*100:.2f})")
                            st.write(f"- **İlk iki sınıf arasındaki fark:** %{res['margin']*100:.2f}")
                            st.write(f"- **Confidence threshold:** %{config.CONFIDENCE_THRESHOLD*100:.0f}")
                            st.write(f"- **Margin threshold:** %{config.MARGIN_THRESHOLD*100:.0f}")
                            st.write("---")
                            if res["confidence"] < config.CONFIDENCE_THRESHOLD:
                                st.write("🔹 *Tetiklenen kural:* Güven eşiği aşılamadı (Confidence < Threshold).")
                            if res["margin"] < config.MARGIN_THRESHOLD:
                                st.write("🔹 *Tetiklenen kural:* Sınıflar arası fark eşiği aşılamadı (Margin < Threshold).")

                    # --- Grad-CAM Section ---
                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">Model Attention (Grad-CAM)</div>', unsafe_allow_html=True)

                    if st.session_state['gradcam_result_id'] != res['id']:
                        try:
                            with st.spinner("Isı haritası oluşturuluyor..."):
                                gc = api_client.gradcam(res['image_bytes'], res['filename'], res['predicted_class'])
                            st.session_state['gradcam_data'] = gc
                            st.session_state['gradcam_error'] = None
                        except api_client.BackendError as e:
                            st.session_state['gradcam_data'] = None
                            st.session_state['gradcam_error'] = str(e)
                        st.session_state['gradcam_result_id'] = res['id']

                    if st.session_state['gradcam_error']:
                        st.error(f"Grad-CAM oluşturulurken hata oluştu: {st.session_state['gradcam_error']}")
                    elif st.session_state['gradcam_data']:
                        gc = st.session_state['gradcam_data']
                        heatmap_img = Image.open(io.BytesIO(base64.b64decode(gc["heatmap_png_base64"])))
                        overlay_img = Image.open(io.BytesIO(base64.b64decode(gc["overlay_png_base64"])))
                        original_resized = Image.open(io.BytesIO(res['image_bytes'])).convert("RGB").resize(
                            (config.IMAGE_SIZE, config.IMAGE_SIZE)
                        )

                        g_col1, g_col2, g_col3 = st.columns(3)
                        with g_col1:
                            st.image(original_resized, caption="Original MRI", use_container_width=True)
                        with g_col2:
                            st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                        with g_col3:
                            st.image(overlay_img, caption="Overlay", use_container_width=True)

                        st.info("💡 Grad-CAM görselleştirmesi modelin sınıflandırma kararında etkili olan bölgeleri "
                                "yaklaşık olarak gösterir. Tümör segmentasyonu veya klinik lokalizasyon sonucu değildir.")

                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

                    if pdf_report.FPDF_AVAILABLE:
                        pdf_bytes = pdf_report.create_pdf_report(
                            date_str=res["date"],
                            filename=res["filename"],
                            prediction=res["predicted_class"],
                            confidence=res["confidence"],
                            status=res["status"],
                            prob_list=res["probabilities"],
                            uncertain=res["uncertain"],
                            top1_prob=res["top1_prob"],
                            top2_prob=res["top2_prob"]
                        )
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Analiz Raporu Oluştur (PDF)",
                                data=pdf_bytes,
                                file_name=f"NeuroScan_Report_{res['filename'].split('.')[0]}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    else:
                        st.info("💡 PDF Raporu oluşturmak için sisteminizde `fpdf2` kütüphanesi eksik.")

    elif mode == "Toplu Analiz":
        st.markdown('<div class="section-header">Toplu MRI Görüntüsü Yükle</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Çoklu dosya seçebilirsiniz (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("🚀 Toplu Analizi Başlat", type="primary"):
                files_payload = [(f.name, f.getvalue()) for f in uploaded_files]

                with st.spinner("Görüntüler analiz ediliyor..."):
                    try:
                        batch_response = api_client.predict_batch(files_payload)
                    except api_client.BackendError as e:
                        st.error(f"Toplu analiz sırasında hata oluştu: {e}")
                        batch_response = None

                if batch_response is not None:
                    results_list = []
                    for item in batch_response["results"]:
                        if item["success"]:
                            r = item["result"]
                            results_list.append({
                                "File": item["filename"],
                                "Prediction": r["predicted_class"].title(),
                                "Confidence": f"%{r['confidence']*100:.2f}",
                                "Status": r["status"],
                                "Top-2 Class": r["top2_class"].title(),
                                "Margin": f"%{r['margin']*100:.2f}",
                            })
                        else:
                            results_list.append({
                                "File": item["filename"],
                                "Prediction": "ERROR",
                                "Confidence": "N/A",
                                "Status": item["error"],
                                "Top-2 Class": "N/A",
                                "Margin": "N/A",
                            })

                    st.success(f"{len(uploaded_files)} görüntü analiz edildi.")
                    df_results = pd.DataFrame(results_list)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)

                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Sonuçları CSV Olarak İndir",
                        data=csv,
                        file_name="batch_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.8rem; border-top: 1px solid #334155; padding-top: 1rem;">
            Bu sistem yalnızca eğitim ve araştırma amacıyla geliştirilmiş bir yapay zeka prototipidir.
            Tıbbi tanı veya tedavi amacıyla kullanılamaz ve uzman hekim değerlendirmesinin yerine geçmez.
        </div>
    """, unsafe_allow_html=True)

# --- SAYFA 2: ANALİZ GEÇMİŞİ ---
elif page == "Analiz Geçmişi":
    st.markdown('<div class="section-header" style="font-size: 1.8rem;">Analiz Geçmişi</div>', unsafe_allow_html=True)

    if not backend_up:
        st.error(f"Backend API'sine ulaşılamıyor ({backend_error_msg}). "
                 f"`python -m uvicorn backend.main:app --reload` çalıştığından emin olun.")
        st.stop()

    try:
        history_response = api_client.get_history()
    except api_client.BackendError as e:
        st.error(f"Geçmiş yüklenemedi: {e}")
        st.stop()

    hist_items = history_response["items"]

    if len(hist_items) == 0:
        st.info("Henüz yapılmış herhangi bir analiz bulunmuyor.")
    else:
        rows = []
        for item in hist_items:
            rows.append({
                "Date": item["created_at"],
                "File": item["filename"],
                "Prediction": item["predicted_class"].title(),
                "Confidence": f"%{item['confidence']*100:.2f}",
                "Status": item["status"],
                "Top-2 Class": item["top2_class"].title(),
                "Margin": f"%{item['margin']*100:.2f}",
            })
        df = pd.DataFrame(rows)

        # Metric Cards
        total_analyses = len(df)
        high_conf = len(df[df["Status"] == "High Confidence"])
        review_req = len(df[df["Status"] == "Review Required"])

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Analyses</div>
                <div class="metric-value">{total_analyses}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">High Confidence</div>
                <div class="metric-value" style="color: #4ade80;">{high_conf}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Review Required</div>
                <div class="metric-value" style="color: #fbbf24;">{review_req}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            pred_filter = st.selectbox("Filtre: Prediction", ["All"] + [c.title() for c in config.CLASS_NAMES])
        with col_f2:
            status_filter = st.selectbox("Filtre: Status", ["All", "High Confidence", "Review Required"])

        # Apply Filters
        filtered_df = df.copy()
        if pred_filter != "All":
            filtered_df = filtered_df[filtered_df["Prediction"] == pred_filter]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["Status"] == status_filter]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Geçmişi Temizle"):
            try:
                api_client.clear_history()
            except api_client.BackendError as e:
                st.error(f"Geçmiş temizlenemedi: {e}")
            else:
                st.rerun()

# --- SAYFA 3: MODEL PERFORMANSI ---
elif page == "Model Performansı":
    st.markdown('<div class="section-header" style="font-size: 1.8rem;">Model Performansı</div>', unsafe_allow_html=True)

    summary_path = os.path.join(config.REPORTS_DIR, "evaluation_summary.csv")
    per_class_path = os.path.join(config.REPORTS_DIR, "evaluation_per_class.csv")

    # 1. Evaluation Results (Gerçek Metrikler)
    if os.path.exists(summary_path) and os.path.exists(per_class_path):
        try:
            summary_df = pd.read_csv(summary_path)
            class_df = pd.read_csv(per_class_path)

            cols = st.columns(4)
            for idx, metric_name in enumerate(
                ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
            ):
                row = summary_df.loc[summary_df["Metrik"] == metric_name, "Değer"]
                if not row.empty:
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="metric-container" style="margin-bottom: 1rem;">
                            <div class="metric-label">{metric_name}</div>
                            <div class="metric-value" style="color: #60a5fa;">{float(row.iloc[0]):.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("### Class Performance")
            display_df = class_df.copy()
            for c in ["Precision", "Recall", "F1-Score"]:
                display_df[c] = display_df[c].astype(float).map("{:.4f}".format)
            st.table(display_df)

        except Exception as e:
            st.error(f"Metrikler okunurken hata oluştu: {e}")
    else:
        st.info("Değerlendirme sonuçları (evaluation_summary.csv / evaluation_per_class.csv) bulunamadı.")

    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)

    # 2. Confusion Matrix & Training History
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        cm_path = os.path.join(config.FIGURES_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.markdown("### Confusion Matrix")
            st.image(Image.open(cm_path), use_container_width=True)
    with col_fig2:
        th_path = os.path.join(config.FIGURES_DIR, "training_history.png")
        if os.path.exists(th_path):
            st.markdown("### Training History")
            st.image(Image.open(th_path), use_container_width=True)

    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)

    # 3. Dataset Overview
    st.markdown("### Dataset Overview")
    ds_stats = get_dataset_overview()
    st.write(f"**Total Classes:** {ds_stats['num_classes']} | **Total Images:** {ds_stats['total']}")

    ds_col1, ds_col2, ds_col3 = st.columns(3)
    ds_col1.metric("Train Images", ds_stats['splits'].get("Train", 0))
    ds_col2.metric("Validation Images", ds_stats['splits'].get("Validation", 0))
    ds_col3.metric("Test Images", ds_stats['splits'].get("Test", 0))

    cd_path = os.path.join(config.FIGURES_DIR, "class_distribution.png")
    if os.path.exists(cd_path):
        st.image(Image.open(cd_path), use_container_width=True, caption="Class Distribution")

    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)

    # 4. External Validation
    st.markdown("### External Validation")
    ext_path = os.path.join(config.PROJECT_ROOT, "external_test_results.csv")
    if os.path.exists(ext_path):
        try:
            ext_df = pd.read_csv(ext_path)
            total_ext = len(ext_df)

            if "correct" not in ext_df.columns:
                ext_df["correct"] = (ext_df["actual_class"].str.lower() == ext_df["prediction"].str.lower())

            correct_preds = ext_df["correct"].sum()
            incorrect_preds = total_ext - correct_preds
            ext_acc = correct_preds / total_ext if total_ext > 0 else 0

            review_req_count = len(ext_df[ext_df["confidence"] < config.CONFIDENCE_THRESHOLD])
            if "margin" in ext_df.columns:
                review_req_count = len(ext_df[
                    (ext_df["confidence"] < config.CONFIDENCE_THRESHOLD) |
                    (ext_df["margin"] < config.MARGIN_THRESHOLD)
                ])

            review_rate = review_req_count / total_ext if total_ext > 0 else 0

            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            ecol1.metric("Total Images", total_ext)
            ecol2.metric("Accuracy", f"%{ext_acc*100:.2f}")
            ecol3.metric("Correct / Incorrect", f"{correct_preds} / {incorrect_preds}")
            ecol4.metric("Review Required Rate", f"%{review_rate*100:.2f}")

            st.markdown("### Error Analysis")
            errors_df = ext_df[ext_df["correct"] == False].copy()
            if len(errors_df) > 0:
                cols_to_show = ["filename", "actual_class", "prediction", "confidence", "top2_class", "margin"]
                show_df = errors_df[[c for c in cols_to_show if c in errors_df.columns]]
                st.dataframe(show_df, use_container_width=True, hide_index=True)
            else:
                st.success("Tüm harici doğrulama verileri doğru tahmin edildi!")

        except Exception as e:
            st.error(f"External validation dosyası okunamadı: {e}")
    else:
        st.info("External validation sonuçları henüz sisteme aktarılmadı.")
        st.markdown("### Error Analysis")
        st.info("Error analysis için harici test verisi bulunmuyor.")

# --- SAYFA 4: SİSTEM HAKKINDA ---
elif page == "Sistem Hakkında":
    st.markdown('<div class="section-header" style="font-size: 1.8rem;">NeuroScan AI</div>', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #94a3b8; margin-top: -15px;'>Brain MRI Intelligence Platform</h4>", unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size: 1.1rem; line-height: 1.6; margin-top: 1.5rem; margin-bottom: 2rem;">
        NeuroScan AI, beyin MRI görüntülerini Glioma, Meningioma, Pituitary ve No Tumor sınıflarında sınıflandırmak üzere geliştirilmiş PyTorch tabanlı bir yapay zeka karar destek prototipidir.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### Problem")
        st.write("Beyin tümörlerinin tespiti ve sınıflandırılması radyologlar için zaman alan, uzmanlık gerektiren bir süreçtir. Yapay zeka tabanlı bir karar destek mekanizması, ikinci bir görüş sağlayarak teşhis sürecini hızlandırmayı hedefler.")

        st.markdown("### AI Model")
        st.write("Sistemimiz derin öğrenme tabanlı evrişimli sinir ağları (CNN) kullanmaktadır. Model görüntü özelliklerini çıkararak belirtilen dört sınıftan birine atanmasını sağlamaktadır.")

        st.markdown("### Classes")
        st.markdown("""
        - **Glioma:** Beyin ve omurilikte destekleyici dokulardan köken alan tümör.
        - **Meningioma:** Beyni saran zarlarda oluşan tümör.
        - **Pituitary:** Hipofiz bezinde oluşan tümör.
        - **No Tumor:** Tümör bulgusu olmayan normal beyin yapısı.
        """)

    with col2:
        st.markdown("### Inference Pipeline")
        st.write("Girdi olarak verilen MRI görüntüleri gerekli önişlemelerden (yeniden boyutlandırma, normalizasyon) geçirildikten sonra FastAPI backend'i üzerinden modelden geçirilir. Tahmin sırasında gradient hesaplamaları kapatılarak (`inference_mode`) hızlı ve verimli sonuç alınır.")

        st.markdown("### Confidence & Uncertainty Mechanism")
        st.write("Model yalnızca en yüksek olasılığa sahip sınıfı seçmekle kalmaz; güven skoru hesaplayarak sonucun güvenilirliğini sorgular. Belirlenen eşik (ör. %80) aşılamadığında veya en yüksek iki olasılık birbirine çok yakın olduğunda sistem durumu 'Review Required' olarak işaretler.")

        st.markdown("### Technology Stack")
        st.markdown("""
        - **Python**
        - **PyTorch & Torchvision**
        - **FastAPI** (backend / inference API)
        - **Streamlit** (frontend)
        - **SQLite** (analiz geçmişi)
        - **Pillow**
        """)

    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)

    with st.expander("🔍 Model Details"):
        st.markdown(f"""
        - **Framework:** PyTorch
        - **Architecture:** ResNet18 (Transfer Learning)
        - **Input Size:** {config.IMAGE_SIZE}x{config.IMAGE_SIZE}
        - **Number of Classes:** {config.NUM_CLASSES}
        - **Inference Device:** {health_info['device']}
        - **Model Status:** {'Ready' if health_info['model_loaded'] else 'Not Loaded'}
        - **API Base URL:** {api_client.BASE_URL}
        """)

    st.markdown("### Medical Disclaimer")
    st.warning("Bu sistem yalnızca eğitim ve araştırma amacıyla geliştirilmiş bir yapay zeka prototipidir. Tıbbi tanı veya tedavi amacıyla kullanılamaz ve uzman hekim değerlendirmesinin yerine geçmez. 'Decision-support prototype' olarak tasarlanmıştır.")
