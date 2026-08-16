"""
frontend/pdf_report.py — Analiz sonucu icin PDF raporu olusturma.

fpdf2 kullanir (klasik `fpdf` paketiyle ayni `from fpdf import FPDF` API'sine
sahiptir, ama `output()` her zaman bytearray dondurur ve `dest` parametresi
gerektirmez — eski app.py'deki `.output(dest='S').encode('latin-1')` cagrisi
kurulu pakete gore kirilabiliyordu, bu surum o riski ortadan kaldirir).
"""

from typing import List, Tuple

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


def create_pdf_report(
    date_str: str,
    filename: str,
    prediction: str,
    confidence: float,
    status: str,
    prob_list: List[Tuple[str, float]],
    uncertain: bool,
    top1_prob: float,
    top2_prob: float,
):
    if not FPDF_AVAILABLE:
        return None

    try:
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(9, 30, 58)
        pdf.cell(200, 10, txt="NeuroScan AI", ln=True, align='C')
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(200, 10, txt="MRI Analysis Report", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(50, 10, txt="Date:")
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=date_str, ln=True)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Image Filename:")
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=filename, ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(9, 30, 58)
        pdf.cell(200, 10, txt="AI Classification", ln=True)
        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Prediction:")
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=prediction.upper(), ln=True)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Confidence:")
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=f"%{confidence * 100:.2f}", ln=True)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Status:")
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=status, ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(9, 30, 58)
        pdf.cell(200, 10, txt="Class Probabilities", ln=True)
        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Arial", '', 12)
        for cls_name, prob in prob_list:
            pdf.cell(50, 8, txt=cls_name.title())
            pdf.cell(50, 8, txt=f"%{prob * 100:.2f}", ln=True)

        pdf.ln(10)

        if uncertain:
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(245, 158, 11)
            pdf.cell(200, 10, txt="Confidence Analysis", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 11)

            msg = (
                f"Modelin en yuksek tahmini {prediction.title()} sinifidir ancak "
                "tahmin guveni veya ilk iki sinif arasindaki fark yeterince yuksek "
                "degildir. Sonuc belirsiz olarak isaretlenmistir."
            )
            pdf.multi_cell(0, 8, txt=msg)
            pdf.ln(5)

        pdf.ln(10)
        pdf.set_font("Arial", 'I', 9)
        pdf.set_text_color(100, 100, 100)
        disclaimer = (
            "DISCLAIMER: Bu sistem yalnizca egitim ve arastirma amaciyla gelistirilmis "
            "bir yapay zeka prototipidir. Tibbi tani veya tedavi amaciyla kullanilamaz "
            "ve uzman hekim degerlendirmesinin yerine gecmez."
        )
        pdf.multi_cell(0, 6, txt=disclaimer)

        return bytes(pdf.output())

    except Exception as e:
        print(f"PDF Error: {e}")
        return None
