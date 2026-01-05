###############################################################
# final.py — FULL FIXED VERSION
###############################################################

import io
import os
import re
import json
import base64
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from db import insert_pdf_invoice, fetch_all_pdfs
import streamlit as st
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
import shutil
from PIL import Image
import numpy as np
import cv2
import pandas as pd

import boto3
S3_BUCKET = "pdf-plumber-storage"
s3 = boto3.client("s3")
###############################################################
# Import database functions safely
try:
    from db import insert_pdf_invoice, fetch_all_pdfs
except Exception as e:
    raise Exception("❌ db.py not found or has errors. Error:", e)


###############################################################
# Remove unsafe mysql import (causes ModuleNotFound crash)
###############################################################
try:
    import mysql.connector  # not required here but safe
except:
    mysql = None


###############################################################
# Configuration
###############################################################
class Config:
    POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    SUPPORTED_LANGUAGES = ["eng", "deu", "fra", "spa", "ita"]
    OCR_THREAD_POOL_SIZE = 4
    DEFAULT_OCR_CONF_THRESHOLD = 60.0


###############################################################
# Logging
###############################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-invoice")


###############################################################
# Tesseract detection
###############################################################
def _find_tesseract_executable() -> Optional[str]:
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    possible_roots = [
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        r"C:\Program Files",
        r"C:\Program Files (x86)",
    ]
    for root in possible_roots:
        if not root:
            continue
        candidate = os.path.join(root, "Tesseract-OCR", "tesseract.exe")
        if os.path.exists(candidate):
            return candidate

    if Config.TESSERACT_CMD and os.path.exists(Config.TESSERACT_CMD):
        return Config.TESSERACT_CMD

    return None


TESSERACT_PATH = _find_tesseract_executable()
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    TESSERACT_AVAILABLE = True
    logger.info(f"Tesseract found: {TESSERACT_PATH}")
else:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not found. OCR disabled.")


###############################################################
# Dataclasses
###############################################################
@dataclass
class ExtractionStats:
    digital_success: bool = False
    ocr_used: bool = False
    pages_processed: int = 0
    confidence_score: float = 0.0


###############################################################
# Helpers
###############################################################
def is_pdf_bytes(file_bytes: bytes) -> bool:
    return file_bytes[:4] == b"%PDF"


def normalize_amount(value: str) -> str:
    if not value or not isinstance(value, str):
        return value
    value = value.strip()
    leading_currency = ""
    m = re.match(r"^([€$£¥])\s*(.+)$", value)
    if m:
        leading_currency = m.group(1) + " "
        value = m.group(2)

    value = value.replace("\u202f", " ").replace(" ", "")

    if value.count(",") and value.count("."):
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif value.count(",") and not value.count("."):
        parts = value.split(",")
        if len(parts[-1]) in (1, 2):
            value = ".".join(["".join(parts[:-1]), parts[-1]])
        else:
            value = "".join(parts)

    value = re.sub(r"[^\d\.\-]", "", value)
    return leading_currency + value


def safe_read_confidences(conf_list) -> List[float]:
    vals = []
    for c in conf_list:
        try:
            f = float(c)
            if f >= 0:
                vals.append(f)
        except:
            pass
    return vals


###############################################################
# Preprocessing
###############################################################
def preprocess_image_cv(img_cv: np.ndarray, method: str = "standard") -> np.ndarray:
    if img_cv is None:
        raise ValueError("img_cv is None")
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv.copy()

    if method == "standard":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 3
        )
    elif method == "morphology":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        thresh = gray

    thresh = cv2.medianBlur(thresh, 3)
    return thresh


###############################################################
# Cache PDF Images
###############################################################
@st.cache_resource
def get_pdf_images_cached(file_bytes: bytes) -> List[Image.Image]:
    try:
        return convert_from_bytes(file_bytes, poppler_path=Config.POPPLER_PATH)
    except:
        try:
            return convert_from_bytes(file_bytes)
        except:
            logger.exception("PDF → image conversion failed")
            return []


###############################################################
# Digital text extraction
###############################################################
def extract_digital_text_bytes(file_bytes: bytes) -> Tuple[str, bool, int]:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for i, page in enumerate(doc):
            t = page.get_text("text")
            if t.strip():
                texts.append(f"\n--- Page {i+1} ---\n{t}")
        total_pages = len(doc)
        doc.close()
        text = "\n".join(texts).strip()
        return text, bool(text), total_pages
    except:
        logger.exception("Digital extraction failed")
        return "", False, 0


###############################################################
# OCR worker
###############################################################
def ocr_page_worker(img: Image.Image, lang: str, methods: List[str]) -> Tuple[str, float]:
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    best_text = ""
    best_conf = 0.0
    for m in methods:
        try:
            pre = preprocess_image_cv(img_cv, m)
            data = pytesseract.image_to_data(pre, lang=lang, output_type=pytesseract.Output.DICT)
            confs = safe_read_confidences(data.get("conf", []))
            avg_conf = (sum(confs) / len(confs)) if confs else 0.0
            text = pytesseract.image_to_string(pre, lang=lang)
            if avg_conf > best_conf and text.strip():
                best_conf = avg_conf
                best_text = text
        except:
            continue
    return best_text, best_conf


###############################################################
# OCR extraction
###############################################################
def extract_ocr_text_bytes(file_bytes: bytes, lang: str, deep_mode: bool):
    images = get_pdf_images_cached(file_bytes)
    if not images:
        return "", 0.0, 0

    methods = ["standard"] if not deep_mode else ["standard", "adaptive", "morphology"]

    texts = []
    confs = []

    with ThreadPoolExecutor(max_workers=Config.OCR_THREAD_POOL_SIZE) as pool:
        futures = [pool.submit(ocr_page_worker, img, lang, methods) for img in images]
        for i, f in enumerate(futures):
            try:
                t, c = f.result()
                if t.strip():
                    texts.append(f"\n--- Page {i+1} (OCR {c:.1f}) ---\n{t}")
                confs.append(c)
            except:
                confs.append(0)

    avg_conf = (sum(confs) / len(confs)) if confs else 0.0

    return "\n".join(texts).strip(), avg_conf, len(images)


###############################################################
# Cached extraction wrapper
###############################################################
@st.cache_data
def extract_text_cached(file_bytes, lang, deep_mode):
    stats = ExtractionStats()

    if len(file_bytes) > Config.MAX_FILE_SIZE:
        return "", {"error": "File too large"}

    if not is_pdf_bytes(file_bytes):
        return "", {"error": "Invalid PDF"}

    digital_text, digital_success, page_count = extract_digital_text_bytes(file_bytes)
    stats.digital_success = digital_success
    stats.pages_processed = page_count

    result_text = digital_text

    if (not digital_success) or len(digital_text.strip()) < 50:
        ocr_text, ocr_conf, ocr_pages = extract_ocr_text_bytes(file_bytes, lang, deep_mode)
        stats.ocr_used = True
        stats.confidence_score = ocr_conf
        stats.pages_processed = ocr_pages
        if ocr_text.strip():
            result_text = ocr_text

    return result_text, asdict(stats)


###############################################################
# Invoice Info Extraction
###############################################################
class InvoiceInfoExtractor:
    def __init__(self):
        self.patterns = {
            'invoice_number': [
                r"(?:Invoice|Invoice\s*No|Invoice\s*#|INV)[\s:\#\-]*([A-Z0-9\-/\.]{3,})",
                r"Invoice\s*Number[\s:\#\-]*([A-Z0-9\-/\.]{3,})",
                r"\bRef[:\s]*([A-Z0-9\-/\.]{3,})\b",
            ],
            'customer_name': [
                r"(?:Bill\s*To|Billed\s*To|Sold\s*To)[:\s]*([A-Za-z0-9\.,\-\s]{3,})",
                r"Customer[:\s]*([A-Za-z0-9\.,\-\s]{3,})",
            ],
            'customer_person': [
                r"(?:Contact|Attn\.?|Attention)[:\s]*([A-Za-z\s]{3,})",
                r"(?:Mr|Mrs|Ms|Dr)\.?\s+([A-Za-z\s]{3,})",
            ],
            'total_amount': [
                r"Total\s*(?:Due|Amount|Payable)?[:\s]*([€$£¥]?\s*[0-9\.,\s]+)",
                r"Amount\s*Due[:\s]*([€$£¥]?\s*[0-9\.,\s]+)",
                r"Grand\s*Total[:\s]*([€$£¥]?\s*[0-9\.,\s]+)",
            ],
            'invoice_date': [
                r"Invoice\s*Date[:\s]*([0-3]?\d[\/\-\.][01]?\d[\/\-\.]\d{2,4})",
                r"Date[:\s]*([0-3]?\d[\/\-\.][01]?\d[\/\-\.]\d{2,4})",
            ],
            'due_date': [
                r"(?:Due\s*Date|Payment\s*Due)[:\s]*([0-3]?\d[\/\-\.][01]?\d[\/\-\.]\d{2,4})"
            ],
            'vendor_name': [
                r"^(?P<vendor>[A-Z][A-Za-z0-9&\.\s,-]{2,}(?:GmbH|AG|Ltd|Inc|Corp|LLC|S\.A\.))",
                r"^([A-Z][A-Za-z0-9\.\s,-]{3,})",
            ],
            'vendor_address': [
                r"(\d{1,4}\s+[A-Za-z0-9\.,\-\s]+(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Strasse)?)",
                r"PO Box\s*\d+\b",
            ],
            'vat_number': [
                r"\bVAT(?:\s*No\.?)?[:\s]*([A-Z0-9\-\s]{6,})\b",
                r"\bVAT\s*ID[:\s]*([A-Z0-9\-\s]{6,})\b",
            ],
            'period': [
                r"(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}\s*-\s*\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})"
            ],
        }

    def clean_text(self, t: str) -> str:
        t = t.replace("\r", "\n")
        t = re.sub(r"\n{2,}", "\n\n", t)
        return t.strip()

    def extract_with_confidence(self, text: str, field: str) -> Tuple[str, float]:
        ct = self.clean_text(text)
        patterns = self.patterns.get(field, [])
        for i, p in enumerate(patterns):
            try:
                m = re.findall(p, ct, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    val = m[0]
                    if isinstance(val, tuple):
                        val = val[0]
                    val = val.strip()
                    if field == "total_amount":
                        val = normalize_amount(val)
                    conf = max(0.0, 1.0 - i * 0.15)
                    if val:
                        return val, conf
            except:
                continue
        return "Not found", 0.0

    def extract_line_items(self, text: str):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        items = []
        patterns = [
            r"^(.+?)\s+([0-9\.,\s]+)\s+([0-9]+)\s+([0-9\.,\s]+)$",
            r"^(.+?)\s+([€$£¥]?\s*[0-9\.,\s]+)\s+x\s*([0-9]+)$",
            r"^(.+?)\s+([€$£¥]?\s*[0-9\.,\s]+)$",
        ]
        for line in lines:
            for p in patterns:
                m = re.search(p, line)
                if m:
                    g = m.groups()
                    item = {
                        "description": g[0].strip(),
                        "amount": normalize_amount(g[1].strip()) if len(g) > 1 else "",
                        "quantity": g[2].strip() if len(g) > 2 else "1",
                    }
                    if len(item["description"]) > 2:
                        items.append(item)
                    break
        return items

    def extract_invoice_info(self, text: str):
        if not text or len(text.strip()) < 10:
            return {"error": "Text too short"}

        info = {}
        confs = {}

        fields = [
            "invoice_number", "customer_name", "customer_person", "total_amount",
            "invoice_date", "due_date", "vendor_name", "vendor_address",
            "vat_number", "period"
        ]

        for f in fields:
            val, c = self.extract_with_confidence(text, f)
            info[f] = val
            confs[f] = c

        # combine customer
        if info.get("customer_name") != "Not found" and info.get("customer_person") != "Not found":
            full_customer = f"{info['customer_name']} - {info['customer_person']}"
        elif info.get("customer_name") != "Not found":
            full_customer = info['customer_name']
        elif info.get("customer_person") != "Not found":
            full_customer = info['customer_person']
        else:
            full_customer = "Not found"
        info["full_customer"] = full_customer

        info["line_items"] = self.extract_line_items(text)
        info["extraction_confidence"] = confs
        info["overall_confidence"] = sum(confs.values()) / len(confs) * 100
        info["extraction_timestamp"] = datetime.now().isoformat()
        info["text_length"] = len(text)
        info["line_count"] = len(text.splitlines())

        return info


###############################################################
# Q/A System
###############################################################
class SmartAnswerSystem:
    def __init__(self):
        self.keyword_mappings = {
            'amount': ['total_amount', 'amount', 'total', 'price', 'sum'],
            'invoice': ['invoice_number', 'invoice', 'inv', 'number'],
            'customer': ['customer', 'client', 'buyer', 'full_customer'],
            'date': ['invoice_date', 'due_date', 'period', 'date'],
            'vendor': ['vendor', 'supplier', 'vendor_name'],
            'vat': ['vat_number', 'vat', 'tax'],
            'items': ['line_items', 'items', 'products', 'services'],
        }

    def find_in_structured_data(self, q: str, info: Dict[str, Any]):
        q = q.lower()
        for cat, keys in self.keyword_mappings.items():
            if any(k in q for k in keys):
                if cat == "amount":
                    return f"💰 Total Amount: {info.get('total_amount')}"
                if cat == "invoice":
                    return f"📑 Invoice Number: {info.get('invoice_number')}"
                if cat == "customer":
                    return f"👤 Customer: {info.get('full_customer')}"
                if cat == "date":
                    msg = []
                    if info.get("invoice_date") != "Not found":
                        msg.append(f"📅 Invoice Date: {info.get('invoice_date')}")
                    if info.get("due_date") != "Not found":
                        msg.append(f"⏰ Due Date: {info.get('due_date')}")
                    return "\n".join(msg) if msg else None
                if cat == "vendor":
                    return f"🏢 Vendor: {info.get('vendor_name')}"
                if cat == "vat":
                    return f"🔢 VAT: {info.get('vat_number')}"
                if cat == "items":
                    items = info.get("line_items", [])[:5]
                    out = "📋 Line Items:\n"
                    for i, it in enumerate(items, 1):
                        out += f"{i}. {it['description']} — {it['amount']}\n"
                    return out
        return None

    def search_text(self, query, text):
        q = query.lower()
        lines = [l for l in text.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            if q in line.lower():
                ctx = []
                if i > 0:
                    ctx.append(lines[i-1])
                ctx.append(f">>> {line}")
                if i + 1 < len(lines):
                    ctx.append(lines[i+1])
                return "\n".join(ctx)
        return None

    def smart_answer(self, query, text, info):
        if not query.strip():
            return "❓ Please ask a question."

        q = query.lower()

        if any(w in q for w in ["summary", "all details", "everything"]):
            return self.generate_summary(info)

        structured = self.find_in_structured_data(q, info)
        if structured:
            return structured

        text_result = self.search_text(q, text)
        if text_result:
            return text_result

        return f"⚠️ No direct answer found for '{query}'."

    def generate_summary(self, info):
        parts = [
            "📄 **INVOICE SUMMARY**",
            f"Invoice Number: {info.get('invoice_number')}",
            f"Date: {info.get('invoice_date')}",
            f"Total: {info.get('total_amount')}",
            f"Vendor: {info.get('vendor_name')}",
            f"Customer: {info.get('full_customer')}",
            f"Confidence: {info.get('overall_confidence'):.1f}%",
        ]
        if info.get("line_items"):
            parts.append("Line Items:")
            for it in info["line_items"][:3]:
                parts.append(f"- {it['description']} — {it['amount']}")
        return "\n".join(parts)


###############################################################
# Export Helpers
###############################################################
def export_to_json(info):
    return json.dumps(info, indent=2, ensure_ascii=False)


def export_to_csv(info):
    flat = {}
    for k, v in info.items():
        if k == "line_items":
            flat["line_items_count"] = len(v)
        elif k == "extraction_confidence":
            for kk, cc in v.items():
                flat[f"{kk}_conf"] = cc
        else:
            flat[k] = v
    df = pd.DataFrame([flat])
    return df.to_csv(index=False)


###############################################################
# PDF Preview
###############################################################
def get_pdf_base64(b):
    return base64.b64encode(b).decode()


def show_pdf_preview(file_bytes, images, mode, default_page=1, zoom=1.0):
    if mode == "Images":
        if not images:
            st.error("❌ Could not convert PDF to images.")
            return
        st.write(f"Pages: {len(images)}")
        page = st.selectbox("Select Page", range(1, len(images) + 1), index=default_page - 1)
        img = images[page - 1]
        if zoom != 1.0:
            w, h = img.size
            img = img.resize((int(w * zoom), int(h * zoom)))
        st.image(img, caption=f"Page {page}", use_column_width=True)
    else:
        b64 = get_pdf_base64(file_bytes)
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="700px"></iframe>',
            unsafe_allow_html=True
        )


###############################################################
# MAIN UI
###############################################################
def main():
    st.set_page_config(page_title="Smart Invoice Reader", page_icon="📄", layout="wide")

    st.title("📄 Smart Invoice Reader — FIXED VERSION")
    st.write("Extract invoice data with digital + OCR fallback, and save to MySQL.")

    ###############################################################
    # Sidebar
    ###############################################################
    with st.sidebar:
        st.header("Settings")
        selected_lang = st.selectbox("OCR Language", Config.SUPPORTED_LANGUAGES)
        deep_mode = st.checkbox("Deep OCR (slower, more accurate)", False)
        show_raw = st.checkbox("Show Raw Text", True)
        show_conf = st.checkbox("Show Confidence", True)
        debug = st.checkbox("Debug Mode", False)

        st.header("PDF Display")
        display_method = st.selectbox("PDF Display Method", ["Images", "Embed"])

    ###############################################################
    # File upload
    ###############################################################
    uploaded_file = st.file_uploader("Upload PDF invoice", type=["pdf"])
    if not uploaded_file:
        st.info("Please upload a PDF file.")
        return

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    if len(file_bytes) > Config.MAX_FILE_SIZE:
        st.error("File too large.")
        return

    if not is_pdf_bytes(file_bytes):
        st.error("Invalid PDF.")
        return

    st.success(f"Uploaded: {uploaded_file.name}")

    # Upload PDF to S3
    try:
        uploaded_file.seek(0)
        s3.upload_fileobj(
            uploaded_file,
            S3_BUCKET,
            f"pdfs/{uploaded_file.name}"
        )
        s3_path = f"s3://{S3_BUCKET}/pdfs/{uploaded_file.name}"
    except Exception as e:
        st.error(f"S3 upload failed: {e}")
        return

    ###############################################################
    # Convert PDF → images
    ###############################################################
    with st.spinner("Converting PDF to images..."):
        images = get_pdf_images_cached(file_bytes)

    ###############################################################
    # Extract text
    ###############################################################
    with st.spinner("Extracting text..."):
        text, stats = extract_text_cached(file_bytes, selected_lang, deep_mode)

    if "error" in stats:
        st.error(stats["error"])
        return

    ###############################################################
    # Extract invoice fields
    ###############################################################
    extractor = InvoiceInfoExtractor()
    info = extractor.extract_invoice_info(text)
    info["raw_text"] = text  # REQUIRED BY DB

    if "error" in info:
        st.error(info["error"])
        return

    qa = SmartAnswerSystem()

    ###############################################################
    # Tabs
    ###############################################################
    tab_info, tab_qa, tab_preview, tab_raw, tab_analytics = st.tabs(
        ["Invoice Info", "Q&A", "PDF Preview", "Raw Text", "Analytics"]
    )

    ###############################################################
    # TAB: Invoice Info
    ###############################################################
    with tab_info:
        st.subheader("Extracted Invoice Information")

        if show_conf:
            st.write(f"Overall Confidence: {info['overall_confidence']:.1f}%")

        left, right = st.columns(2)

        with left:
            st.markdown("### Basic Fields")
            for k in ["invoice_number", "invoice_date", "total_amount", "period", "vat_number"]:
                v = info.get(k, "Not found")
                c = info["extraction_confidence"].get(k, 0)
                if show_conf:
                    st.write(f"**{k.replace('_',' ').title()}:** {v} _(conf {c:.0%})_")
                else:
                    st.write(f"**{k.replace('_',' ').title()}:** {v}")

        with right:
            st.markdown("### Party Information")
            for k in ["vendor_name", "vendor_address", "full_customer", "customer_name", "customer_person"]:
                v = info.get(k, "Not found")
                c = info["extraction_confidence"].get(k, 0)
                if show_conf:
                    st.write(f"**{k.replace('_',' ').title()}:** {v} _(conf {c:.0%})_")
                else:
                    st.write(f"**{k.replace('_',' ').title()}:** {v}")

        if info["line_items"]:
            st.subheader("Line Items")
            st.dataframe(pd.DataFrame(info["line_items"]), use_container_width=True)

        ###############################################################
        # Fixed Database Block (INDENTATION FIXED)
        ###############################################################
        with st.expander("Database", expanded=False):
            if st.button("💾 Save to database"):
                try:
                    success = insert_pdf_invoice(
                        info=info,
                        file_name=uploaded_file.name,
                        s3_path=s3_path
                    )

                    if success:
                        st.success("PDF, bill & items saved successfully!")
                    else:
                        st.error("Failed to save data to database.")

                except Exception as e:
                    st.error(f"Database insert error: {e}")

            st.divider()

            if st.button("📂 Show all PDFs"):
                try:
                    rows = fetch_all_pdfs()
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        st.info("No PDFs found.")
                except Exception as e:
                    st.error(f"Error fetching PDFs: {e}")


    ###############################################################
    # TAB: Q/A
    ###############################################################
    with tab_qa:
        st.subheader("Ask questions about the invoice")
        query = st.text_input("Ask a question:")
        if query.strip():
            answer = qa.smart_answer(query, text, info)
            st.markdown(f"**Q:** {query}\n\n**A:** {answer}")

    ###############################################################
    # TAB: PDF Preview
    ###############################################################
    with tab_preview:
        st.subheader("PDF Preview")
        show_pdf_preview(file_bytes, images, display_method)

    ###############################################################
    # TAB: Raw Text
    ###############################################################
    with tab_raw:
        if show_raw:
            st.subheader("Extracted Text")
            st.text_area("Raw Text", text, height=400)

    ###############################################################
    # TAB: Analytics
    ###############################################################
    with tab_analytics:
        st.subheader("Document Analytics")
        st.metric("Characters", len(text))
        st.metric("Words", len(text.split()))
        st.metric("Lines", len(text.splitlines()))

    ###############################################################
    # Debug block (fixed)
    ###############################################################
    if debug:
        with st.expander("Debug Data", expanded=False):
            st.write("Stats:", stats)
            st.write("Info keys:", list(info.keys()))


###############################################################
# MAIN ENTRY
###############################################################
if __name__ == "__main__":
    main()
