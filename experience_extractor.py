# app.py
import streamlit as st
import os
import re
import json
import tempfile
import time
from datetime import datetime
from typing import Optional

# ---------- Styling (colorful UI) ----------
st.set_page_config(page_title="Experience Extractor", layout="wide")
st.markdown(
    """
    <style>
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 6px 20px rgba(2,8,23,0.6);
    }
    .title {
        font-size:20px; font-weight:800; margin-bottom:6px;
    }
    .subtle { color: #bfc6cc; font-size:13px; }
    .big-num { font-size:28px; font-weight:800; color:#ffd66b; }
    .human { font-size:18px; font-weight:700; color:#9ef3c9; }
    .note { font-size:13px; color:#b3c1c9; margin-top:8px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:18px; align-items:start; }
    .file-row { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
    .file-name { font-weight:700; color:#7ed0ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Optional dependencies ----------
# We reuse the same parsers as the original script. If missing, we show helpful errors.
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# ---------- Gemini client attempt (optional) ----------
GEMINI_CLIENT = None
try:
    from google import genai as _genai
    GEMINI_CLIENT = _genai
except Exception:
    GEMINI_CLIENT = None

# ---------- Key retrieval (no hardcoded key) ----------
def get_gemini_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()

# ---------- Text extraction (from original script) ----------
def extract_text_from_pdf(path):
    if pdfplumber is None:
        raise RuntimeError("Install pdfplumber: pip install pdfplumber")
    parts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            parts.append(p.extract_text() or "")
    raw = " ".join(parts)
    return re.sub(r'\s+', ' ', raw).strip()

def extract_text_from_docx(path):
    if docx is None:
        raise RuntimeError("Install python-docx: pip install python-docx")
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text and p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text and cell.text.strip():
                    parts.append(cell.text.strip())
    raw = " ".join(parts)
    return re.sub(r'\s+', ' ', raw).strip()

def extract_text_from_txt(path):
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    return re.sub(r'\s+', ' ', text).strip()

def extract_text(path):
    p = path.lower()
    if p.endswith(".pdf"):
        return extract_text_from_pdf(path)
    if p.endswith(".docx"):
        return extract_text_from_docx(path)
    if p.endswith(".txt"):
        return extract_text_from_txt(path)
    raise ValueError("Unsupported file type. Supported: .pdf, .docx, .txt")

# ---------- Heuristic fallback for years (from original) ----------
def fallback_years(text):
    vals = []
    if not text:
        return 0.0
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)', text.lower()):
        try:
            vals.append(float(m.group(1)))
        except:
            pass
    return round(max(vals), 1) if vals else 0.0

# ---------- Gemini LLM call (adapted, no hardcoded key) ----------
def ask_gemini_for_years(text: str, api_key: str, max_chars=12000) -> float:
    if not text:
        return 0.0
    if len(text) > max_chars:
        text = text[:max_chars]

    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
Return ONLY JSON: {{ "total_years": <float> }}.

Compute TOTAL professional work experience:
- Merge overlapping roles.
- Convert months into decimal years (1 decimal).
- Treat "present/current" as {today}.
- If unsure → return 0.0
- DO NOT output anything except JSON.

Resume:
\"\"\"{text}\"\"\"
""".strip()

    # If no client or key, fallback
    if not api_key or GEMINI_CLIENT is None:
        return fallback_years(text)

    # try to build client (two common forms)
    client = None
    try:
        client = GEMINI_CLIENT.Client(api_key=api_key)
    except Exception:
        try:
            GEMINI_CLIENT.configure(api_key=api_key)
            client = GEMINI_CLIENT
        except Exception:
            client = None

    if client is None:
        return fallback_years(text)

    # call model
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = resp.text if hasattr(resp, "text") else str(resp)

        m = re.search(r'\{.*\}', raw, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                val = float(obj.get("total_years", 0.0))
                return round(val, 1)
            except Exception:
                pass

        m2 = re.search(r'(\d+(?:\.\d+)?)', raw)
        if m2:
            return round(float(m2.group(1)), 1)

        return fallback_years(text)
    except Exception:
        return fallback_years(text)

# ---------- convert decimal years to human readable ----------
def convert_decimal_to_human(decimal_years, method="round"):
    years = int(decimal_years)
    if method == "floor":
        months = int((decimal_years - years) * 12)
    else:
        months = int(round((decimal_years - years) * 12))

    if months >= 12:
        years += 1
        months -= 12

    if years == 0 and months == 0:
        return "0 years"
    if months == 0:
        return f"{years} years"
    if years == 0:
        return f"{months} months"
    return f"{years} years {months} months"

# ---------- Main UI ----------
st.title("🎯 Experience Extractor — Resume Batch")
st.write("Upload resumes (.pdf / .docx / .txt). The app will attempt to use Gemini if `GEMINI_API_KEY` is set in `st.secrets` or environment. Otherwise a heuristic fallback is used.")

api_key = get_gemini_key()
if api_key:
    st.success("Gemini API key loaded — model extraction enabled.")
else:
    st.info("No Gemini API key found — using local heuristic fallback. To enable model extraction, add `GEMINI_API_KEY` in Streamlit Secrets or environment variables.")

uploaded = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded:
    # process when the user clicks
    if st.button("🔎 Extract Experience from Uploads"):
        results = {}
        with st.spinner("Processing resumes — extracting text and computing experience..."):
            for f in uploaded:
                # write to temp file path for parser
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    text = extract_text(tmp_path)
                    decimal = ask_gemini_for_years(text, api_key)
                    human = convert_decimal_to_human(decimal, method="round")
                    results[f.name] = {"decimal": decimal, "human": human}
                except Exception as e:
                    results[f.name] = {"error": str(e)}
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        # ---------- Display full-width cards ----------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### Results")

for fname, out in results.items():
    if "error" in out:
        st.markdown(
            f'<div class="card"><div class="title">{fname}</div>'
            f'<div class="note">Error: {out["error"]}</div></div>',
            unsafe_allow_html=True,
        )
        continue

    decimal = out["decimal"]
    human = out["human"]

    html = f'''
    <div class="card" style="width:100%;">
        <div class="file-row">
            <div class="file-name">{fname}</div>
        </div>

        <div style="display:flex; gap:18px; align-items:center; margin-top:10px;">
            <div>
                <div class="subtle">Experience</div>
                <div class="human">{human}</div>
            </div>
        </div>

        <div class="note">Computed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

else:
    st.info("Upload one or more resume files to get started.")
