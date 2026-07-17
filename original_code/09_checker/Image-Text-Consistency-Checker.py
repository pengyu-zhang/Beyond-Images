# conda activate checker311
# streamlit run Image-Text-Consistency-Checker.py
import os
import re
import json
import glob
import shutil
import hashlib
import traceback
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Image Text Consistency Checker",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_TITLE = "Image Text Consistency Checker"
APP_VERSION = "v11.0 (Modern Refresh)"
SETTINGS_FILE = "settings.json"
CURATED_FILE_NAME = "dataset_curated.json"
MODIFIED_EXPORT_FILE = "dataset_modified_only.json"
ALL_EXPORT_FILE = "dataset_curated_export.json"
THUMB_DIR_NAME = ".cache/thumbnails"


LIGHT_THEME = {
    "--bg": "#f8f9fb",
    "--fg": "#111827",
    "--fg-muted": "#6b7280",
    "--card-bg": "#ffffff",
    "--border": "#e5e7eb",
    "--accent": "#2563eb",
    "--accent-fg": "#ffffff",
    "--input-bg": "#ffffff",
    "--input-border": "#d1d5db",
    "--shadow": "0 12px 32px -12px rgba(37, 99, 235, 0.25)",
    "--warn-bg": "rgba(217, 119, 6, 0.12)",
    "--warn-border": "rgba(217, 119, 6, 0.35)",
    "--warn-fg": "#92400e",
}

DARK_THEME = {
    "--bg": "#0f172a",
    "--fg": "#f8fafc",
    "--fg-muted": "#94a3b8",
    "--card-bg": "#1f2937",
    "--border": "#1e293b",
    "--accent": "#60a5fa",
    "--accent-fg": "#0f172a",
    "--input-bg": "#1e293b",
    "--input-border": "#334155",
    "--shadow": "0 12px 32px -12px rgba(96, 165, 250, 0.25)",
    "--warn-bg": "rgba(251, 191, 36, 0.2)",
    "--warn-border": "rgba(251, 191, 36, 0.45)",
    "--warn-fg": "#fcd34d",
}

COMPONENT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body, .stApp { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--fg); }
div.block-container { padding-top: 1.5rem; max-width: 1800px; }
[data-testid="stHeader"] { background: transparent; }
label, .stTextInput label, .stSelectbox label, .stToggle label { color: var(--fg) !important; font-weight: 600; }
.small-muted { color: var(--fg-muted) !important; font-size: 0.9rem; }
.info-bar { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; padding: 12px 18px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; margin-bottom: 1.75rem; }
.info-bar a { color: var(--accent); text-decoration: none; font-weight: 600; }
.info-bar a:hover { text-decoration: underline; }
.status-pill { padding: 2px 10px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em; background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-border); }
.stRadio div[role="radiogroup"] label,
.stRadio div[role="radiogroup"] label * {
    color: var(--fg-muted) !important;
}
.stToggle label span { color: var(--fg) !important; }
.stButton > button { border-radius: 10px; border: 1px solid var(--accent); background: var(--accent); color: var(--accent-fg); transition: all 0.18s ease-in-out; box-shadow: var(--shadow); }
.stButton > button[kind="secondary"] { background: var(--card-bg); color: var(--fg); border: 1px solid var(--input-border); box-shadow: none; }
.stButton > button:hover { transform: translateY(-1px); }
textarea, .stTextArea textarea { background: var(--input-bg) !important; color: var(--fg) !important; border: 1px solid var(--input-border) !important; border-radius: 10px; padding: 10px; }
.stSelectbox div[data-baseweb="select"] > div { background: var(--card-bg) !important; color: var(--fg) !important; border: 1px solid var(--input-border) !important; border-radius: 10px; padding: 10px; font-weight: 600; transition: border-color 0.18s ease-in-out, box-shadow 0.18s ease-in-out; }
textarea:focus, .stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(37,99,235,0.18) !important; }
.stSelectbox div[data-baseweb="select"] span { color: var(--fg) !important; font-weight: 500; }
.stSelectbox div[data-baseweb="select"] input { color: var(--fg) !important; }
.stSelectbox div[data-baseweb="select"] { color: var(--fg) !important; }
.stSelectbox div[data-baseweb="select"] svg { fill: var(--fg) !important; }
.stSelectbox div[data-baseweb="select"] div[role="option"] { color: var(--fg) !important; }
.image-container { position: relative; border-radius: 12px; overflow: hidden; border: 1px solid rgba(148,163,184,0.4); }
.image-container img { width: 100%; display: block; }
.image-container.removed { filter: grayscale(80%); opacity: 0.55; }
.image-container.removed::after { content: "REMOVED"; position: absolute; top: 10px; left: 10px; background: rgba(15,23,42,0.75); color: #fff; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; letter-spacing: 0.05em; }
</style>
"""


def apply_theme(name: str) -> None:
    theme_vars = DARK_THEME if name == "Dark" else LIGHT_THEME
    theme_css = ":root {" + "".join(f"{key}: {value};" for key, value in theme_vars.items()) + "}"
    st.markdown(f"<style>{theme_css}</style>{COMPONENT_CSS}", unsafe_allow_html=True)


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_json_ordered(path: str) -> OrderedDict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=OrderedDict)


def save_json_ordered(path: str, data: OrderedDict) -> None:
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def hash_str(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def safe_get(dct: Dict, path: str, default=None):
    current = dct
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def safe_set(dct: Dict, path: str, value) -> None:
    current = dct
    parts = path.split(".")
    for key in parts[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[parts[-1]] = value


def detect_qid(text: str) -> str:
    match = re.search(r"Q\d+", text or "")
    return match.group(0) if match else ""


def _natural_key_by_index(path_or_name: str) -> int:
    base = os.path.basename(path_or_name)
    match = re.search(r"_(\d+)", base)
    return int(match.group(1)) if match else 10**9


def list_images_for_qid(images_root: str, qid: str) -> List[str]:
    paths: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        pattern = os.path.join(images_root, f"{qid}_*{ext}")
        paths.extend(glob.glob(pattern))
    paths.sort(key=_natural_key_by_index)
    return paths


def build_qid_index(data: OrderedDict) -> OrderedDict:
    qid_map: OrderedDict[str, Tuple[str, dict]] = OrderedDict()
    for entity_key, record in data.items():
        qid = record.get("entity_qid") or detect_qid(entity_key)
        if qid:
            qid_map[qid] = (entity_key, record)
    return qid_map


def curated_path_of(text_json_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(text_json_path))
    return os.path.join(base_dir, CURATED_FILE_NAME)


def thumb_dir(images_root: str) -> str:
    return os.path.join(images_root, THUMB_DIR_NAME)


def make_thumbnail(img_path: str, out_dir: str, size: Tuple[int, int] = (360, 360)) -> str:
    ensure_dir(out_dir)
    key = f"{os.path.basename(img_path)}_{int(os.path.getmtime(img_path))}"
    thumb_path = os.path.join(out_dir, hash_str(key) + ".jpg")
    if not os.path.exists(thumb_path):
        try:
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image)
            image.thumbnail(size)
            image = image.convert("RGB")
            image.save(thumb_path, quality=88)
        except Exception:
            shutil.copyfile(img_path, thumb_path)
    return thumb_path


def init_curated_copy(text_json_path: str) -> str:
    curated_path = curated_path_of(text_json_path)
    if not os.path.exists(curated_path):
        data = load_json_ordered(text_json_path)
        for record in data.values():
            record.setdefault(
                "_curation",
                {
                    "verdict": None,
                    "reason": None,
                    "notes": "",
                    "removed_images": [],
                    "annotator": "",
                    "last_updated": None,
                    "modified": False,
                },
            )
        save_json_ordered(curated_path, data)
    return curated_path


def format_filesize(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def count_status(curated_data: OrderedDict) -> Dict[str, int]:
    total = len(curated_data)
    reviewed = 0
    mismatch = 0
    uncertain = 0
    for record in curated_data.values():
        verdict = record.get("_curation", {}).get("verdict")
        if verdict:
            reviewed += 1
            if verdict == "mismatch":
                mismatch += 1
            elif verdict == "uncertain":
                uncertain += 1
    return {
        "total": total,
        "reviewed": reviewed,
        "pending": total - reviewed,
        "mismatch": mismatch,
        "uncertain": uncertain,
    }


def export_modified_only(curated_data: OrderedDict, export_path: str) -> None:
    modified = OrderedDict(
        (key, record)
        for key, record in curated_data.items()
        if record.get("_curation", {}).get("modified")
    )
    save_json_ordered(export_path, modified)


def _mark_save() -> None:
    st.session_state["_do_save"] = True


def _mark_save_next() -> None:
    st.session_state["_do_save_next"] = True


def _mark_prev() -> None:
    st.session_state["_do_prev"] = True


def _mark_next() -> None:
    st.session_state["_do_next"] = True


def inject_keyboard_shortcuts() -> None:
    js_code = """
    <script>
    document.addEventListener('keydown', (event) => {
        const withinInput = ['INPUT', 'TEXTAREA'].includes(event.target.tagName);
        const clickByLabel = (label) => {
            const buttons = window.parent.document.querySelectorAll('button');
            for (const button of buttons) {
                if (button.innerText.trim() === label) {
                    button.click();
                    return true;
                }
            }
            return false;
        };
        if (withinInput) {
            if (event.ctrlKey && event.key.toLowerCase() === 's') {
                event.preventDefault();
                clickByLabel('Save');
            }
            return;
        }
        if (event.key === 'ArrowRight') {
            event.preventDefault();
            clickByLabel('Next');
        } else if (event.key === 'ArrowLeft') {
            event.preventDefault();
            clickByLabel('Previous');
        } else if (event.ctrlKey && event.key.toLowerCase() === 's') {
            event.preventDefault();
            clickByLabel('Save');
        }
    });
    </script>
    """
    components.html(js_code, height=0, width=0)


def load_settings() -> Dict:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "text_json_path": "",
        "images_root": "",
        "theme": "Light",
        "annotator": "",
        "danger_delete_files": False,
        "auto_save": True,
        "llm_model": "None",
    }


def save_settings(cfg: Dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, ensure_ascii=False, indent=2)


def track_pristine_state(record: Dict) -> None:
    st.session_state.pristine_record = {
        "text": safe_get(record, "images.images_t5_descriptions", ""),
        "verdict": record.get("_curation", {}).get("verdict"),
        "reason": record.get("_curation", {}).get("reason"),
        "notes": record.get("_curation", {}).get("notes", ""),
        "removed_images": record.get("_curation", {}).get("removed_images", []),
    }
    st.session_state.is_dirty = False


def check_for_unsaved_changes(qid: str) -> None:
    if "pristine_record" not in st.session_state:
        return
    pristine = st.session_state.pristine_record
    _, record = st.session_state.qid_map[qid]
    cur_text = st.session_state.get(f"text_{qid}", pristine["text"])
    cur_verdict = st.session_state.get(f"rail_verdict_{qid}", pristine["verdict"])
    cur_reason = st.session_state.get(f"rail_reason_{qid}", pristine["reason"])
    cur_notes = st.session_state.get(f"rail_notes_{qid}", pristine["notes"])
    cur_removed = record.get("_curation", {}).get("removed_images", [])
    has_changes = (
        cur_text != pristine["text"]
        or cur_verdict != pristine["verdict"]
        or (cur_verdict == "mismatch" and cur_reason != pristine["reason"])
        or cur_notes != pristine["notes"]
        or set(cur_removed) != set(pristine["removed_images"])
    )
    st.session_state.is_dirty = has_changes


def ensure_session_defaults() -> None:
    st.session_state.setdefault("page", "Setup")
    st.session_state.setdefault("filter_mode", "All")
    st.session_state.setdefault("gallery_columns", 3)
    st.session_state.setdefault("show_removed_images", True)
    st.session_state.setdefault("idx", 0)


def dataset_loaded() -> bool:
    return all(key in st.session_state for key in ("paths", "data_cur", "qid_map"))


def filtered_qids() -> List[str]:
    if "qid_map" not in st.session_state:
        return []
    mode = st.session_state.get("filter_mode", "All")
    qids = list(st.session_state.qid_map.keys())
    if mode == "All":
        return qids
    filtered: List[str] = []
    for qid in qids:
        record = st.session_state.qid_map[qid][1]
        verdict = record.get("_curation", {}).get("verdict")
        if mode == "Pending" and not verdict:
            filtered.append(qid)
        elif mode == "Reviewed" and verdict:
            filtered.append(qid)
        elif mode == "Mismatch" and verdict == "mismatch":
            filtered.append(qid)
        elif mode == "Uncertain" and verdict == "uncertain":
            filtered.append(qid)
    return filtered


def render_progress_summary(counts: Dict[str, int]) -> None:
    total = counts.get("total", 0)
    reviewed = counts.get("reviewed", 0)
    ratio = reviewed / total if total else 0
    st.progress(ratio, text=f"Reviewed {reviewed} of {total} records")
    metric_cols = st.columns(4)
    for col, (label, key) in zip(
        metric_cols,
        [
            ("Pending", "pending"),
            ("Reviewed", "reviewed"),
            ("Mismatch", "mismatch"),
            ("Uncertain", "uncertain"),
        ],
    ):
        col.metric(label, counts.get(key, 0))


def render_record_header(
    entity_key: str,
    record: Dict,
    qid: str,
    index: int,
    total: int,
    is_dirty: bool,
) -> str:
    entity_name = record.get("entity_name") or entity_key
    link_parts = []
    for label, key in (("DBpedia", "dbpedia_url"), ("Wikidata", "wikidata_url")):
        url = record.get(key)
        if url:
            link_parts.append(f"<a href='{url}' target='_blank'>{label}</a>")
    links_html = " • ".join(link_parts) if link_parts else "&nbsp;"
    status_html = "<span class='status-pill'>Unsaved changes</span>" if is_dirty else ""
    st.markdown(
        f"""
        <div class="info-bar">
            <span><strong>Entity:</strong> {entity_name}</span>
            <span><strong>QID:</strong> {qid}</span>
            <span>{links_html}</span>
            <span class="small-muted" style="margin-left: auto;">Record {index + 1} of {total}</span>
            {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return entity_name


def render_action_panel(record: Dict, qid: str, total_records: int) -> None:
    cur = record.get("_curation", {})
    nav_prev, nav_next = st.columns(2)
    nav_prev.button(
        "Previous",
        use_container_width=True,
        on_click=_mark_prev,
        disabled=st.session_state.idx <= 0,
        type="secondary",
    )
    nav_next.button(
        "Next",
        use_container_width=True,
        on_click=_mark_next,
        disabled=st.session_state.idx >= total_records - 1,
        type="secondary",
    )
    save_col, save_next_col = st.columns(2)
    save_col.button("Save", type="primary", use_container_width=True, on_click=_mark_save)
    save_next_col.button(
        "Save & Next",
        use_container_width=True,
        on_click=_mark_save_next,
    )
    st.caption(f"Auto-save on navigation: {'On' if st.session_state.cfg.get('auto_save', True) else 'Off'}")
    st.divider()
    verdict_options = ["match", "mismatch", "uncertain"]
    default_verdict = cur.get("verdict") or "match"
    selected_verdict = st.radio(
        "Verdict",
        verdict_options,
        index=verdict_options.index(default_verdict),
        key=f"rail_verdict_{qid}",
    )
    if selected_verdict == "mismatch":
        reason_options = ["wrong_entity", "irrelevant", "low_quality", "other"]
        default_reason = cur.get("reason")
        reason_index = reason_options.index(default_reason) if default_reason in reason_options else 0
        st.selectbox(
            "Mismatch reason",
            reason_options,
            index=reason_index,
            key=f"rail_reason_{qid}",
        )
    st.text_area(
        "Notes",
        value=cur.get("notes", ""),
        height=130,
        placeholder="Add brief rationale or evidence...",
        key=f"rail_notes_{qid}",
    )


def render_text_panels(record: Dict, qid: str) -> None:
    description = safe_get(record, "images.images_t5_descriptions", "")
    st.subheader("Text Description")
    st.text_area(
        "Description",
        value=description,
        height=260,
        placeholder="No description available.",
        key=f"text_{qid}",
        label_visibility="collapsed",
    )
    st.caption(f"Characters: {len(description)}")
    st.markdown("---")
    header_cols = st.columns([0.6, 0.4])
    header_cols[0].subheader("LLM Assist")
    header_cols[1].selectbox(
        "Model",
        ["None", "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro"],
        key=f"llm_model_{qid}",
    )
    st.text_area(
        "LLM Output",
        height=220,
        placeholder="Paste or write the generated description here...",
        key=f"llm_{qid}",
        label_visibility="collapsed",
    )
    # st.caption("This section is a staging area only; no API calls are made.")


def update_removed_images(record: Dict, entity_key: str, qid: str, filename: str, remove: bool) -> None:
    curation = record.setdefault("_curation", {})
    removed_images = set(curation.get("removed_images", []))
    if remove:
        removed_images.add(filename)
    else:
        removed_images.discard(filename)
    curation["removed_images"] = sorted(removed_images, key=_natural_key_by_index)
    curation["modified"] = True
    curation["last_updated"] = now_iso()
    annotator = st.session_state.cfg.get("annotator", "")
    if annotator:
        curation["annotator"] = annotator
    st.session_state.data_cur[entity_key] = record
    st.session_state.qid_map[qid] = (entity_key, record)
    st.session_state.is_dirty = True
    st.rerun()


def render_image_gallery(record: Dict, entity_key: str, qid: str, entity_name: str) -> None:
    st.subheader(f"Images · {entity_name}")
    images_root = st.session_state.paths["images_root"]
    thumbs_root = st.session_state.paths["thumb_dir"]
    ensure_dir(thumbs_root)
    all_images = list_images_for_qid(images_root, qid)
    removed_set = set(record.get("_curation", {}).get("removed_images", []))
    include_removed = st.toggle(
        "Show removed images",
        value=st.session_state.get("show_removed_images", True),
        key="show_removed_images",
    )
    if not include_removed:
        all_images = [path for path in all_images if os.path.basename(path) not in removed_set]
    if not all_images:
        message = "No images found for this QID." if not removed_set else "No images to display with the current filter."
        st.info(message)
        return
    columns = max(1, st.session_state.get("gallery_columns", 2))
    grid_columns = st.columns(columns)
    for index, img_path in enumerate(all_images):
        target_col = grid_columns[index % columns]
        with target_col:
            fname = os.path.basename(img_path)
            is_removed = fname in removed_set
            container_class = "image-container removed" if is_removed else "image-container"
            try:
                thumb_path = make_thumbnail(img_path, thumbs_root)
            except FileNotFoundError:
                continue
            st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
            st.image(thumb_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            try:
                size_text = format_filesize(os.path.getsize(img_path))
            except OSError:
                size_text = "missing"
            st.caption(f"{fname} · {size_text}")
            button_label = "Restore" if is_removed else "Remove"
            if st.button(
                button_label,
                key=f"{qid}_{fname}",
                use_container_width=True,
                type="secondary",
            ):
                update_removed_images(record, entity_key, qid, fname, remove=not is_removed)


def setup_page() -> None:
    st.title(APP_TITLE)
    st.caption("Load your dataset and start verifying descriptions with a refreshed workflow.")
    cfg = st.session_state.cfg
    dataset_ready = dataset_loaded()
    if dataset_ready:
        counts = count_status(st.session_state.data_cur)
        st.success(
            f"Dataset loaded from **{cfg.get('text_json_path', 'unknown path')}** · {counts['total']} records."
        )
    with st.form("setup_form"):
        st.subheader("Dataset configuration")
        text_json_path = st.text_input(
            "Path to text JSON file",
            value=cfg.get("text_json_path", ""),
            placeholder="C:\\data\\dataset.json",
        )
        images_root = st.text_input(
            "Path to images folder",
            value=cfg.get("images_root", ""),
            placeholder="C:\\data\\images",
        )
        col1, col2 = st.columns(2)
        annotator = col1.text_input("Annotator name (optional)", value=cfg.get("annotator", ""))
        auto_save = col1.toggle("Auto-save when navigating", value=cfg.get("auto_save", True))
        danger = col2.toggle(
            "Delete removed image files from disk",
            value=cfg.get("danger_delete_files", False),
            help="When enabled, removed images are deleted permanently on save.",
        )
        submitted = st.form_submit_button("Load dataset", type="primary", use_container_width=True)
    if submitted:
        errors = []
        if not file_exists(text_json_path):
            errors.append("Text JSON file not found.")
        if not os.path.isdir(images_root):
            errors.append("Images folder not found.")
        if errors:
            for message in errors:
                st.error(message)
            return
        cfg.update(
            {
                "text_json_path": text_json_path,
                "images_root": images_root,
                "annotator": annotator,
                "auto_save": auto_save,
                "danger_delete_files": danger,
            }
        )
        save_settings(cfg)
        with st.spinner("Preparing curated dataset..."):
            curated_path = init_curated_copy(text_json_path)
            st.session_state.paths = {
                "text": text_json_path,
                "curated": curated_path,
                "images_root": images_root,
                "thumb_dir": thumb_dir(images_root),
            }
            st.session_state.data_cur = load_json_ordered(curated_path)
            st.session_state.qid_map = build_qid_index(st.session_state.data_cur)
            st.session_state.qids = list(st.session_state.qid_map.keys())
        st.session_state.idx = 0
        st.session_state.filter_mode = "All"
        st.session_state.page = "Main UI"
        st.success("Dataset loaded. Redirecting to the main workspace…")
        st.rerun()
    st.divider()
    with st.expander("Tips for getting started", expanded=not dataset_ready):
        st.markdown(
            """
            - Use absolute paths to avoid surprises after restarting the app.
            - The curated JSON is saved alongside the original file (`dataset_curated.json`).
            - Removed images are only hidden by default—enable deletion in the settings if you want to clean up disk usage.
            - Keyboard shortcuts: <kbd>Ctrl</kbd> + <kbd>S</kbd> to save, arrow keys to move between records.
            """
        )


def sidebar_nav() -> None:
    cfg = st.session_state.cfg
    with st.sidebar:
        st.markdown(f"## {APP_TITLE}")
        st.caption(APP_VERSION)
        theme_is_dark = cfg.get("theme", "Light") == "Dark"
        theme_toggle = st.toggle("🌙 Dark mode", value=theme_is_dark, key="theme_toggle")
        desired_theme = "Dark" if theme_toggle else "Light"
        if desired_theme != cfg.get("theme"):
            cfg["theme"] = desired_theme
            save_settings(cfg)
            st.session_state.cfg = cfg
            st.rerun()
        pages = ["Setup", "Main UI", "Settings", "About"]
        current_page = st.session_state.get("page", "Setup")
        page_index = pages.index(current_page) if current_page in pages else 0
        selected_page = st.radio("Navigate", pages, index=page_index)
        st.session_state.page = selected_page
        if dataset_loaded():
            st.divider()
            st.markdown("### Progress")
            counts = count_status(st.session_state.data_cur)
            for label, key in [
                ("Total", "total"),
                ("Reviewed", "reviewed"),
                ("Pending", "pending"),
                ("Mismatch", "mismatch"),
                ("Uncertain", "uncertain"),
            ]:
                st.write(f"{label}: **{counts[key]}**")
            st.divider()
            st.markdown("### Filters")
            modes = ["All", "Pending", "Reviewed", "Mismatch", "Uncertain"]
            current_mode = st.session_state.get("filter_mode", "All")
            st.session_state.filter_mode = st.selectbox(
                "Curation status",
                modes,
                index=modes.index(current_mode),
            )
            st.session_state.gallery_columns = st.slider(
                "Image grid columns",
                1,
                4,
                value=st.session_state.get("gallery_columns", 2),
            )
            total_records = len(st.session_state.qids)
            st.markdown("### Quick Jump")
            if total_records:
                st.caption(f"Total records: {total_records}")
                with st.form("jump_to_record", clear_on_submit=False):
                    current_position = st.session_state.get("idx", 0) + 1
                    default_position = min(max(current_position, 1), total_records)
                    target_position = st.number_input(
                        "Record position (1-based)",
                        min_value=1,
                        max_value=total_records,
                        value=default_position,
                        step=1,
                        format="%d",
                        help=f"Enter a record number between 1 and {total_records}.",
                    )
                    jump = st.form_submit_button("Go", use_container_width=True)
                    if jump:
                        st.session_state.filter_mode = "All"
                        st.session_state.pending_target_idx = int(target_position) - 1
                        st.rerun()
            else:
                st.caption("Load a dataset to enable record jumping.")
            st.divider()
            st.markdown("### Export")
            if st.button("Export modified only", use_container_width=True):
                export_modified_only(st.session_state.data_cur, MODIFIED_EXPORT_FILE)
                st.success(f"Saved to {MODIFIED_EXPORT_FILE}")
            if st.button("Export full curated dataset", use_container_width=True):
                save_json_ordered(ALL_EXPORT_FILE, st.session_state.data_cur)
                st.success(f"Saved to {ALL_EXPORT_FILE}")


def main_ui() -> None:
    available_qids = filtered_qids()
    if not available_qids:
        st.session_state.view_qids = []
        st.info("No records match the current filter yet.")
        return
    if st.session_state.get("view_qids") != available_qids:
        st.session_state.view_qids = available_qids
    pending_idx = st.session_state.pop("pending_target_idx", None)
    current_idx = st.session_state.get("idx", 0)
    if pending_idx is not None:
        current_idx = pending_idx
    current_idx = max(0, min(current_idx, len(available_qids) - 1))
    st.session_state.idx = current_idx
    qid = available_qids[st.session_state.idx]
    entity_key, record = st.session_state.qid_map[qid]
    if st.session_state.get("current_qid") != qid:
        st.session_state.current_qid = qid
        track_pristine_state(record)
    check_for_unsaved_changes(qid)
    inject_keyboard_shortcuts()
    counts = count_status(st.session_state.data_cur)
    render_progress_summary(counts)
    dirty = st.session_state.get("is_dirty", False)
    entity_name = render_record_header(
        entity_key,
        record,
        qid,
        st.session_state.idx,
        len(available_qids),
        dirty,
    )
    action_col, text_col, image_col = st.columns((0.26, 0.38, 0.36), gap="large")
    with action_col:
        render_action_panel(record, qid, len(available_qids))
    with text_col:
        render_text_panels(record, qid)
    with image_col:
        render_image_gallery(record, entity_key, qid, entity_name)


def apply_edited_fields(record: Dict, edited_text: str, verdict: str, reason: Optional[str], notes: str) -> bool:
    changed = False
    text_path = "images.images_t5_descriptions"
    if edited_text != safe_get(record, text_path, ""):
        safe_set(record, text_path, edited_text)
        changed = True
    curation = record.setdefault("_curation", {})
    if verdict != curation.get("verdict"):
        curation["verdict"] = verdict
        changed = True
    if verdict == "mismatch":
        if reason != curation.get("reason"):
            curation["reason"] = reason
            changed = True
    else:
        if "reason" in curation:
            del curation["reason"]
            changed = True
    if notes != curation.get("notes", ""):
        curation["notes"] = notes
        changed = True
    if changed:
        curation["modified"] = True
        curation["last_updated"] = now_iso()
        annotator = st.session_state.cfg.get("annotator", "")
        if annotator:
            curation["annotator"] = annotator
    return changed


def save_current(qid: str, feedback_container) -> None:
    try:
        entity_key, record = st.session_state.qid_map[qid]
        curated_path = st.session_state.paths["curated"]
        data_cur = st.session_state.data_cur
        data_cur[entity_key] = record
        save_json_ordered(curated_path, data_cur)
        st.session_state.data_cur = data_cur
        st.session_state.qid_map = build_qid_index(data_cur)
        st.session_state.qids = list(st.session_state.qid_map.keys())
        track_pristine_state(record)
        if feedback_container:
            feedback_container.success("Changes saved.")
        if st.session_state.cfg.get("danger_delete_files", False):
            _danger_delete_files_for_record(record)
    except Exception:
        st.error("Save failed.")
        st.code(traceback.format_exc())


def _danger_delete_files_for_record(record: Dict) -> None:
    images_root = st.session_state.paths["images_root"]
    for filename in record.get("_curation", {}).get("removed_images", []):
        target = os.path.join(images_root, filename)
        if os.path.exists(target):
            os.remove(target)


def settings_page() -> None:
    st.title("Settings")
    cfg = st.session_state.cfg
    with st.form("settings_form"):
        annotator = st.text_input("Annotator name (optional)", value=cfg.get("annotator", ""))
        auto_save = st.toggle("Auto-save on navigation", value=cfg.get("auto_save", True))
        danger = st.toggle(
            "Permanently delete removed images",
            value=cfg.get("danger_delete_files", False),
            help="When enabled, removed image files are deleted from disk during save.",
        )
        submitted = st.form_submit_button("Save settings", type="primary")
    if submitted:
        cfg.update(
            {
                "annotator": annotator,
                "auto_save": auto_save,
                "danger_delete_files": danger,
            }
        )
        save_settings(cfg)
        st.success("Settings saved.")


def about_page() -> None:
    st.title("About")
    st.markdown(f"**{APP_TITLE}** — {APP_VERSION}")
    st.markdown(
        """
        This local-first tool streamlines the human review of image and caption consistency.

        - Modern, responsive layout tuned for productivity.
        - Keyboard shortcuts for rapid annotation (`Ctrl + S`, `←`, `→`).
        - Curated data is stored alongside the original dataset for easy reintegration.
        """
    )
    st.markdown("Built with Streamlit and focused on offline workflows.")


def main() -> None:
    if "cfg" not in st.session_state:
        st.session_state.cfg = load_settings()
    ensure_session_defaults()
    apply_theme(st.session_state.cfg.get("theme", "Light"))
    sidebar_nav()
    page = st.session_state.get("page", "Setup")
    if page not in {"Setup", "About"} and not dataset_loaded():
        st.warning("Please complete the Setup page first.")
        st.session_state.page = "Setup"
        st.rerun()
        return
    if page == "Setup":
        setup_page()
        return
    if page == "Main UI":
        feedback_placeholder = st.empty()
        main_ui()
        view_qids = st.session_state.get("view_qids", [])
        if not view_qids:
            return
        qid = view_qids[st.session_state.idx]
        _, record = st.session_state.qid_map[qid]
        save_and_stay = st.session_state.pop("_do_save", False)
        save_and_next = st.session_state.pop("_do_save_next", False)
        if save_and_stay or save_and_next:
            changed = apply_edited_fields(
                record,
                st.session_state.get(f"text_{qid}", ""),
                st.session_state.get(f"rail_verdict_{qid}", "match"),
                st.session_state.get(f"rail_reason_{qid}"),
                st.session_state.get(f"rail_notes_{qid}", ""),
            )
            if changed or save_and_stay or save_and_next:
                save_current(qid, feedback_placeholder)
        if save_and_next and st.session_state.idx < len(view_qids) - 1:
            st.session_state.idx += 1
            st.rerun()
        elif save_and_stay:
            st.rerun()
        auto_save_on_nav = st.session_state.cfg.get("auto_save", True) and st.session_state.get("is_dirty", False)
        if st.session_state.pop("_do_next", False):
            if auto_save_on_nav:
                apply_edited_fields(
                    record,
                    st.session_state.get(f"text_{qid}", ""),
                    st.session_state.get(f"rail_verdict_{qid}", "match"),
                    st.session_state.get(f"rail_reason_{qid}"),
                    st.session_state.get(f"rail_notes_{qid}", ""),
                )
                save_current(qid, feedback_container=None)
            if st.session_state.idx < len(view_qids) - 1:
                st.session_state.idx += 1
                st.rerun()
        if st.session_state.pop("_do_prev", False):
            if auto_save_on_nav:
                apply_edited_fields(
                    record,
                    st.session_state.get(f"text_{qid}", ""),
                    st.session_state.get(f"rail_verdict_{qid}", "match"),
                    st.session_state.get(f"rail_reason_{qid}"),
                    st.session_state.get(f"rail_notes_{qid}", ""),
                )
                save_current(qid, feedback_container=None)
            if st.session_state.idx > 0:
                st.session_state.idx -= 1
                st.rerun()
        return
    if page == "Settings":
        settings_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
