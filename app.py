
import os
import io
import json
import re
from copy import deepcopy
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

load_dotenv()

st.set_page_config(
    page_title="Professional Software Estimation Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKSPACE_FILE = "project_workspace.json"
HISTORY_FILE = "estimation_history.csv"
AI_MODEL_NAME = "gpt-4.1-mini"
APP_SCHEMA_VERSION = "2.1.0"

MODES = {
    "Organic": {"a": 3.2, "b": 1.05, "c": 2.5, "d": 0.38},
    "Semi-detached": {"a": 3.0, "b": 1.12, "c": 2.5, "d": 0.35},
    "Embedded": {"a": 2.8, "b": 1.20, "c": 2.5, "d": 0.32},
}

COST_DRIVER_GROUPS = {
    "Product Attributes": ["RELY", "DATA", "CPLX"],
    "Computer / Platform Attributes": ["TIME", "STOR", "VIRT", "TURN"],
    "Personnel Attributes": ["ACAP", "AEXP", "PCAP", "VEXP", "LEXP"],
    "Project Attributes": ["MODP", "TOOL", "SCED"],
}

COST_DRIVERS = {
    "RELY": {"label": "Required Software Reliability", "help": "Mức độ yêu cầu độ tin cậy của phần mềm.", "values": {"Very Low": 0.75, "Low": 0.88, "Nominal": 1.00, "High": 1.15, "Very High": 1.40}},
    "DATA": {"label": "Database Size", "help": "Quy mô dữ liệu mà hệ thống cần xử lý.", "values": {"Low": 0.94, "Nominal": 1.00, "High": 1.08, "Very High": 1.16}},
    "CPLX": {"label": "Product Complexity", "help": "Độ phức tạp tổng thể của sản phẩm.", "values": {"Very Low": 0.70, "Low": 0.85, "Nominal": 1.00, "High": 1.15, "Very High": 1.30, "Extra High": 1.65}},
    "TIME": {"label": "Execution Time Constraint", "help": "Ràng buộc về thời gian thực thi.", "values": {"Nominal": 1.00, "High": 1.11, "Very High": 1.30, "Extra High": 1.66}},
    "STOR": {"label": "Storage Constraint", "help": "Ràng buộc về bộ nhớ hoặc lưu trữ.", "values": {"Nominal": 1.00, "High": 1.06, "Very High": 1.21, "Extra High": 1.56}},
    "VIRT": {"label": "Virtual Machine Volatility", "help": "Mức độ thay đổi của môi trường chạy.", "values": {"Low": 0.87, "Nominal": 1.00, "High": 1.15, "Very High": 1.30}},
    "TURN": {"label": "Computer Turnaround Time", "help": "Tốc độ phản hồi và chu kỳ xử lý của hệ thống.", "values": {"Low": 0.87, "Nominal": 1.00, "High": 1.07, "Very High": 1.15}},
    "ACAP": {"label": "Analyst Capability", "help": "Năng lực của người phân tích hệ thống.", "values": {"Very Low": 1.46, "Low": 1.19, "Nominal": 1.00, "High": 0.86, "Very High": 0.71}},
    "AEXP": {"label": "Application Experience", "help": "Kinh nghiệm với domain nghiệp vụ tương tự.", "values": {"Very Low": 1.29, "Low": 1.13, "Nominal": 1.00, "High": 0.91, "Very High": 0.82}},
    "PCAP": {"label": "Programmer Capability", "help": "Năng lực lập trình viên.", "values": {"Very Low": 1.42, "Low": 1.17, "Nominal": 1.00, "High": 0.86, "Very High": 0.70}},
    "VEXP": {"label": "Virtual Machine Experience", "help": "Kinh nghiệm với môi trường/hệ thống mục tiêu.", "values": {"Very Low": 1.21, "Low": 1.10, "Nominal": 1.00, "High": 0.90}},
    "LEXP": {"label": "Language Experience", "help": "Kinh nghiệm với ngôn ngữ lập trình đang dùng.", "values": {"Very Low": 1.14, "Low": 1.07, "Nominal": 1.00, "High": 0.95}},
    "MODP": {"label": "Modern Programming Practices", "help": "Mức độ áp dụng thực hành lập trình hiện đại.", "values": {"Very Low": 1.24, "Low": 1.10, "Nominal": 1.00, "High": 0.91, "Very High": 0.82}},
    "TOOL": {"label": "Use of Software Tools", "help": "Mức độ hỗ trợ của tool, IDE, testing, CI/CD...", "values": {"Very Low": 1.24, "Low": 1.10, "Nominal": 1.00, "High": 0.91, "Very High": 0.83}},
    "SCED": {"label": "Required Development Schedule", "help": "Mức độ gắt của deadline.", "values": {"Very Low": 1.23, "Low": 1.08, "Nominal": 1.00, "High": 1.04, "Very High": 1.10}},
}

PRESET_PROJECTS = {
    "Start from Scratch": None,
    "Personal Expense Tracker": {"project_name": "Personal Expense Tracker", "description": "A simple personal expense tracking web application with category management, monthly reports, and budget summaries.", "mode": "Organic", "kloc": 8.0, "cost_per_pm": 12000000.0, "drivers": {"RELY": "Low", "DATA": "Low", "CPLX": "Low", "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Low", "TURN": "Low", "ACAP": "High", "AEXP": "High", "PCAP": "High", "VEXP": "Nominal", "LEXP": "High", "MODP": "High", "TOOL": "Very High", "SCED": "Nominal"}},
    "School Management System": {"project_name": "School Management System", "description": "A school management system with student records, teacher management, attendance tracking, exam scores, and report generation.", "mode": "Organic", "kloc": 18.0, "cost_per_pm": 12000000.0, "drivers": {"RELY": "Nominal", "DATA": "Nominal", "CPLX": "Nominal", "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal", "ACAP": "Nominal", "AEXP": "Nominal", "PCAP": "Nominal", "VEXP": "Nominal", "LEXP": "Nominal", "MODP": "High", "TOOL": "High", "SCED": "Nominal"}},
    "Hotel Management System": {"project_name": "Hotel Management System", "description": "A medium-sized hotel management system with booking, customer management, billing, room service tracking, reports, and staff administration.", "mode": "Semi-detached", "kloc": 30.0, "cost_per_pm": 12000000.0, "drivers": {"RELY": "High", "DATA": "Nominal", "CPLX": "High", "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal", "ACAP": "Nominal", "AEXP": "Nominal", "PCAP": "Nominal", "VEXP": "Nominal", "LEXP": "Nominal", "MODP": "Nominal", "TOOL": "Nominal", "SCED": "High"}},
    "E-commerce Platform": {"project_name": "E-commerce Platform", "description": "An e-commerce platform with user accounts, product catalog, online payments, inventory tracking, order management, and admin analytics.", "mode": "Semi-detached", "kloc": 45.0, "cost_per_pm": 13000000.0, "drivers": {"RELY": "High", "DATA": "High", "CPLX": "High", "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal", "ACAP": "Nominal", "AEXP": "Low", "PCAP": "Nominal", "VEXP": "Nominal", "LEXP": "Nominal", "MODP": "Nominal", "TOOL": "High", "SCED": "High"}},
    "Medical Record System": {"project_name": "Medical Record System", "description": "A medical record management system for hospitals with patient histories, diagnostic records, laboratory integration, doctor access control, and strict reliability requirements.", "mode": "Embedded", "kloc": 55.0, "cost_per_pm": 15000000.0, "drivers": {"RELY": "Very High", "DATA": "High", "CPLX": "High", "TIME": "High", "STOR": "High", "VIRT": "Nominal", "TURN": "Nominal", "ACAP": "Nominal", "AEXP": "Low", "PCAP": "Nominal", "VEXP": "Nominal", "LEXP": "Nominal", "MODP": "High", "TOOL": "High", "SCED": "High"}},
}

FP_WEIGHTS = {
    "EI": {"Low": 3, "Average": 4, "High": 6},
    "EO": {"Low": 4, "Average": 5, "High": 7},
    "EQ": {"Low": 3, "Average": 4, "High": 6},
    "ILF": {"Low": 7, "Average": 10, "High": 15},
    "EIF": {"Low": 5, "Average": 7, "High": 10},
}

FP_COMPONENT_LABELS = {
    "EI": "External Inputs",
    "EO": "External Outputs",
    "EQ": "External Inquiries",
    "ILF": "Internal Logical Files",
    "EIF": "External Interface Files",
}

GSC_NAMES = [
    "Data communications", "Distributed data processing", "Performance", "Heavily used configuration",
    "Transaction rate", "Online data entry", "End-user efficiency", "Online update",
    "Complex processing", "Reusability", "Installation ease", "Operational ease",
    "Multiple sites", "Facilitate change",
]

LANGUAGE_LOC_PER_FP = {"Python": 50, "Java": 53, "C": 128, "C++": 53, "JavaScript": 47, "C#": 54, "PHP": 50, "Go": 55, "Ruby": 40}


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()


def generate_id(prefix="ID"):
    return f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(2).hex().upper()}"


def format_currency_short(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B VND"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M VND"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K VND"
    return f"{value:,.0f} VND"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def load_workspace():
    if os.path.exists(WORKSPACE_FILE):
        try:
            with open(WORKSPACE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"schema_version": APP_SCHEMA_VERSION, "projects": []}
    return {"schema_version": APP_SCHEMA_VERSION, "projects": []}


def persist_workspace(workspace: dict):
    with open(WORKSPACE_FILE, "w", encoding="utf-8") as f:
        json.dump(workspace, f, ensure_ascii=False, indent=2)


def refresh_workspace():
    st.session_state.workspace = load_workspace()


def init_state():
    defaults = {
        "selected_preset": "Start from Scratch",
        "preset_selector": "Start from Scratch",
        "project_name": "",
        "description": "",
        "mode": "Organic",
        "size_input_mode": "KLOC",
        "kloc": 1.0,
        "sloc": 1000.0,
        "cost_per_pm": 12000000.0,
        "ai_result": "",
        "ai_suggestion": None,
        "ai_fp_result": "",
        "pending_fp_transfer": False,
        "fp_transfer_kloc": 1.0,
        "pending_ai_apply": False,
        "fp_language": "Python",
        "fp_custom_loc_per_fp": float(LANGUAGE_LOC_PER_FP["Python"]),
        "fp_mode": "Organic",
        "fp_cost_per_pm": 12000000.0,
        "workspace": load_workspace(),
        "active_project_id": "",
        "active_version_id": "",
        "import_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for driver, meta in COST_DRIVERS.items():
        if driver not in st.session_state:
            values = list(meta["values"].keys())
            st.session_state[driver] = "Nominal" if "Nominal" in values else values[0]
    for component in FP_WEIGHTS:
        for level in ["Low", "Average", "High"]:
            fp_key = f"fp_{component}_{level.lower()}"
            if fp_key not in st.session_state:
                st.session_state[fp_key] = 0
    for idx in range(14):
        gsc_key = f"fp_gsc_{idx}"
        if gsc_key not in st.session_state:
            st.session_state[gsc_key] = 0


def reset_form():
    st.session_state.project_name = ""
    st.session_state.description = ""
    st.session_state.mode = "Organic"
    st.session_state.size_input_mode = "KLOC"
    st.session_state.kloc = 1.0
    st.session_state.sloc = 1000.0
    st.session_state.cost_per_pm = 12000000.0
    st.session_state.ai_result = ""
    st.session_state.ai_suggestion = None
    st.session_state.ai_fp_result = ""
    st.session_state.selected_preset = "Start from Scratch"
    st.session_state.active_version_id = ""
    for driver, meta in COST_DRIVERS.items():
        values = list(meta["values"].keys())
        st.session_state[driver] = "Nominal" if "Nominal" in values else values[0]
    st.session_state.fp_language = "Python"
    st.session_state.fp_custom_loc_per_fp = float(LANGUAGE_LOC_PER_FP["Python"])
    st.session_state.fp_mode = "Organic"
    st.session_state.fp_cost_per_pm = 12000000.0
    for component in FP_WEIGHTS:
        for level in ["Low", "Average", "High"]:
            st.session_state[f"fp_{component}_{level.lower()}"] = 0
    for idx in range(14):
        st.session_state[f"fp_gsc_{idx}"] = 0


def apply_preset(name: str):
    preset = PRESET_PROJECTS.get(name)
    if not preset:
        reset_form()
        st.session_state.selected_preset = "Start from Scratch"
        return
    st.session_state.project_name = preset["project_name"]
    st.session_state.description = preset["description"]
    st.session_state.mode = preset["mode"]
    st.session_state.size_input_mode = "KLOC"
    st.session_state.kloc = float(preset["kloc"])
    st.session_state.sloc = float(preset["kloc"]) * 1000.0
    st.session_state.cost_per_pm = float(preset["cost_per_pm"])
    st.session_state.ai_result = ""
    st.session_state.ai_suggestion = None
    st.session_state.ai_fp_result = ""
    st.session_state.selected_preset = name
    for driver, rating in preset["drivers"].items():
        st.session_state[driver] = rating


def process_pending_actions():
    if st.session_state.get("pending_fp_transfer"):
        kloc_value = max(float(st.session_state.get("fp_transfer_kloc", 1.0)), 0.001)
        st.session_state.selected_preset = "Start from Scratch"
        st.session_state.preset_selector = "Start from Scratch"
        st.session_state.size_input_mode = "KLOC"
        st.session_state.kloc = kloc_value
        st.session_state.sloc = kloc_value * 1000.0
        st.session_state.pending_fp_transfer = False
    if st.session_state.get("pending_ai_apply"):
        suggestion = st.session_state.get("ai_suggestion")
        if suggestion:
            suggested_mode = suggestion.get("suggested_mode")
            if suggested_mode in MODES:
                st.session_state.mode = suggested_mode
            for driver, rating in suggestion.get("suggested_changes", {}).items():
                if driver in COST_DRIVERS and rating in COST_DRIVERS[driver]["values"]:
                    st.session_state[driver] = rating
            st.session_state.selected_preset = "Start from Scratch"
            st.session_state.preset_selector = "Start from Scratch"
        st.session_state.pending_ai_apply = False


def get_driver_group(driver: str) -> str:
    for group, drivers in COST_DRIVER_GROUPS.items():
        if driver in drivers:
            return group
    return "Other"


def get_driver_selections():
    return {driver: st.session_state[driver] for driver in COST_DRIVERS}


def get_effective_kloc():
    if st.session_state.size_input_mode == "SLOC":
        return max(float(st.session_state.sloc), 1.0) / 1000.0
    return max(float(st.session_state.kloc), 0.001)


def calculate_eaf(selections: dict) -> float:
    eaf = 1.0
    for driver, rating in selections.items():
        eaf *= COST_DRIVERS[driver]["values"][rating]
    return eaf


def cocomo_estimate(mode: str, kloc: float, eaf: float):
    params = MODES[mode]
    effort = params["a"] * (kloc ** params["b"]) * eaf
    tdev = params["c"] * (effort ** params["d"])
    staff = effort / tdev if tdev > 0 else 0
    return effort, tdev, staff


def estimate_cost(effort: float, cost_per_pm: float) -> float:
    return effort * cost_per_pm


def suggest_mode_rule_based(description: str) -> str:
    text = (description or "").lower()
    if any(word in text for word in ["bank", "banking", "medical", "critical", "real-time", "iot", "embedded", "avionics", "telecom", "monitoring"]):
        return "Embedded"
    if any(word in text for word in ["hotel", "inventory", "management", "school", "e-commerce", "booking", "erp", "crm", "portal"]):
        return "Semi-detached"
    return "Organic"


def get_risk_level(eaf: float, selections: dict) -> str:
    high_risk_flags = 0
    if eaf >= 1.25:
        high_risk_flags += 1
    if selections["CPLX"] in ["High", "Very High", "Extra High"]:
        high_risk_flags += 1
    if selections["RELY"] in ["High", "Very High"]:
        high_risk_flags += 1
    if selections["SCED"] in ["High", "Very High"]:
        high_risk_flags += 1
    if selections["ACAP"] in ["Very Low", "Low"] or selections["PCAP"] in ["Very Low", "Low"]:
        high_risk_flags += 1
    if high_risk_flags >= 3:
        return "High"
    if high_risk_flags == 2:
        return "Medium"
    return "Low"


def compute_result(project_name: str, description: str, mode: str, kloc: float, cost_per_pm: float, selections: dict):
    eaf = calculate_eaf(selections)
    effort, tdev, staff = cocomo_estimate(mode, kloc, eaf)
    cost = estimate_cost(effort, cost_per_pm)
    risk_level = get_risk_level(eaf, selections)
    return {"project_name": project_name, "description": description, "mode": mode, "kloc": kloc, "cost_per_pm": cost_per_pm, "eaf": eaf, "effort": effort, "tdev": tdev, "staff": staff, "cost": cost, "risk_level": risk_level}


def build_driver_df(selections: dict) -> pd.DataFrame:
    rows = []
    for driver, rating in selections.items():
        multiplier = COST_DRIVERS[driver]["values"][rating]
        rows.append({"Driver": driver, "Name": COST_DRIVERS[driver]["label"], "Rating": rating, "Multiplier": multiplier, "Impact": abs(multiplier - 1.0), "Group": get_driver_group(driver)})
    return pd.DataFrame(rows)


def get_top_impact_drivers(driver_df: pd.DataFrame) -> pd.DataFrame:
    return driver_df.sort_values(by="Impact", ascending=False).head(5)



def get_inline_warnings(project_name, description, mode, kloc, cost_per_pm, selections, fp_snapshot):
    warnings = {"project_name": [], "description": [], "mode": [], "size": [], "cost": [], "drivers": [], "fp": []}
    suggested_mode = suggest_mode_rule_based(description)

    if not project_name.strip():
        warnings["project_name"].append("Project Name is empty.")
    if len((description or "").strip()) < 15:
        warnings["description"].append("Description is too short. AI and estimation may lack context.")
    if mode != suggested_mode and description.strip():
        warnings["mode"].append(f"Description currently suggests {suggested_mode} more than {mode}.")
    if kloc < 1:
        warnings["size"].append("KLOC is very small. Please verify the size unit.")
    if kloc > 1000:
        warnings["size"].append("KLOC is very large. Please verify KLOC/SLOC.")
    if cost_per_pm < 3000000:
        warnings["cost"].append("Cost per Person-Month is very low and may be unrealistic.")
    if selections["RELY"] in ["High", "Very High"] and mode == "Organic":
        warnings["mode"].append("High reliability with Organic mode may be inconsistent.")
    if selections["TIME"] in ["Very High", "Extra High"] and mode == "Organic":
        warnings["mode"].append("Very high time constraint with Organic mode may be inconsistent.")
    if selections["SCED"] in ["High", "Very High"] and selections["TOOL"] in ["Very Low", "Low", "Nominal"]:
        warnings["drivers"].append("Schedule pressure is high while TOOL support is still low.")
    if selections["ACAP"] in ["Very Low", "Low"] and selections["PCAP"] in ["Very Low", "Low"]:
        warnings["drivers"].append("Both analyst capability and programmer capability are low.")
    if selections["CPLX"] in ["Very High", "Extra High"] and kloc < 5:
        warnings["drivers"].append("Product complexity is very high while KLOC is still very small.")
    if fp_snapshot["fp"] > 0 and kloc > 0:
        diff_pct = abs(fp_snapshot["kloc"] - kloc) / kloc
        if diff_pct >= 0.5:
            warnings["fp"].append(f"Manual KLOC ({kloc:.2f}) differs a lot from FP KLOC ({fp_snapshot['kloc']:.2f}).")
    return warnings


def flatten_warnings(warnings: dict):
    all_msgs = []
    for msgs in warnings.values():
        all_msgs.extend(msgs)
    return all_msgs


def fallback_ai_summary(description, mode, kloc, eaf, effort, tdev, staff, cost, selections):
    risks = []
    recommendations = []

    if selections["CPLX"] in ["High", "Very High", "Extra High"]:
        risks.append("Product complexity is high, so effort increases significantly.")
        recommendations.append("Split the system into smaller modules or phases.")
    if selections["RELY"] in ["High", "Very High"]:
        risks.append("High reliability requirements increase testing and verification effort.")
        recommendations.append("Allocate more time to testing, review, QA, and security.")
    if selections["ACAP"] in ["Very Low", "Low"] or selections["PCAP"] in ["Very Low", "Low"]:
        risks.append("Current team capability may reduce actual productivity.")
        recommendations.append("Increase mentoring or assign stronger members to core modules.")
    if selections["TOOL"] in ["Very Low", "Low"]:
        risks.append("Weak tool support can slow down development.")
        recommendations.append("Improve IDE, test tools, workflow, and automation.")
    if selections["SCED"] in ["High", "Very High"]:
        risks.append("Aggressive schedule increases delivery and quality risk.")
        recommendations.append("Reduce scope or relax the schedule.")
    if not risks:
        risks.append("No major risk stands out from the current cost driver configuration.")
    if not recommendations:
        recommendations.append("The current setup is relatively balanced.")

    return f"""
### AI Project Analysis

**1. Mode assessment**
- Current mode: **{mode}**
- Suggested mode from description: **{suggest_mode_rule_based(description)}**
- The current mode should be reviewed if it differs from the project description.

**2. Why effort and cost are at this level**
- KLOC: **{kloc:.2f}**
- EAF: **{eaf:.3f}**
- Effort: **{effort:.2f} person-months**
- Development time: **{tdev:.2f} months**
- Team size: **{staff:.2f}**
- Estimated cost: **{cost:,.0f} VND**

**3. Main risks**
{chr(10).join([f"- {r}" for r in risks])}

**4. Practical recommendations**
{chr(10).join([f"- {r}" for r in recommendations])}

**5. Conclusion**
- The project is feasible, but cost and effort can still be optimized.
- The final result strongly depends on the selected mode, cost drivers, and the realism of the KLOC estimate.
""".strip()


def fallback_fp_ai_summary(fp_data: dict, fp_based_result: dict):
    fp_value = fp_data["fp"]
    if fp_value < 50:
        fp_scale = "small"
    elif fp_value < 200:
        fp_scale = "medium"
    else:
        fp_scale = "large"

    return f"""
### AI Analysis for Function Point

**1. FP size assessment**
- FP size is currently **{fp_scale}** with **{fp_value:.2f} FP**.

**2. Meaning of VAF**
- DI = **{fp_data["di"]}**
- VAF = **{fp_data["vaf"]:.2f}**
- A higher VAF means general system characteristics have stronger impact on functional size.

**3. Conversion meaning**
- Language: **{fp_data["language"]}**
- LOC per FP: **{fp_data["loc_per_fp"]:.2f}**
- SLOC: **{fp_data["sloc"]:,.0f}**
- KLOC: **{fp_data["kloc"]:.2f}**
- This KLOC can be compared with manual KLOC to validate scope and size.

**4. COCOMO from FP**
- Effort: **{fp_based_result["effort"]:.2f} PM**
- Time: **{fp_based_result["tdev"]:.2f} months**
- Team Size: **{fp_based_result["staff"]:.2f}**
- Cost: **{fp_based_result["cost"]:,.0f} VND**
""".strip()


def call_ai_analysis(description, mode, kloc, eaf, effort, tdev, staff, cost, selections):
    api_key = get_openai_api_key()
    if not api_key or OpenAI is None:
        return None, "AI unavailable"
    try:
        client = OpenAI(api_key=api_key)
        driver_text = "\n".join([f"{k}: {v} (multiplier={COST_DRIVERS[k]['values'][v]})" for k, v in selections.items()])
        prompt = f"""
Bạn là chuyên gia ước lượng chi phí phần mềm.

Phân tích dự án dưới đây bằng tiếng Việt, rõ ràng, chi tiết vừa phải, chia đúng 5 mục:

1. Đánh giá development mode hiện tại có phù hợp không
2. Giải thích vì sao effort / cost ở mức này
3. Liệt kê 3 rủi ro lớn nhất
4. Đưa ra 3 khuyến nghị cụ thể để tối ưu effort / cost / risk
5. Kết luận ngắn cho project manager hoặc giảng viên

Dữ liệu:
Project description:
{description}

Mode: {mode}
KLOC: {kloc:.2f}
EAF: {eaf:.3f}
Effort: {effort:.2f} person-months
Development time: {tdev:.2f} months
Average team size: {staff:.2f}
Estimated cost: {cost:,.0f} VND

Cost drivers:
{driver_text}

Viết bằng tiếng Việt, dễ hiểu, có tiêu đề cho từng mục, không lan man.
"""
        response = client.responses.create(model=AI_MODEL_NAME, input=prompt)
        text = response.output_text
        if text and text.strip():
            return text, None
        return None, "AI unavailable"
    except Exception:
        return None, "AI unavailable"


def call_ai_fp_analysis(fp_data: dict, fp_based_result: dict):
    api_key = get_openai_api_key()
    if not api_key or OpenAI is None:
        return None, "AI unavailable"
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
Bạn là trợ lý phân tích Function Point.

Hãy phân tích bằng tiếng Việt với 4 mục:
1. Nhận xét quy mô FP
2. Ý nghĩa của VAF
3. Ý nghĩa của KLOC quy đổi từ FP
4. Kết luận ngắn

Dữ liệu FP:
- UFP: {fp_data["ufp"]:.2f}
- DI: {fp_data["di"]}
- VAF: {fp_data["vaf"]:.2f}
- FP: {fp_data["fp"]:.2f}
- Language: {fp_data["language"]}
- LOC per FP: {fp_data["loc_per_fp"]:.2f}
- SLOC: {fp_data["sloc"]:,.0f}
- KLOC: {fp_data["kloc"]:.2f}

Kết quả COCOMO từ FP:
- Mode: {fp_based_result["mode"]}
- Effort: {fp_based_result["effort"]:.2f}
- Time: {fp_based_result["tdev"]:.2f}
- Team Size: {fp_based_result["staff"]:.2f}
- Cost: {fp_based_result["cost"]:,.0f} VND
"""
        response = client.responses.create(model=AI_MODEL_NAME, input=prompt)
        text = response.output_text
        if text and text.strip():
            return text, None
        return None, "AI unavailable"
    except Exception:
        return None, "AI unavailable"


def extract_json_object(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def sanitize_ai_suggestion(data: dict, current_mode: str, current_selections: dict):
    suggestion = {"suggested_mode": current_mode, "suggested_changes": {}, "goals": [], "reasoning": [], "expected_effect": ""}
    if not isinstance(data, dict):
        return suggestion

    mode = data.get("suggested_mode", current_mode)
    if mode in MODES:
        suggestion["suggested_mode"] = mode

    changes = data.get("suggested_changes", {})
    if isinstance(changes, dict):
        for driver, rating in changes.items():
            if driver in COST_DRIVERS and rating in COST_DRIVERS[driver]["values"] and rating != current_selections[driver]:
                suggestion["suggested_changes"][driver] = rating

    goals = data.get("goals", [])
    if isinstance(goals, list):
        suggestion["goals"] = [str(x) for x in goals][:5]

    reasoning = data.get("reasoning", [])
    if isinstance(reasoning, list):
        suggestion["reasoning"] = [str(x) for x in reasoning][:6]

    suggestion["expected_effect"] = str(data.get("expected_effect", ""))
    return suggestion


def fallback_ai_optimization(current_mode: str, selections: dict):
    changes = {}
    if selections["TOOL"] in ["Very Low", "Low", "Nominal"]:
        changes["TOOL"] = "High"
    if selections["MODP"] in ["Very Low", "Low", "Nominal"]:
        changes["MODP"] = "High"
    if selections["SCED"] in ["High", "Very High"]:
        changes["SCED"] = "Nominal"
    if selections["ACAP"] in ["Very Low", "Low"]:
        changes["ACAP"] = "High"
    if selections["PCAP"] in ["Very Low", "Low"]:
        changes["PCAP"] = "High"
    if not changes and selections["CPLX"] in ["Very High", "Extra High"]:
        changes["CPLX"] = "High"

    return {
        "suggested_mode": current_mode,
        "suggested_changes": changes,
        "goals": ["Giảm effort", "Giảm rủi ro"],
        "reasoning": [
            "Better tool support can reduce development effort.",
            "Modern practices can improve productivity.",
            "Reducing schedule pressure lowers delivery risk.",
        ],
        "expected_effect": "The suggested setup may reduce effort, cost, and overall risk.",
    }



def call_ai_optimization(project_name: str, description: str, current_result: dict, selections: dict):
    api_key = get_openai_api_key()
    fallback = fallback_ai_optimization(current_result["mode"], selections)

    if not api_key or OpenAI is None:
        return fallback, "AI unavailable"

    try:
        client = OpenAI(api_key=api_key)
        driver_text = "\n".join([f"{k}: {v} (multiplier={COST_DRIVERS[k]['values'][v]})" for k, v in selections.items()])

        prompt = f"""
Bạn là trợ lý tối ưu estimate dự án phần mềm.

Mục tiêu:
- giảm effort
- giảm cost
- giảm risk
- giữ đề xuất thực tế

Dự án hiện tại:
Project name: {project_name}
Description: {description}
Mode: {current_result["mode"]}
KLOC: {current_result["kloc"]:.2f}
EAF: {current_result["eaf"]:.3f}
Effort: {current_result["effort"]:.2f}
Development time: {current_result["tdev"]:.2f}
Average team size: {current_result["staff"]:.2f}
Estimated cost: {current_result["cost"]:,.0f} VND
Risk level: {current_result["risk_level"]}

Current cost drivers:
{driver_text}

Hãy trả về JSON hợp lệ, không markdown, chỉ thay đổi tối đa 5 cost drivers, không đổi KLOC.

JSON format:
{{
  "suggested_mode": "Organic or Semi-detached or Embedded",
  "suggested_changes": {{
    "TOOL": "High"
  }},
  "goals": ["Giảm effort", "Giảm rủi ro"],
  "reasoning": ["..."],
  "expected_effect": "..."
}}
"""
        response = client.responses.create(model=AI_MODEL_NAME, input=prompt)
        parsed = extract_json_object(response.output_text)
        if parsed is None:
            return fallback, "Invalid AI JSON"
        suggestion = sanitize_ai_suggestion(parsed, current_result["mode"], selections)
        if not suggestion["suggested_changes"] and suggestion["suggested_mode"] == current_result["mode"]:
            return fallback, "Weak AI suggestion"
        return suggestion, None
    except Exception:
        return fallback, "AI unavailable"


def apply_suggestion_to_result(current_result: dict, current_selections: dict, suggestion: dict):
    suggested_mode = suggestion.get("suggested_mode", current_result["mode"])
    suggested_selections = current_selections.copy()
    for driver, rating in suggestion.get("suggested_changes", {}).items():
        if driver in COST_DRIVERS and rating in COST_DRIVERS[driver]["values"]:
            suggested_selections[driver] = rating
    suggested_result = compute_result(
        project_name=current_result["project_name"],
        description=current_result["description"],
        mode=suggested_mode,
        kloc=current_result["kloc"],
        cost_per_pm=current_result["cost_per_pm"],
        selections=suggested_selections,
    )
    return suggested_selections, suggested_result


def build_metric_compare_df(current_result: dict, suggested_result: dict):
    return pd.DataFrame([
        {"Metric": "EAF", "Current": round(current_result["eaf"], 3), "Suggested": round(suggested_result["eaf"], 3), "Delta": round(suggested_result["eaf"] - current_result["eaf"], 3)},
        {"Metric": "Effort (PM)", "Current": round(current_result["effort"], 2), "Suggested": round(suggested_result["effort"], 2), "Delta": round(suggested_result["effort"] - current_result["effort"], 2)},
        {"Metric": "Time (Months)", "Current": round(current_result["tdev"], 2), "Suggested": round(suggested_result["tdev"], 2), "Delta": round(suggested_result["tdev"] - current_result["tdev"], 2)},
        {"Metric": "Team Size", "Current": round(current_result["staff"], 2), "Suggested": round(suggested_result["staff"], 2), "Delta": round(suggested_result["staff"] - current_result["staff"], 2)},
        {"Metric": "Estimated Cost (VND)", "Current": round(current_result["cost"], 0), "Suggested": round(suggested_result["cost"], 0), "Delta": round(suggested_result["cost"] - current_result["cost"], 0)},
    ])


def build_driver_change_df(current_selections: dict, suggested_selections: dict):
    rows = []
    for driver in COST_DRIVERS:
        current_rating = current_selections[driver]
        suggested_rating = suggested_selections[driver]
        if current_rating != suggested_rating:
            rows.append({
                "Driver": driver,
                "Current": current_rating,
                "Suggested": suggested_rating,
                "Current Multiplier": COST_DRIVERS[driver]["values"][current_rating],
                "Suggested Multiplier": COST_DRIVERS[driver]["values"][suggested_rating],
            })
    return pd.DataFrame(rows)


def get_fp_counts():
    counts = {}
    for component in FP_WEIGHTS:
        counts[component] = {}
        for level in ["Low", "Average", "High"]:
            counts[component][level] = int(st.session_state[f"fp_{component}_{level.lower()}"])
    return counts


def calculate_ufp(counts: dict) -> int:
    total = 0
    for component, level_counts in counts.items():
        for level, count in level_counts.items():
            total += count * FP_WEIGHTS[component][level]
    return total


def calculate_vaf():
    di = sum(int(st.session_state[f"fp_gsc_{idx}"]) for idx in range(14))
    vaf = 0.65 + (0.01 * di)
    return di, vaf


def get_fp_loc_per_fp():
    if st.session_state.fp_language == "Custom":
        return max(float(st.session_state.fp_custom_loc_per_fp), 1.0)
    return float(LANGUAGE_LOC_PER_FP[st.session_state.fp_language])


def get_fp_snapshot():
    counts = get_fp_counts()
    ufp = calculate_ufp(counts)
    di, vaf = calculate_vaf()
    fp_value = ufp * vaf
    loc_per_fp = get_fp_loc_per_fp()
    sloc = fp_value * loc_per_fp
    kloc = sloc / 1000.0
    gsc = {GSC_NAMES[idx]: int(st.session_state[f"fp_gsc_{idx}"]) for idx in range(14)}

    return {
        "language": st.session_state.fp_language,
        "loc_per_fp": loc_per_fp,
        "counts": counts,
        "gsc": gsc,
        "ufp": ufp,
        "di": di,
        "vaf": vaf,
        "fp": fp_value,
        "sloc": sloc,
        "kloc": kloc,
        "fp_mode": st.session_state.fp_mode,
        "fp_cost_per_pm": st.session_state.fp_cost_per_pm,
        "ai_fp_result": st.session_state.ai_fp_result,
    }


def save_history_row(data: dict):
    row_df = pd.DataFrame([data])
    if os.path.exists(HISTORY_FILE):
        old_df = pd.read_csv(HISTORY_FILE)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
        new_df.to_csv(HISTORY_FILE, index=False)
    else:
        row_df.to_csv(HISTORY_FILE, index=False)


def load_history_df() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()


def export_text_report(result: dict, selections: dict, ai_text: str, fp_data: dict, warnings: list):
    lines = [
        "PROFESSIONAL SOFTWARE ESTIMATION REPORT",
        "=" * 50,
        f"Timestamp: {now_text()}",
        f"Project Name: {result['project_name']}",
        f"Description: {result['description']}",
        f"Mode: {result['mode']}",
        f"KLOC: {result['kloc']:.2f}",
        f"Cost per PM: {result['cost_per_pm']:,.0f} VND",
        "",
        "RESULTS",
        "-" * 20,
        f"EAF: {result['eaf']:.3f}",
        f"Effort: {result['effort']:.2f} person-months",
        f"Development Time: {result['tdev']:.2f} months",
        f"Average Team Size: {result['staff']:.2f}",
        f"Estimated Cost: {result['cost']:,.0f} VND",
        f"Risk Level: {result['risk_level']}",
        "",
        "INLINE WARNINGS",
        "-" * 20,
    ]
    if warnings:
        for msg in warnings:
            lines.append(f"- {msg}")
    else:
        lines.append("No warning.")
    lines.extend(["", "COST DRIVERS", "-" * 20])
    for driver, rating in selections.items():
        multiplier = COST_DRIVERS[driver]["values"][rating]
        lines.append(f"{driver} - {rating} (multiplier={multiplier})")
    lines.extend([
        "",
        "FUNCTION POINT",
        "-" * 20,
        f"Language: {fp_data['language']}",
        f"UFP: {fp_data['ufp']:.2f}",
        f"DI: {fp_data['di']}",
        f"VAF: {fp_data['vaf']:.2f}",
        f"FP: {fp_data['fp']:.2f}",
        f"SLOC: {fp_data['sloc']:,.0f}",
        f"KLOC: {fp_data['kloc']:.2f}",
        "",
        "AI ANALYSIS",
        "-" * 20,
        ai_text if ai_text else "No AI analysis.",
    ])
    return "\n".join(lines)



def make_bar_drawing(title: str, categories, values, width=460, height=220):
    drawing = Drawing(width, height)
    drawing.add(String(10, height - 18, title, fontSize=12))
    chart = VerticalBarChart()
    chart.x = 40
    chart.y = 35
    chart.height = height - 70
    chart.width = width - 70
    chart.data = [tuple(values)]
    chart.categoryAxis.categoryNames = [str(x) for x in categories]
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = colors.HexColor("#4e79a7")
    chart.strokeColor = colors.black
    drawing.add(chart)
    return drawing


def make_pie_drawing(title: str, labels, values, width=460, height=240):
    drawing = Drawing(width, height)
    drawing.add(String(10, height - 18, title, fontSize=12))
    pie = Pie()
    pie.x = 80
    pie.y = 30
    pie.width = 140
    pie.height = 140
    pie.data = [max(float(v), 0.0) for v in values]
    pie.labels = [str(x) for x in labels]
    pie.slices.strokeWidth = 0.5
    drawing.add(pie)
    return drawing


def make_pdf_report_bytes(project_name: str, result: dict, selections: dict, fp_snapshot: dict, warnings: list, ai_text: str):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=28, rightMargin=28, topMargin=26, bottomMargin=26)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22)
    heading_style = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], alignment=TA_LEFT, fontSize=12, textColor=colors.HexColor("#0d47a1"))
    normal_style = ParagraphStyle("NormalCustom", parent=styles["Normal"], fontSize=9, leading=13)

    story = []
    story.append(Paragraph("Professional Software Estimation Report", title_style))
    story.append(Spacer(1, 10))

    summary_data = [
        ["Generated At", now_text()],
        ["Project Name", project_name or "-"],
        ["Mode", result["mode"]],
        ["KLOC", f"{result['kloc']:.2f}"],
        ["EAF", f"{result['eaf']:.3f}"],
        ["Effort", f"{result['effort']:.2f} PM"],
        ["Time", f"{result['tdev']:.2f} months"],
        ["Team Size", f"{result['staff']:.2f}"],
        ["Cost", f"{result['cost']:,.0f} VND"],
        ["Risk", result["risk_level"]],
    ]
    story.append(Paragraph("1. Executive Summary", heading_style))
    summary_table = Table(summary_data, colWidths=[120, 360])
    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e3f2fd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Charts", heading_style))
    driver_df = build_driver_df(selections)
    top5_df = get_top_impact_drivers(driver_df)
    story.append(make_bar_drawing("Top 5 Driver Multipliers", list(top5_df["Driver"]), list(top5_df["Multiplier"])))
    story.append(Spacer(1, 8))

    cost_breakdown_labels = ["Development", "Testing & QA", "Management", "Tools & Infrastructure"]
    cost_breakdown_values = [result["cost"] * 0.55, result["cost"] * 0.20, result["cost"] * 0.15, result["cost"] * 0.10]
    story.append(make_pie_drawing("Estimated Cost Breakdown", cost_breakdown_labels, cost_breakdown_values))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Inline Warnings", heading_style))
    if warnings:
        for msg in warnings:
            story.append(Paragraph(f"- {msg}", normal_style))
    else:
        story.append(Paragraph("No inline warning.", normal_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Cost Drivers", heading_style))
    driver_rows = [["Driver", "Rating", "Multiplier"]]
    for driver, rating in selections.items():
        driver_rows.append([driver, rating, str(COST_DRIVERS[driver]["values"][rating])])
    driver_table = Table(driver_rows, colWidths=[80, 200, 100])
    driver_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#bbdefb")),
    ]))
    story.append(driver_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Function Point Summary", heading_style))
    fp_rows = [
        ["Language", fp_snapshot["language"]],
        ["LOC per FP", f"{fp_snapshot['loc_per_fp']:.2f}"],
        ["UFP", f"{fp_snapshot['ufp']:.2f}"],
        ["DI", f"{fp_snapshot['di']}"],
        ["VAF", f"{fp_snapshot['vaf']:.2f}"],
        ["FP", f"{fp_snapshot['fp']:.2f}"],
        ["SLOC", f"{fp_snapshot['sloc']:,.0f}"],
        ["KLOC", f"{fp_snapshot['kloc']:.2f}"],
    ]
    fp_table = Table(fp_rows, colWidths=[120, 180])
    fp_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
    ]))
    story.append(fp_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("6. AI Project Analysis", heading_style))
    ai_render = (ai_text or "No AI analysis.").replace("\n", "<br/>")
    story.append(Paragraph(ai_render, normal_style))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf



def find_project_by_id(project_id: str):
    for project in st.session_state.workspace.get("projects", []):
        if project["project_id"] == project_id:
            return project
    return None


def find_version_by_id(project: dict, version_id: str):
    for version in project.get("versions", []):
        if version["version_id"] == version_id:
            return version
    return None


def get_next_version_no(project: dict):
    versions = project.get("versions", [])
    if not versions:
        return 1
    return max(v.get("version_no", 0) for v in versions) + 1


def build_current_version_payload(result: dict, selections: dict, fp_snapshot: dict, warnings: list):
    return {
        "version_id": st.session_state.active_version_id or generate_id("VER"),
        "version_no": 0,
        "title": "Auto Save",
        "note": "",
        "created_at": now_text(),
        "updated_at": now_text(),
        "is_baseline": False,
        "project_meta": {
            "project_name": st.session_state.project_name,
            "description": st.session_state.description,
        },
        "input": {
            "mode": st.session_state.mode,
            "size_input_mode": st.session_state.size_input_mode,
            "kloc": result["kloc"],
            "sloc": result["kloc"] * 1000.0,
            "cost_per_pm": st.session_state.cost_per_pm,
        },
        "cost_drivers": deepcopy(selections),
        "cost_driver_multipliers": {driver: COST_DRIVERS[driver]["values"][rating] for driver, rating in selections.items()},
        "result": deepcopy(result),
        "inline_warnings": deepcopy(warnings),
        "ai": {"analysis": st.session_state.ai_result, "suggestion": deepcopy(st.session_state.ai_suggestion)},
        "fp": deepcopy(fp_snapshot),
    }


def create_new_project_with_version(version_payload: dict):
    project_id = generate_id("PRJ")
    project = {
        "project_id": project_id,
        "project_name": version_payload["project_meta"]["project_name"] or "Untitled Project",
        "description": version_payload["project_meta"]["description"],
        "created_at": now_text(),
        "updated_at": now_text(),
        "versions": [],
    }
    version_payload["version_no"] = 1
    version_payload["is_baseline"] = True
    project["versions"].append(version_payload)
    st.session_state.workspace["projects"].append(project)
    persist_workspace(st.session_state.workspace)
    st.session_state.active_project_id = project_id
    st.session_state.active_version_id = version_payload["version_id"]


def save_current_as_new_version(version_payload: dict):
    project = find_project_by_id(st.session_state.active_project_id)
    if not project:
        create_new_project_with_version(version_payload)
        return
    version_payload["version_id"] = generate_id("VER")
    version_payload["version_no"] = get_next_version_no(project)
    version_payload["is_baseline"] = False
    project["project_name"] = version_payload["project_meta"]["project_name"] or project["project_name"]
    project["description"] = version_payload["project_meta"]["description"]
    project["updated_at"] = now_text()
    project["versions"].append(version_payload)
    persist_workspace(st.session_state.workspace)
    st.session_state.active_version_id = version_payload["version_id"]


def set_baseline(project_id: str, version_id: str):
    project = find_project_by_id(project_id)
    if not project:
        return False
    for version in project.get("versions", []):
        version["is_baseline"] = version["version_id"] == version_id
        version["updated_at"] = now_text()
    project["updated_at"] = now_text()
    persist_workspace(st.session_state.workspace)
    return True


def get_baseline_version(project: dict):
    for version in project.get("versions", []):
        if version.get("is_baseline"):
            return version
    return None


def load_version_to_form(project_id: str, version_id: str):
    project = find_project_by_id(project_id)
    if not project:
        return False
    version = find_version_by_id(project, version_id)
    if not version:
        return False

    st.session_state.active_project_id = project_id
    st.session_state.active_version_id = version_id
    st.session_state.project_name = version["project_meta"].get("project_name", "")
    st.session_state.description = version["project_meta"].get("description", "")
    st.session_state.mode = version["input"].get("mode", "Organic")
    st.session_state.size_input_mode = version["input"].get("size_input_mode", "KLOC")
    st.session_state.kloc = safe_float(version["input"].get("kloc", 1.0), 1.0)
    st.session_state.sloc = safe_float(version["input"].get("sloc", st.session_state.kloc * 1000), 1000.0)
    st.session_state.cost_per_pm = safe_float(version["input"].get("cost_per_pm", 12000000), 12000000.0)
    st.session_state.ai_result = version.get("ai", {}).get("analysis", "")
    st.session_state.ai_suggestion = version.get("ai", {}).get("suggestion", None)

    saved_drivers = version.get("cost_drivers", {})
    for driver in COST_DRIVERS:
        if driver in saved_drivers and saved_drivers[driver] in COST_DRIVERS[driver]["values"]:
            st.session_state[driver] = saved_drivers[driver]

    fp_data = version.get("fp", {})
    st.session_state.fp_language = fp_data.get("language", "Python")
    st.session_state.fp_custom_loc_per_fp = safe_float(fp_data.get("loc_per_fp", LANGUAGE_LOC_PER_FP["Python"]), LANGUAGE_LOC_PER_FP["Python"])
    st.session_state.fp_mode = fp_data.get("fp_mode", "Organic")
    st.session_state.fp_cost_per_pm = safe_float(fp_data.get("fp_cost_per_pm", 12000000.0), 12000000.0)
    st.session_state.ai_fp_result = fp_data.get("ai_fp_result", "")

    counts = fp_data.get("counts", {})
    for component in FP_WEIGHTS:
        levels = counts.get(component, {})
        st.session_state[f"fp_{component}_low"] = safe_int(levels.get("Low", 0), 0)
        st.session_state[f"fp_{component}_average"] = safe_int(levels.get("Average", 0), 0)
        st.session_state[f"fp_{component}_high"] = safe_int(levels.get("High", 0), 0)

    gsc = fp_data.get("gsc", {})
    for idx, name in enumerate(GSC_NAMES):
        st.session_state[f"fp_gsc_{idx}"] = safe_int(gsc.get(name, 0), 0)
    return True


def build_project_package(project_id: str, version_id: str):
    project = find_project_by_id(project_id)
    if not project:
        return None
    version = find_version_by_id(project, version_id)
    if not version:
        return None
    return {
        "schema_version": APP_SCHEMA_VERSION,
        "exported_at": now_text(),
        "project": {
            "project_id": project["project_id"],
            "project_name": project["project_name"],
            "description": project["description"],
            "created_at": project["created_at"],
            "updated_at": project["updated_at"],
        },
        "version": deepcopy(version),
    }


def import_project_package(uploaded_file):
    try:
        data = json.load(uploaded_file)
    except Exception:
        return False, "File JSON không hợp lệ."
    if not isinstance(data, dict):
        return False, "Cấu trúc JSON không đúng."
    if "project" not in data or "version" not in data:
        return False, "JSON thiếu project hoặc version."

    project_block = data["project"]
    version_block = data["version"]
    new_project_id = generate_id("PRJ")
    new_version_id = generate_id("VER")

    project = {
        "project_id": new_project_id,
        "project_name": project_block.get("project_name", "Imported Project"),
        "description": project_block.get("description", ""),
        "created_at": now_text(),
        "updated_at": now_text(),
        "versions": [],
    }

    version_copy = deepcopy(version_block)
    version_copy["version_id"] = new_version_id
    version_copy["version_no"] = 1
    version_copy["is_baseline"] = True
    version_copy["created_at"] = now_text()
    version_copy["updated_at"] = now_text()

    project["versions"].append(version_copy)
    st.session_state.workspace["projects"].append(project)
    persist_workspace(st.session_state.workspace)

    st.session_state.active_project_id = new_project_id
    st.session_state.active_version_id = new_version_id
    load_version_to_form(new_project_id, new_version_id)
    return True, f"Import thành công project: {project['project_name']}"


def project_versions_df(project: dict):
    rows = []
    for version in project.get("versions", []):
        result = version.get("result", {})
        rows.append({
            "Version No": version.get("version_no", 0),
            "Baseline": "Yes" if version.get("is_baseline") else "No",
            "Mode": result.get("mode", ""),
            "KLOC": round(safe_float(result.get("kloc", 0)), 2),
            "Effort": round(safe_float(result.get("effort", 0)), 2),
            "Time": round(safe_float(result.get("tdev", 0)), 2),
            "Cost": round(safe_float(result.get("cost", 0)), 0),
            "Risk": result.get("risk_level", ""),
            "Updated At": version.get("updated_at", ""),
            "Version ID": version.get("version_id", ""),
        })
    return pd.DataFrame(rows)


def build_baseline_compare_df(current_result: dict, baseline_result: dict):
    return pd.DataFrame([
        {"Metric": "EAF", "Baseline": round(baseline_result["eaf"], 3), "Current": round(current_result["eaf"], 3), "Delta": round(current_result["eaf"] - baseline_result["eaf"], 3)},
        {"Metric": "Effort (PM)", "Baseline": round(baseline_result["effort"], 2), "Current": round(current_result["effort"], 2), "Delta": round(current_result["effort"] - baseline_result["effort"], 2)},
        {"Metric": "Time (Months)", "Baseline": round(baseline_result["tdev"], 2), "Current": round(current_result["tdev"], 2), "Delta": round(current_result["tdev"] - baseline_result["tdev"], 2)},
        {"Metric": "Team Size", "Baseline": round(baseline_result["staff"], 2), "Current": round(current_result["staff"], 2), "Delta": round(current_result["staff"] - baseline_result["staff"], 2)},
        {"Metric": "Cost (VND)", "Baseline": round(baseline_result["cost"], 0), "Current": round(current_result["cost"], 0), "Delta": round(current_result["cost"] - baseline_result["cost"], 0)},
    ])


def feature_review_notes():
    notes = []
    notes.append("Removed unnecessary fields: Project Status, Version Title, Version Note.")
    notes.append("Validation tab removed. Warnings now appear directly near related inputs.")
    notes.append("PDF report is generated in English to avoid Vietnamese font issues.")
    notes.append("Help tab has been filled with formulas and workflow guidance.")
    notes.append("Workspace is still available for save/load/export/import. Keep it only if your teacher wants versioning.")
    return notes


init_state()
process_pending_actions()

with st.sidebar:
    st.title("Professional Estimation Workspace")
    st.markdown("Estimate project effort, schedule, team size, cost, risk, version history, baseline, JSON package, PDF report and AI analysis.")

    preset = st.selectbox(
        "Project Template",
        list(PRESET_PROJECTS.keys()),
        index=list(PRESET_PROJECTS.keys()).index(st.session_state.selected_preset)
        if st.session_state.selected_preset in PRESET_PROJECTS else 0,
        key="preset_selector",
    )

    csa, csb = st.columns(2)
    with csa:
        if st.button("Apply Template", use_container_width=True):
            apply_preset(preset)
            st.rerun()
    with csb:
        if st.button("Reset Form", use_container_width=True):
            reset_form()
            st.rerun()

    enable_ai = st.checkbox("Enable AI", value=True)

    st.markdown("---")
    st.subheader("Workspace")
    workspace_projects = st.session_state.workspace.get("projects", [])
    project_options = ["-- No project selected --"] + [f"{p['project_name']} ({p['project_id']})" for p in workspace_projects]
    selected_project_label = st.selectbox("Saved projects", project_options)

    if selected_project_label != "-- No project selected --":
        selected_project_id = selected_project_label.split("(")[-1].replace(")", "").strip()
        project_obj = find_project_by_id(selected_project_id)
        if project_obj:
            version_options = [f"v{v['version_no']} ({v['version_id']})" for v in project_obj.get("versions", [])]
            if version_options:
                selected_version_label = st.selectbox("Saved versions", version_options)
                selected_version_id = selected_version_label.split("(")[-1].replace(")", "").strip()
                if st.button("Load Version to Form", use_container_width=True):
                    ok = load_version_to_form(selected_project_id, selected_version_id)
                    if ok:
                        st.success("Version loaded.")
                        st.rerun()

    st.markdown("---")
    st.caption("Project package JSON stores full project + version + drivers + FP + AI.")

st.title("Professional Software Effort Estimation Workspace")
st.caption("Intermediate COCOMO + Function Point + AI Analysis + Versioning + JSON Package + PDF Report")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Estimator",
    "FP Estimator",
    "AI Optimization",
    "Workspace",
    "Compare & What-if",
    "Help",
])

with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Project Information")
        st.text_input("Project Name", key="project_name")
        if not st.session_state.project_name.strip():
            st.warning("Please enter Project Name.")
        st.text_area("Project Description", height=140, key="description")
        st.selectbox("Development Mode", list(MODES.keys()), key="mode")
        st.selectbox("Size Input Unit", ["KLOC", "SLOC"], key="size_input_mode")
        if st.session_state.size_input_mode == "KLOC":
            st.number_input("Estimated Size (KLOC)", min_value=0.001, step=1.0, key="kloc")
        else:
            st.number_input("Estimated Size (SLOC)", min_value=1.0, step=100.0, key="sloc")

        if len((st.session_state.description or "").strip()) < 15:
            st.warning("Project Description is too short.")
        suggested_mode = suggest_mode_rule_based(st.session_state.description)
        if st.session_state.description.strip() and suggested_mode != st.session_state.mode:
            st.warning(f"Description currently suggests {suggested_mode} more than {st.session_state.mode}.")

        effective_cost_input = st.number_input("Cost per Person-Month (VND)", min_value=1000000.0, step=1000000.0, key="cost_per_pm")
        if effective_cost_input < 3000000:
            st.warning("Cost per Person-Month looks quite low.")

    with right:
        st.subheader("Cost Drivers")
        for group_name, drivers in COST_DRIVER_GROUPS.items():
            with st.expander(group_name, expanded=True):
                for driver in drivers:
                    meta = COST_DRIVERS[driver]
                    options = list(meta["values"].keys())
                    st.selectbox(f"{driver} - {meta['label']}", options, key=driver, help=meta["help"])

    selections = get_driver_selections()
    effective_kloc = get_effective_kloc()
    result = compute_result(
        project_name=st.session_state.project_name,
        description=st.session_state.description,
        mode=st.session_state.mode,
        kloc=effective_kloc,
        cost_per_pm=st.session_state.cost_per_pm,
        selections=selections,
    )
    fp_snapshot = get_fp_snapshot()
    inline_warnings = get_inline_warnings(
        st.session_state.project_name,
        st.session_state.description,
        st.session_state.mode,
        effective_kloc,
        st.session_state.cost_per_pm,
        selections,
        fp_snapshot,
    )
    all_warning_messages = flatten_warnings(inline_warnings)

    if inline_warnings["drivers"]:
        for msg in inline_warnings["drivers"]:
            st.warning(msg)

    st.subheader("Estimation Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Effort (PM)", f"{result['effort']:.2f}")
    m2.metric("Time (Months)", f"{result['tdev']:.2f}")
    m3.metric("Team Size", f"{result['staff']:.2f}")
    m4.metric("Cost", format_currency_short(result["cost"]))
    m5.metric("Risk Level", result["risk_level"])

    st.caption(f"Effective KLOC used for calculation: {effective_kloc:.3f}")
    st.caption(f"Full estimated cost: {result['cost']:,.0f} VND")

    driver_df = build_driver_df(selections)
    top5_df = get_top_impact_drivers(driver_df)

    top_col1, top_col2 = st.columns([1.2, 1])
    with top_col1:
        fig1 = px.bar(driver_df, x="Driver", y="Multiplier", color="Group", hover_data=["Name", "Rating"], title="Cost Driver Multipliers")
        st.plotly_chart(fig1, use_container_width=True)
    with top_col2:
        st.subheader("Top 5 Most Influential Drivers")
        st.dataframe(top5_df[["Driver", "Name", "Rating", "Multiplier", "Group"]], use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        compare_rows = []
        for each_mode in MODES:
            mode_result = compute_result(
                project_name=result["project_name"],
                description=result["description"],
                mode=each_mode,
                kloc=effective_kloc,
                cost_per_pm=result["cost_per_pm"],
                selections=selections,
            )
            compare_rows.append({"Mode": each_mode, "Effort": mode_result["effort"], "Development Time": mode_result["tdev"], "Average Staff": mode_result["staff"]})
        compare_df = pd.DataFrame(compare_rows)
        fig2 = px.bar(compare_df, x="Mode", y="Effort", color="Mode", title="Effort Comparison by Mode")
        st.plotly_chart(fig2, use_container_width=True)

    with chart_col2:
        cost_breakdown_df = pd.DataFrame([
            {"Category": "Development", "Cost": result["cost"] * 0.55},
            {"Category": "Testing & QA", "Cost": result["cost"] * 0.20},
            {"Category": "Management", "Cost": result["cost"] * 0.15},
            {"Category": "Tools & Infrastructure", "Cost": result["cost"] * 0.10},
        ])
        fig3 = px.pie(cost_breakdown_df, names="Category", values="Cost", title="Estimated Cost Breakdown")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Driver Table")
    st.dataframe(driver_df[["Driver", "Name", "Group", "Rating", "Multiplier"]], use_container_width=True)

    report_text = export_text_report(result, selections, st.session_state.ai_result, fp_snapshot, all_warning_messages)
    pdf_bytes = make_pdf_report_bytes(st.session_state.project_name, result, selections, fp_snapshot, all_warning_messages, st.session_state.ai_result)

    actions1, actions2, actions3, actions4 = st.columns(4)
    with actions1:
        if st.button("Generate AI Analysis", use_container_width=True):
            if enable_ai:
                ai_text, _ = call_ai_analysis(
                    st.session_state.description,
                    st.session_state.mode,
                    effective_kloc,
                    result["eaf"],
                    result["effort"],
                    result["tdev"],
                    result["staff"],
                    result["cost"],
                    selections,
                )
                if ai_text:
                    st.session_state.ai_result = ai_text
                else:
                    st.session_state.ai_result = fallback_ai_summary(
                        st.session_state.description,
                        st.session_state.mode,
                        effective_kloc,
                        result["eaf"],
                        result["effort"],
                        result["tdev"],
                        result["staff"],
                        result["cost"],
                        selections,
                    )
                    st.warning("AI is temporarily unavailable. Fallback analysis is shown.")
            else:
                st.session_state.ai_result = fallback_ai_summary(
                    st.session_state.description,
                    st.session_state.mode,
                    effective_kloc,
                    result["eaf"],
                    result["effort"],
                    result["tdev"],
                    result["staff"],
                    result["cost"],
                    selections,
                )
            st.rerun()

    with actions2:
        if st.button("Save as New Version", use_container_width=True):
            payload = build_current_version_payload(result, selections, fp_snapshot, all_warning_messages)
            if not st.session_state.active_project_id:
                create_new_project_with_version(payload)
                st.success("New project and first version created.")
            else:
                save_current_as_new_version(payload)
                st.success("New version saved.")
            save_history_row({
                "timestamp": now_text(),
                "project_name": result["project_name"],
                "mode": result["mode"],
                "kloc": result["kloc"],
                "eaf": result["eaf"],
                "effort_pm": result["effort"],
                "development_time_months": result["tdev"],
                "team_size": result["staff"],
                "estimated_cost_vnd": result["cost"],
                "risk_level": result["risk_level"],
            })
            refresh_workspace()
            st.rerun()

    with actions3:
        st.download_button(
            "Download TXT Report",
            data=report_text,
            file_name=f"{(result['project_name'] or 'project').replace(' ', '_').lower()}_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with actions4:
        if pdf_bytes:
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"{(result['project_name'] or 'project').replace(' ', '_').lower()}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("PDF export requires reportlab in requirements.txt")

    st.subheader("AI Project Analysis")
    if st.session_state.ai_result:
        st.markdown(st.session_state.ai_result)
    else:
        st.caption("No AI analysis yet. Click Generate AI Analysis.")



with tab2:
    st.subheader("Function Point Estimator")
    st.caption("Enter 5 FP groups, calculate UFP / VAF / FP, convert to SLOC/KLOC, analyze with AI, and optionally send FP size to COCOMO.")

    st.selectbox("Programming Language", list(LANGUAGE_LOC_PER_FP.keys()) + ["Custom"], key="fp_language")
    if st.session_state.fp_language == "Custom":
        st.number_input("Custom LOC per FP", min_value=1.0, step=1.0, key="fp_custom_loc_per_fp")
    loc_per_fp = get_fp_loc_per_fp()

    fp_left, fp_right = st.columns([1, 1])
    with fp_left:
        st.markdown("### FP Counts")
        for component, label in FP_COMPONENT_LABELS.items():
            st.markdown(f"**{component} - {label}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("Low", min_value=0, step=1, key=f"fp_{component}_low")
            with c2:
                st.number_input("Average", min_value=0, step=1, key=f"fp_{component}_average")
            with c3:
                st.number_input("High", min_value=0, step=1, key=f"fp_{component}_high")

    with fp_right:
        st.markdown("### General System Characteristics (0–5)")
        for idx, name in enumerate(GSC_NAMES):
            st.slider(name, min_value=0, max_value=5, key=f"fp_gsc_{idx}")

    fp_snapshot = get_fp_snapshot()
    if effective_kloc > 0:
        diff_pct = abs(fp_snapshot["kloc"] - effective_kloc) / effective_kloc
        if diff_pct >= 0.5:
            st.warning(f"FP KLOC ({fp_snapshot['kloc']:.2f}) differs a lot from manual KLOC ({effective_kloc:.2f}).")

    fp_based_result = compute_result(
        project_name=f"{st.session_state.project_name or 'FP Project'} (FP-based)",
        description=st.session_state.description or "Function Point derived estimation",
        mode=st.session_state.fp_mode,
        kloc=max(fp_snapshot["kloc"], 0.001),
        cost_per_pm=st.session_state.fp_cost_per_pm,
        selections=get_driver_selections(),
    )

    st.markdown("### FP Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("UFP", f"{fp_snapshot['ufp']:.2f}")
    m2.metric("DI", f"{fp_snapshot['di']}")
    m3.metric("VAF", f"{fp_snapshot['vaf']:.2f}")
    m4.metric("FP", f"{fp_snapshot['fp']:.2f}")
    m5.metric("KLOC", f"{fp_snapshot['kloc']:.2f}")

    st.caption(f"Estimated SLOC from FP: {fp_snapshot['sloc']:,.0f}")
    st.caption(f"LOC per FP factor used: {loc_per_fp:.2f}")

    with st.expander("How Function Point is calculated", expanded=False):
        st.markdown("""
- **UFP** = sum of weighted functions  
- **VAF** = 0.65 + 0.01 × DI  
- **FP** = UFP × VAF  
- **SLOC** = FP × LOC/FP  
- **KLOC** = SLOC / 1000
""")

    fp_mode_col1, fp_mode_col2 = st.columns(2)
    with fp_mode_col1:
        st.selectbox("Mode for FP-based COCOMO", list(MODES.keys()), key="fp_mode")
    with fp_mode_col2:
        st.number_input("FP-based Cost per Person-Month (VND)", min_value=1000000.0, step=1000000.0, key="fp_cost_per_pm")

    st.markdown("### FP-based COCOMO Projection")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Effort (PM)", f"{fp_based_result['effort']:.2f}")
    k2.metric("Time (Months)", f"{fp_based_result['tdev']:.2f}")
    k3.metric("Team Size", f"{fp_based_result['staff']:.2f}")
    k4.metric("Cost", format_currency_short(fp_based_result["cost"]))
    k5.metric("Risk Level", fp_based_result["risk_level"])

    fp_action1, fp_action2 = st.columns(2)
    with fp_action1:
        if st.button("Generate FP AI Analysis", use_container_width=True):
            if enable_ai:
                ai_fp_text, _ = call_ai_fp_analysis(fp_snapshot, fp_based_result)
                if ai_fp_text:
                    st.session_state.ai_fp_result = ai_fp_text
                else:
                    st.session_state.ai_fp_result = fallback_fp_ai_summary(fp_snapshot, fp_based_result)
                    st.warning("AI FP is temporarily unavailable. Fallback analysis is shown.")
            else:
                st.session_state.ai_fp_result = fallback_fp_ai_summary(fp_snapshot, fp_based_result)
            st.rerun()

    with fp_action2:
        if st.button("Send FP Size to COCOMO", use_container_width=True):
            st.session_state.fp_transfer_kloc = max(fp_snapshot["kloc"], 0.001)
            st.session_state.pending_fp_transfer = True
            st.rerun()

    st.subheader("FP AI Analysis")
    if st.session_state.ai_fp_result:
        st.markdown(st.session_state.ai_fp_result)
    else:
        st.caption("No FP AI analysis yet. Click Generate FP AI Analysis.")

with tab3:
    st.subheader("AI Optimization")
    st.caption("AI reads the current project, proposes structured changes, compares current vs suggested, and lets you apply them.")

    current_selections = get_driver_selections()
    current_result = compute_result(
        project_name=st.session_state.project_name,
        description=st.session_state.description,
        mode=st.session_state.mode,
        kloc=get_effective_kloc(),
        cost_per_pm=st.session_state.cost_per_pm,
        selections=current_selections,
    )

    top_row1, top_row2 = st.columns(2)
    with top_row1:
        st.markdown("### Current Baseline")
        st.write(f"**Project:** {current_result['project_name'] or 'Untitled Project'}")
        st.write(f"**Mode:** {current_result['mode']}")
        st.write(f"**KLOC:** {current_result['kloc']:.2f}")
        st.write(f"**Effort:** {current_result['effort']:.2f} PM")
        st.write(f"**Cost:** {current_result['cost']:,.0f} VND")
        st.write(f"**Risk:** {current_result['risk_level']}")

    with top_row2:
        if st.button("Generate AI Optimization", use_container_width=True):
            suggestion, error_msg = call_ai_optimization(
                project_name=current_result["project_name"],
                description=current_result["description"],
                current_result=current_result,
                selections=current_selections,
            )
            st.session_state.ai_suggestion = suggestion
            if error_msg:
                st.warning("AI optimization is temporarily unavailable. Fallback suggestion is shown.")
            else:
                st.success("AI optimization generated.")
            st.rerun()

    suggestion = st.session_state.get("ai_suggestion")
    if suggestion:
        suggested_selections, suggested_result = apply_suggestion_to_result(current_result, current_selections, suggestion)

        st.markdown("### Suggested Changes")
        st.write(f"**Suggested Mode:** {suggestion.get('suggested_mode', current_result['mode'])}")
        if suggestion.get("goals"):
            st.write("**Goals:** " + ", ".join(suggestion["goals"]))
        if suggestion.get("expected_effect"):
            st.write(f"**Expected Effect:** {suggestion['expected_effect']}")
        if suggestion.get("reasoning"):
            st.write("**Reasoning:**")
            for item in suggestion["reasoning"]:
                st.write(f"- {item}")

        driver_change_df = build_driver_change_df(current_selections, suggested_selections)
        if not driver_change_df.empty:
            st.markdown("### Driver Changes")
            st.dataframe(driver_change_df, use_container_width=True)
        else:
            st.info("No cost driver changes. Only mode may have changed.")

        st.markdown("### Current vs Suggested")
        compare_df = build_metric_compare_df(current_result, suggested_result)
        st.dataframe(compare_df, use_container_width=True)

        compare_chart_df = pd.DataFrame([
            {"Scenario": "Current", "Metric": "Effort", "Value": current_result["effort"]},
            {"Scenario": "Suggested", "Metric": "Effort", "Value": suggested_result["effort"]},
            {"Scenario": "Current", "Metric": "Time", "Value": current_result["tdev"]},
            {"Scenario": "Suggested", "Metric": "Time", "Value": suggested_result["tdev"]},
            {"Scenario": "Current", "Metric": "Cost", "Value": current_result["cost"]},
            {"Scenario": "Suggested", "Metric": "Cost", "Value": suggested_result["cost"]},
        ])
        fig_ai = px.bar(compare_chart_df, x="Metric", y="Value", color="Scenario", barmode="group", title="Current vs Suggested")
        st.plotly_chart(fig_ai, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Cost", format_currency_short(current_result["cost"]))
        c2.metric("Suggested Cost", format_currency_short(suggested_result["cost"]))
        c3.metric("Cost Delta", f"{suggested_result['cost'] - current_result['cost']:+,.0f}")

        if st.button("Apply AI Suggestion", use_container_width=True):
            st.session_state.pending_ai_apply = True
            st.rerun()
    else:
        st.info("No AI optimization yet. Click Generate AI Optimization.")

with tab4:
    st.subheader("Workspace")
    refresh_workspace()

    ws_col1, ws_col2 = st.columns([1, 1.2])
    with ws_col1:
        st.markdown("### Import Project Package JSON")
        uploaded_json = st.file_uploader("Choose JSON package", type=["json"])
        if uploaded_json is not None:
            if st.button("Import JSON Package", use_container_width=True):
                ok, msg = import_project_package(uploaded_json)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()

    with ws_col2:
        st.markdown("### Current Workspace Summary")
        projects = st.session_state.workspace.get("projects", [])
        total_versions = sum(len(p.get("versions", [])) for p in projects)
        s1, s2, s3 = st.columns(3)
        s1.metric("Projects", len(projects))
        s2.metric("Versions", total_versions)
        s3.metric("Current Active Project", st.session_state.active_project_id or "-")


    if not st.session_state.workspace.get("projects"):
        st.info("No project in workspace yet.")
    else:
        for project in st.session_state.workspace.get("projects", []):
            with st.expander(f"{project['project_name']} - {project['project_id']}", expanded=False):
                st.write(f"**Description:** {project['description']}")
                st.write(f"**Created:** {project['created_at']}")
                st.write(f"**Updated:** {project['updated_at']}")
                versions_df = project_versions_df(project)
                st.dataframe(versions_df, use_container_width=True)

                version_ids = [v["version_id"] for v in project.get("versions", [])]
                selected_ver_for_action = st.selectbox(
                    f"Choose version - {project['project_id']}",
                    version_ids,
                    key=f"action_version_{project['project_id']}"
                )

                act1, act2, act3 = st.columns(3)
                with act1:
                    if st.button(f"Set Baseline - {project['project_id']}", use_container_width=True):
                        if set_baseline(project["project_id"], selected_ver_for_action):
                            st.success("Baseline updated.")
                            st.rerun()
                with act2:
                    if st.button(f"Load Form - {project['project_id']}", use_container_width=True):
                        load_version_to_form(project["project_id"], selected_ver_for_action)
                        st.success("Version loaded to form.")
                        st.rerun()
                with act3:
                    package = build_project_package(project["project_id"], selected_ver_for_action)
                    if package:
                        st.download_button(
                            f"Export JSON - {project['project_id']}",
                            data=json.dumps(package, ensure_ascii=False, indent=2).encode("utf-8"),
                            file_name=f"{project['project_name'].replace(' ', '_').lower()}_{selected_ver_for_action}.json",
                            mime="application/json",
                            use_container_width=True,
                            key=f"download_json_{project['project_id']}"
                        )

        if st.session_state.active_project_id:
            active_project = find_project_by_id(st.session_state.active_project_id)
            if active_project:
                baseline_version = get_baseline_version(active_project)
                current_result = compute_result(
                    project_name=st.session_state.project_name,
                    description=st.session_state.description,
                    mode=st.session_state.mode,
                    kloc=get_effective_kloc(),
                    cost_per_pm=st.session_state.cost_per_pm,
                    selections=get_driver_selections(),
                )

                st.markdown("### Baseline vs Current")
                if baseline_version:
                    baseline_result = baseline_version["result"]
                    compare_baseline_df = build_baseline_compare_df(current_result, baseline_result)
                    st.dataframe(compare_baseline_df, use_container_width=True)

                    baseline_chart_df = pd.DataFrame([
                        {"Scenario": "Baseline", "Metric": "Effort", "Value": baseline_result["effort"]},
                        {"Scenario": "Current", "Metric": "Effort", "Value": current_result["effort"]},
                        {"Scenario": "Baseline", "Metric": "Time", "Value": baseline_result["tdev"]},
                        {"Scenario": "Current", "Metric": "Time", "Value": current_result["tdev"]},
                        {"Scenario": "Baseline", "Metric": "Cost", "Value": baseline_result["cost"]},
                        {"Scenario": "Current", "Metric": "Cost", "Value": current_result["cost"]},
                    ])
                    fig_baseline = px.bar(baseline_chart_df, x="Metric", y="Value", color="Scenario", barmode="group", title="Baseline vs Current")
                    st.plotly_chart(fig_baseline, use_container_width=True)
                else:
                    st.info("Current project has no baseline yet.")

    st.subheader("History CSV")
    history_df = load_history_df()
    if history_df.empty:
        st.info("No history saved yet.")
    else:
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "Download History CSV",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="estimation_history.csv",
            mime="text/csv",
        )

with tab5:
    st.subheader("Compare Modes & What-if")
    selections = get_driver_selections()
    effective_kloc = get_effective_kloc()

    mode_rows = []
    for each_mode in MODES:
        mode_result = compute_result(
            project_name=st.session_state.project_name,
            description=st.session_state.description,
            mode=each_mode,
            kloc=effective_kloc,
            cost_per_pm=st.session_state.cost_per_pm,
            selections=selections,
        )
        mode_rows.append({
            "Mode": each_mode,
            "Effort (PM)": round(mode_result["effort"], 2),
            "Time (Months)": round(mode_result["tdev"], 2),
            "Team Size": round(mode_result["staff"], 2),
            "Cost (VND)": round(mode_result["cost"], 0),
        })

    mode_df = pd.DataFrame(mode_rows)
    st.dataframe(mode_df, use_container_width=True)
    fig_compare = px.bar(mode_df, x="Mode", y="Cost (VND)", color="Mode", title="Estimated Cost by Mode")
    st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("What-if Analysis")
    scenario_driver = st.selectbox("Choose one driver to change", list(COST_DRIVERS.keys()))
    scenario_options = list(COST_DRIVERS[scenario_driver]["values"].keys())
    current_rating = selections[scenario_driver]
    new_rating = st.selectbox("New rating", scenario_options, index=scenario_options.index(current_rating))

    scenario_selections = selections.copy()
    scenario_selections[scenario_driver] = new_rating

    base_result = compute_result(
        project_name=st.session_state.project_name,
        description=st.session_state.description,
        mode=st.session_state.mode,
        kloc=effective_kloc,
        cost_per_pm=st.session_state.cost_per_pm,
        selections=selections,
    )
    new_result = compute_result(
        project_name=st.session_state.project_name,
        description=st.session_state.description,
        mode=st.session_state.mode,
        kloc=effective_kloc,
        cost_per_pm=st.session_state.cost_per_pm,
        selections=scenario_selections,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EAF Change", f"{new_result['eaf']:.3f}", f"{new_result['eaf'] - base_result['eaf']:+.3f}")
    c2.metric("Effort Change", f"{new_result['effort']:.2f}", f"{new_result['effort'] - base_result['effort']:+.2f}")
    c3.metric("Time Change", f"{new_result['tdev']:.2f}", f"{new_result['tdev'] - base_result['tdev']:+.2f}")
    c4.metric("Cost Change", format_currency_short(new_result["cost"]), f"{new_result['cost'] - base_result['cost']:+,.0f}")

    what_if_df = pd.DataFrame([
        {"Scenario": "Current", "EAF": base_result["eaf"], "Effort": base_result["effort"], "Time": base_result["tdev"], "Cost": base_result["cost"]},
        {"Scenario": "What-if", "EAF": new_result["eaf"], "Effort": new_result["effort"], "Time": new_result["tdev"], "Cost": new_result["cost"]},
    ])

    fig_what_if = px.bar(
        what_if_df.melt(id_vars="Scenario", value_vars=["Effort", "Time", "Cost"]),
        x="variable",
        y="value",
        color="Scenario",
        barmode="group",
        title="What-if Comparison",
    )
    st.plotly_chart(fig_what_if, use_container_width=True)

with tab6:
    st.subheader("Help")
    st.markdown("""
### 1. Intermediate COCOMO
- **Effort** = a × (KLOC ^ b) × EAF  
- **Time** = c × (Effort ^ d)  
- **Team Size** = Effort / Time  

### 2. Function Point
- **UFP** = sum of weighted functions  
- **VAF** = 0.65 + 0.01 × DI  
- **FP** = UFP × VAF  
- **SLOC** = FP × LOC/FP  
- **KLOC** = SLOC / 1000  

### 3. Recommended Workflow
1. Enter project description and KLOC or use FP  
2. Select development mode and 15 cost drivers  
3. Review effort, time, cost, and risk  
4. Generate AI Project Analysis  
5. Generate AI Optimization if needed  
6. Save version or export JSON / PDF report  

### 4. Inline Warnings
Warnings are now shown directly near the related input or section:
- Project Name
- Description
- Mode
- KLOC / SLOC
- Cost per PM
- Cost drivers
- FP KLOC mismatch

### 5. Notes
- If PDF export is not available on Streamlit Cloud, add `reportlab` to `requirements.txt`
- PDF report is generated in English to avoid Vietnamese font issues
""")

    st.subheader("Feature Review")
    for note in feature_review_notes():
        st.write(f"- {note}")

st.markdown("---")
st.caption("Final version: removed unnecessary fields, inline warnings, English PDF report with charts, richer AI analysis, and fixed Help content.")
