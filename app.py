import os
import io
import json
import math
import uuid
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
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
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
    "RELY": {
        "label": "Required Software Reliability",
        "help": "Mức độ yêu cầu độ tin cậy của phần mềm.",
        "values": {"Very Low": 0.75, "Low": 0.88, "Nominal": 1.00, "High": 1.15, "Very High": 1.40},
    },
    "DATA": {
        "label": "Database Size",
        "help": "Quy mô dữ liệu mà hệ thống cần xử lý.",
        "values": {"Low": 0.94, "Nominal": 1.00, "High": 1.08, "Very High": 1.16},
    },
    "CPLX": {
        "label": "Product Complexity",
        "help": "Độ phức tạp tổng thể của sản phẩm.",
        "values": {"Very Low": 0.70, "Low": 0.85, "Nominal": 1.00, "High": 1.15, "Very High": 1.30, "Extra High": 1.65},
    },
    "TIME": {
        "label": "Execution Time Constraint",
        "help": "Ràng buộc về thời gian thực thi.",
        "values": {"Nominal": 1.00, "High": 1.11, "Very High": 1.30, "Extra High": 1.66},
    },
    "STOR": {
        "label": "Storage Constraint",
        "help": "Ràng buộc về bộ nhớ hoặc lưu trữ.",
        "values": {"Nominal": 1.00, "High": 1.06, "Very High": 1.21, "Extra High": 1.56},
    },
    "VIRT": {
        "label": "Virtual Machine Volatility",
        "help": "Mức độ thay đổi của môi trường chạy.",
        "values": {"Low": 0.87, "Nominal": 1.00, "High": 1.15, "Very High": 1.30},
    },
    "TURN": {
        "label": "Computer Turnaround Time",
        "help": "Tốc độ phản hồi và chu kỳ xử lý của hệ thống.",
        "values": {"Low": 0.87, "Nominal": 1.00, "High": 1.07, "Very High": 1.15},
    },
    "ACAP": {
        "label": "Analyst Capability",
        "help": "Năng lực của người phân tích hệ thống.",
        "values": {"Very Low": 1.46, "Low": 1.19, "Nominal": 1.00, "High": 0.86, "Very High": 0.71},
    },
    "AEXP": {
        "label": "Application Experience",
        "help": "Kinh nghiệm với domain nghiệp vụ tương tự.",
        "values": {"Very Low": 1.29, "Low": 1.13, "Nominal": 1.00, "High": 0.91, "Very High": 0.82},
    },
    "PCAP": {
        "label": "Programmer Capability",
        "help": "Năng lực lập trình viên.",
        "values": {"Very Low": 1.42, "Low": 1.17, "Nominal": 1.00, "High": 0.86, "Very High": 0.70},
    },
    "VEXP": {
        "label": "Virtual Machine Experience",
        "help": "Kinh nghiệm với môi trường/hệ thống mục tiêu.",
        "values": {"Very Low": 1.21, "Low": 1.10, "Nominal": 1.00, "High": 0.90},
    },
    "LEXP": {
        "label": "Language Experience",
        "help": "Kinh nghiệm với ngôn ngữ lập trình đang dùng.",
        "values": {"Very Low": 1.14, "Low": 1.07, "Nominal": 1.00, "High": 0.95},
    },
    "MODP": {
        "label": "Modern Programming Practices",
        "help": "Mức độ áp dụng thực hành lập trình hiện đại.",
        "values": {"Very Low": 1.24, "Low": 1.10, "Nominal": 1.00, "High": 0.91, "Very High": 0.82},
    },
    "TOOL": {
        "label": "Use of Software Tools",
        "help": "Mức độ hỗ trợ của tool, IDE, testing, CI/CD...",
        "values": {"Very Low": 1.24, "Low": 1.10, "Nominal": 1.00, "High": 0.91, "Very High": 0.83},
    },
    "SCED": {
        "label": "Required Development Schedule",
        "help": "Mức độ gắt của deadline.",
        "values": {"Very Low": 1.23, "Low": 1.08, "Nominal": 1.00, "High": 1.04, "Very High": 1.10},
    },
}

PRESET_PROJECTS = {
    "Start from Scratch": None,
    "Personal Expense Tracker": {
        "project_name": "Personal Expense Tracker",
        "description": "A simple personal expense tracking web application with category management, monthly reports, and budget summaries.",
        "mode": "Organic",
        "kloc": 8.0,
        "cost_per_pm": 12000000.0,
        "drivers": {
            "RELY": "Low", "DATA": "Low", "CPLX": "Low",
            "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Low", "TURN": "Low",
            "ACAP": "High", "AEXP": "High", "PCAP": "High", "VEXP": "Nominal", "LEXP": "High",
            "MODP": "High", "TOOL": "Very High", "SCED": "Nominal",
        },
    },
    "School Management System": {
        "project_name": "School Management System",
        "description": "A school management system with student records, teacher management, attendance tracking, exam scores, and report generation.",
        "mode": "Organic",
        "kloc": 18.0,
        "cost_per_pm": 12000000.0,
        "drivers": {
            "RELY": "Nominal", "DATA": "Nominal", "CPLX": "Nominal",
            "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal",
            "ACAP": "Nominal", "AEXP": "Nominal", "PCAP": "Nominal", "VEXP": "Nominal", "LEXP": "Nominal",
            "MODP": "High", "TOOL": "High", "SCED": "Nominal",
        },
    },
    "Hotel Management System": {
        "project_name": "Hotel Management System",
        "description": "A medium-sized hotel management system with booking, customer management, billing, room service tracking, reports, and staff administration.",
        "mode": "Semi-detached",
        "kloc": 30.0,
        "cost_per_pm": 12000000.0,
        "drivers": {
            "RELY": "High", "DATA": "Nominal", "CPLX": "High",
            "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal",
            "ACAP": "Nominal", "AEXP": "Nominal", "PCAP": "Nominal", "VEXP": "Nominal", "LEXP": "Nominal",
            "MODP": "Nominal", "TOOL": "Nominal", "SCED": "High",
        },
    },
    "E-commerce Platform": {
        "project_name": "E-commerce Platform",
        "description": "An e-commerce platform with user accounts, product catalog, online payments, inventory tracking, order management, and admin analytics.",
        "mode": "Semi-detached",
        "kloc": 45.0,
        "cost_per_pm": 13000000.0,
        "drivers": {
            "RELY": "High", "DATA": "High", "CPLX": "High",
            "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal",
            "ACAP": "Nominal", "AEXP": "Low", "PCAP": "Nominal", "VEXP": "Nominal", "LEXP": "Nominal",
            "MODP": "Nominal", "TOOL": "High", "SCED": "High",
        },
    },
    "Medical Record System": {
        "project_name": "Medical Record System",
        "description": "A medical record management system for hospitals with patient histories, diagnostic records, laboratory integration, doctor access control, and strict reliability requirements.",
        "mode": "Embedded",
        "kloc": 55.0,
        "cost_per_pm": 15000000.0,
        "drivers": {
            "RELY": "Very High", "DATA": "High", "CPLX": "High",
            "TIME": "High", "STOR": "High", "VIRT": "Nominal", "TURN": "Nominal",
            "ACAP": "Nominal", "AEXP": "Low", "PCAP": "Nominal", "VEXP": "Nominal", "LEXP": "Nominal",
            "MODP": "High", "TOOL": "High", "SCED": "High",
        },
    },
    "Banking Transaction System": {
        "project_name": "Banking Transaction System",
        "description": "A critical banking transaction system with strong security, high reliability, strict schedule requirements, and high transaction volume.",
        "mode": "Embedded",
        "kloc": 60.0,
        "cost_per_pm": 15000000.0,
        "drivers": {
            "RELY": "Very High", "DATA": "Very High", "CPLX": "Very High",
            "TIME": "Very High", "STOR": "High", "VIRT": "High", "TURN": "High",
            "ACAP": "Nominal", "AEXP": "Low", "PCAP": "Nominal", "VEXP": "Low", "LEXP": "Nominal",
            "MODP": "Nominal", "TOOL": "Nominal", "SCED": "Very High",
        },
    },
    "Real-Time IoT Monitoring System": {
        "project_name": "Real-Time IoT Monitoring System",
        "description": "A real-time IoT monitoring system for industrial devices with sensor streaming, fault alerts, dashboard visualization, and strict execution time constraints.",
        "mode": "Embedded",
        "kloc": 70.0,
        "cost_per_pm": 16000000.0,
        "drivers": {
            "RELY": "Very High", "DATA": "High", "CPLX": "Very High",
            "TIME": "Extra High", "STOR": "High", "VIRT": "High", "TURN": "High",
            "ACAP": "Nominal", "AEXP": "Low", "PCAP": "Nominal", "VEXP": "Low", "LEXP": "Nominal",
            "MODP": "Nominal", "TOOL": "Nominal", "SCED": "Very High",
        },
    },
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
    "Data communications",
    "Distributed data processing",
    "Performance",
    "Heavily used configuration",
    "Transaction rate",
    "Online data entry",
    "End-user efficiency",
    "Online update",
    "Complex processing",
    "Reusability",
    "Installation ease",
    "Operational ease",
    "Multiple sites",
    "Facilitate change",
]

LANGUAGE_LOC_PER_FP = {
    "Python": 50,
    "Java": 53,
    "C": 128,
    "C++": 53,
    "JavaScript": 47,
    "C#": 54,
    "PHP": 50,
    "Go": 55,
    "Ruby": 40,
}

APP_SCHEMA_VERSION = "2.0.0"


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
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


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


def init_state():
    defaults = {
        "selected_preset": "Start from Scratch",
        "preset_selector": "Start from Scratch",
        "project_name": "",
        "project_owner": "Default User",
        "project_status": "Draft",
        "project_domain": "",
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
        "version_title": "Version mới",
        "version_note": "",
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
    st.session_state.project_owner = "Default User"
    st.session_state.project_status = "Draft"
    st.session_state.project_domain = ""
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
    st.session_state.version_title = "Version mới"
    st.session_state.version_note = ""
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


def build_driver_df(selections: dict) -> pd.DataFrame:
    rows = []
    for driver, rating in selections.items():
        multiplier = COST_DRIVERS[driver]["values"][rating]
        rows.append({
            "Driver": driver,
            "Name": COST_DRIVERS[driver]["label"],
            "Rating": rating,
            "Multiplier": multiplier,
            "Impact": abs(multiplier - 1.0),
            "Group": get_driver_group(driver),
        })
    return pd.DataFrame(rows)


def get_top_impact_drivers(driver_df: pd.DataFrame) -> pd.DataFrame:
    return driver_df.sort_values(by="Impact", ascending=False).head(5)


def compute_result(project_name: str, description: str, mode: str, kloc: float, cost_per_pm: float, selections: dict):
    eaf = calculate_eaf(selections)
    effort, tdev, staff = cocomo_estimate(mode, kloc, eaf)
    cost = estimate_cost(effort, cost_per_pm)
    risk_level = get_risk_level(eaf, selections)
    return {
        "project_name": project_name,
        "description": description,
        "mode": mode,
        "kloc": kloc,
        "cost_per_pm": cost_per_pm,
        "eaf": eaf,
        "effort": effort,
        "tdev": tdev,
        "staff": staff,
        "cost": cost,
        "risk_level": risk_level,
    }


def fallback_ai_summary(description, mode, kloc, eaf, effort, tdev, staff, cost, selections):
    risks = []
    recommendations = []

    if selections["CPLX"] in ["High", "Very High", "Extra High"]:
        risks.append("Độ phức tạp dự án cao, làm effort tăng đáng kể.")
        recommendations.append("Tách hệ thống thành module nhỏ hơn hoặc chia giai đoạn triển khai.")

    if selections["RELY"] in ["High", "Very High"]:
        risks.append("Yêu cầu độ tin cậy cao làm effort cho kiểm thử và xác minh tăng mạnh.")
        recommendations.append("Bổ sung thêm thời gian cho test, review, QA và bảo mật.")

    if selections["ACAP"] in ["Very Low", "Low"] or selections["PCAP"] in ["Very Low", "Low"]:
        risks.append("Năng lực nhân sự hiện tại có thể làm giảm năng suất thực tế.")
        recommendations.append("Tăng mentoring, review hoặc phân công người mạnh hơn vào phần lõi.")

    if selections["TOOL"] in ["Very Low", "Low"]:
        risks.append("Tool support còn yếu nên tốc độ phát triển sẽ giảm.")
        recommendations.append("Nâng cấp IDE, test tools, Git workflow, CI/CD hoặc automation.")

    if selections["SCED"] in ["High", "Very High"]:
        risks.append("Deadline gắt làm tăng khả năng trễ tiến độ và giảm chất lượng.")
        recommendations.append("Nên kiểm soát scope hoặc nới timeline.")

    if not risks:
        risks.append("Chưa thấy rủi ro nổi bật từ bộ cost drivers hiện tại.")

    if not recommendations:
        recommendations.append("Cấu hình hiện tại tương đối cân bằng.")

    return f"""
### Phân tích estimate dự án

**Mode đang dùng:** {mode}  
**KLOC:** {kloc:.2f}  
**EAF:** {eaf:.3f}  
**Effort:** {effort:.2f} person-months  
**Development Time:** {tdev:.2f} months  
**Average Team Size:** {staff:.2f}  
**Estimated Cost:** {cost:,.0f} VND  

**Đánh giá nhanh**
- Mode gợi ý theo mô tả: **{suggest_mode_rule_based(description)}**
- Rủi ro tổng thể: **{get_risk_level(eaf, selections)}**

**Rủi ro chính**
{chr(10).join([f"- {r}" for r in risks])}

**Khuyến nghị**
{chr(10).join([f"- {r}" for r in recommendations])}
""".strip()


def fallback_fp_ai_summary(fp_data: dict, fp_based_result: dict):
    fp_value = fp_data["fp"]
    ufp = fp_data["ufp"]
    di = fp_data["di"]
    vaf = fp_data["vaf"]
    sloc = fp_data["sloc"]
    kloc = fp_data["kloc"]
    language = fp_data["language"]
    loc_per_fp = fp_data["loc_per_fp"]

    comments = []
    if fp_value < 50:
        comments.append("Quy mô FP đang ở mức nhỏ, phù hợp với hệ thống đơn giản hoặc phạm vi hẹp.")
    elif fp_value < 200:
        comments.append("Quy mô FP đang ở mức trung bình, phù hợp với phần lớn đồ án hoặc hệ thống nghiệp vụ vừa.")
    else:
        comments.append("Quy mô FP khá lớn, nên rà soát lại phạm vi chức năng và chia module hợp lý.")

    if vaf >= 1.00:
        comments.append("VAF tương đối cao, cho thấy hệ thống có nhiều đặc tính tổng quát làm tăng độ khó.")
    else:
        comments.append("VAF chưa quá cao, nghĩa là các đặc tính tổng quát chưa làm hệ thống tăng độ khó mạnh.")

    return f"""
### Phân tích Function Point

**UFP:** {ufp:.2f}  
**DI:** {di}  
**VAF:** {vaf:.2f}  
**FP:** {fp_value:.2f}  
**Ngôn ngữ quy đổi:** {language}  
**LOC per FP:** {loc_per_fp:.2f}  
**SLOC ước tính:** {sloc:,.0f}  
**KLOC ước tính:** {kloc:.2f}  

**Đánh giá**
- {comments[0]}
- {comments[1]}

**COCOMO từ FP**
- Effort: **{fp_based_result["effort"]:.2f} PM**
- Time: **{fp_based_result["tdev"]:.2f} months**
- Team Size: **{fp_based_result["staff"]:.2f}**
- Cost: **{fp_based_result["cost"]:,.0f} VND**

**Khuyến nghị**
- Nên so sánh KLOC nhập tay với KLOC suy ra từ FP để tránh ước lượng lệch.
- Nếu FP-derived KLOC lệch quá xa manual KLOC thì nên review lại scope hoặc cách chấm FP.
""".strip()


def call_ai_analysis(description, mode, kloc, eaf, effort, tdev, staff, cost, selections):
    api_key = get_openai_api_key()
    if not api_key or OpenAI is None:
        return None, "AI unavailable"

    try:
        client = OpenAI(api_key=api_key)

        driver_text = "\n".join(
            [f"{k}: {v} (multiplier={COST_DRIVERS[k]['values'][v]})" for k, v in selections.items()]
        )

        prompt = f"""
Bạn là trợ lý hỗ trợ project manager trong ước lượng dự án phần mềm.

Mô tả dự án:
{description}

Development mode đang chọn:
{mode}

Kích thước ước lượng:
{kloc:.2f} KLOC

Cost drivers đã chọn:
{driver_text}

EAF đã tính:
{eaf:.3f}

Effort đã tính:
{effort:.2f} person-months

Development time đã tính:
{tdev:.2f} months

Average team size:
{staff:.2f}

Estimated cost:
{cost:,.0f} VND

Hãy trả lời bằng tiếng Việt, ngắn gọn, rõ ràng, có 5 phần:
1. Mode hiện tại có phù hợp không
2. Giải thích vì sao effort/cost ở mức này
3. Rủi ro chính
4. Khuyến nghị giảm effort/cost
5. Kết luận 2-3 câu cho manager

Không dùng markdown phức tạp.
"""
        response = client.responses.create(
            model=AI_MODEL_NAME,
            input=prompt,
        )
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
Bạn là trợ lý phân tích Function Point cho project estimation.

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

Hãy trả lời bằng tiếng Việt với 4 phần:
1. Nhận xét quy mô FP
2. Ý nghĩa của VAF và ảnh hưởng của nó
3. Đánh giá KLOC quy đổi từ FP có hợp lý không
4. Kết luận ngắn gọn cho project manager hoặc sinh viên

Ngắn gọn, thực tế, dễ hiểu.
"""
        response = client.responses.create(
            model=AI_MODEL_NAME,
            input=prompt,
        )
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
    suggestion = {
        "suggested_mode": current_mode,
        "suggested_changes": {},
        "goals": [],
        "reasoning": [],
        "expected_effect": "",
    }

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

    expected_effect = data.get("expected_effect", "")
    suggestion["expected_effect"] = str(expected_effect)

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
            "Tăng chất lượng tool support giúp giảm effort phát triển.",
            "Tăng modern practices giúp cải thiện năng suất team.",
            "Giảm schedule pressure giúp giảm rủi ro tiến độ.",
        ],
        "expected_effect": "Cấu hình đề xuất có khả năng giảm effort, cost và risk so với baseline hiện tại.",
    }


def call_ai_optimization(project_name: str, description: str, current_result: dict, selections: dict):
    api_key = get_openai_api_key()
    fallback = fallback_ai_optimization(current_result["mode"], selections)

    if not api_key or OpenAI is None:
        return fallback, "AI unavailable"

    try:
        client = OpenAI(api_key=api_key)

        driver_text = "\n".join(
            [f"{k}: {v} (multiplier={COST_DRIVERS[k]['values'][v]})" for k, v in selections.items()]
        )

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

Hãy đề xuất một cấu hình tốt hơn theo dạng JSON hợp lệ, không markdown, không giải thích ngoài JSON.
Chỉ dùng đúng các driver hiện có.
Chỉ thay đổi tối đa 5 cost drivers.
Không được thay đổi KLOC.

JSON phải có đúng cấu trúc:
{{
  "suggested_mode": "Organic or Semi-detached or Embedded",
  "suggested_changes": {{
    "TOOL": "High",
    "MODP": "High"
  }},
  "goals": ["Giảm effort", "Giảm rủi ro"],
  "reasoning": [
    "reason 1",
    "reason 2"
  ],
  "expected_effect": "short summary"
}}
"""
        response = client.responses.create(
            model=AI_MODEL_NAME,
            input=prompt,
        )
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


def export_text_report(result: dict, selections: dict, ai_text: str, fp_data: dict, validation_messages: list):
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
        "VALIDATION",
        "-" * 20,
    ]

    if validation_messages:
        for msg in validation_messages:
            lines.append(f"- {msg}")
    else:
        lines.append("Không có cảnh báo logic đáng chú ý.")

    lines.extend([
        "",
        "COST DRIVERS",
        "-" * 20,
    ])
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
        ai_text if ai_text else "Chưa có AI analysis.",
    ])
    return "\n".join(lines)


def validate_inputs(project_name, description, mode, kloc, cost_per_pm, selections, fp_snapshot):
    warnings = []
    text = (description or "").lower()
    suggested_mode = suggest_mode_rule_based(description)

    if not project_name.strip():
        warnings.append("Project Name đang để trống.")
    if len((description or "").strip()) < 15:
        warnings.append("Project Description quá ngắn, AI và estimate dễ thiếu ngữ cảnh.")
    if kloc < 1:
        warnings.append("KLOC rất nhỏ, cần chắc chắn đây không phải nhập thiếu đơn vị.")
    if kloc > 1000:
        warnings.append("KLOC quá lớn, cần kiểm tra lại đơn vị KLOC/SLOC.")
    if cost_per_pm < 3000000:
        warnings.append("Cost per Person-Month khá thấp, có thể không thực tế.")
    if mode != suggested_mode and description.strip():
        warnings.append(f"Mode hiện tại là {mode} nhưng mô tả đang gợi ý {suggested_mode}.")
    if selections["RELY"] in ["High", "Very High"] and mode == "Organic":
        warnings.append("RELY cao nhưng mode đang là Organic, nên kiểm tra lại.")
    if selections["TIME"] in ["Very High", "Extra High"] and mode == "Organic":
        warnings.append("TIME rất cao nhưng mode đang là Organic, có thể chưa phù hợp.")
    if selections["CPLX"] in ["Very High", "Extra High"] and kloc < 5:
        warnings.append("CPLX rất cao nhưng KLOC lại khá nhỏ, nên rà soát lại scope.")
    if selections["SCED"] in ["High", "Very High"] and selections["TOOL"] in ["Very Low", "Low", "Nominal"]:
        warnings.append("Deadline gắt nhưng TOOL support thấp, rủi ro trễ tiến độ khá cao.")
    if selections["ACAP"] in ["Very Low", "Low"] and selections["PCAP"] in ["Very Low", "Low"]:
        warnings.append("Cả năng lực analyst và programmer đều thấp, estimate có thể quá lạc quan.")
    if fp_snapshot["fp"] > 0:
        manual_kloc = kloc
        fp_kloc = fp_snapshot["kloc"]
        if manual_kloc > 0:
            diff_pct = abs(fp_kloc - manual_kloc) / manual_kloc
            if diff_pct >= 0.5:
                warnings.append(
                    f"KLOC nhập tay ({manual_kloc:.2f}) lệch nhiều so với KLOC từ FP ({fp_kloc:.2f})."
                )
    return warnings


def build_validation_df(messages: list):
    if not messages:
        return pd.DataFrame([{"Severity": "OK", "Message": "Không có cảnh báo logic đáng chú ý."}])
    return pd.DataFrame([{"Severity": "Warning", "Message": msg} for msg in messages])


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


def build_current_version_payload(result: dict, selections: dict, fp_snapshot: dict, validation_messages: list):
    return {
        "version_id": st.session_state.active_version_id or generate_id("VER"),
        "version_no": 0,
        "title": st.session_state.version_title.strip() or "Version mới",
        "note": st.session_state.version_note.strip(),
        "created_at": now_text(),
        "updated_at": now_text(),
        "is_baseline": False,
        "project_meta": {
            "project_name": st.session_state.project_name,
            "project_owner": st.session_state.project_owner,
            "project_status": st.session_state.project_status,
            "project_domain": st.session_state.project_domain,
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
        "cost_driver_multipliers": {
            driver: COST_DRIVERS[driver]["values"][rating] for driver, rating in selections.items()
        },
        "result": deepcopy(result),
        "validation_messages": deepcopy(validation_messages),
        "ai": {
            "analysis": st.session_state.ai_result,
            "suggestion": deepcopy(st.session_state.ai_suggestion),
        },
        "fp": deepcopy(fp_snapshot),
    }


def create_new_project_with_version(version_payload: dict):
    project_id = generate_id("PRJ")
    project = {
        "project_id": project_id,
        "project_name": version_payload["project_meta"]["project_name"] or "Untitled Project",
        "project_owner": version_payload["project_meta"]["project_owner"],
        "project_status": version_payload["project_meta"]["project_status"],
        "project_domain": version_payload["project_meta"]["project_domain"],
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
    project["project_owner"] = version_payload["project_meta"]["project_owner"]
    project["project_status"] = version_payload["project_meta"]["project_status"]
    project["project_domain"] = version_payload["project_meta"]["project_domain"]
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
    st.session_state.project_owner = version["project_meta"].get("project_owner", "Default User")
    st.session_state.project_status = version["project_meta"].get("project_status", "Draft")
    st.session_state.project_domain = version["project_meta"].get("project_domain", "")
    st.session_state.description = version["project_meta"].get("description", "")
    st.session_state.mode = version["input"].get("mode", "Organic")
    st.session_state.size_input_mode = version["input"].get("size_input_mode", "KLOC")
    st.session_state.kloc = safe_float(version["input"].get("kloc", 1.0), 1.0)
    st.session_state.sloc = safe_float(version["input"].get("sloc", st.session_state.kloc * 1000), 1000.0)
    st.session_state.cost_per_pm = safe_float(version["input"].get("cost_per_pm", 12000000), 12000000.0)
    st.session_state.version_title = version.get("title", "Version mới")
    st.session_state.version_note = version.get("note", "")
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

    package = {
        "schema_version": APP_SCHEMA_VERSION,
        "exported_at": now_text(),
        "project": {
            "project_id": project["project_id"],
            "project_name": project["project_name"],
            "project_owner": project["project_owner"],
            "project_status": project["project_status"],
            "project_domain": project["project_domain"],
            "description": project["description"],
            "created_at": project["created_at"],
            "updated_at": project["updated_at"],
        },
        "version": deepcopy(version),
    }
    return package


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
        "project_owner": project_block.get("project_owner", "Imported User"),
        "project_status": project_block.get("project_status", "Draft"),
        "project_domain": project_block.get("project_domain", ""),
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


def make_pdf_report_bytes(project_meta: dict, result: dict, selections: dict, fp_snapshot: dict, validation_messages: list, ai_text: str):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=32,
        rightMargin=32,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22)
    heading_style = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], alignment=TA_LEFT, fontSize=12, textColor=colors.HexColor("#0d47a1"))
    normal_style = ParagraphStyle("NormalCustom", parent=styles["Normal"], fontSize=9, leading=13)

    story = []
    story.append(Paragraph("Professional Software Estimation Report", title_style))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Thời gian xuất", now_text()],
        ["Project Name", project_meta.get("project_name", "")],
        ["Owner", project_meta.get("project_owner", "")],
        ["Status", project_meta.get("project_status", "")],
        ["Domain", project_meta.get("project_domain", "")],
        ["Mode", result["mode"]],
        ["KLOC", f"{result['kloc']:.2f}"],
        ["Effort", f"{result['effort']:.2f} PM"],
        ["Time", f"{result['tdev']:.2f} months"],
        ["Team Size", f"{result['staff']:.2f}"],
        ["Cost", f"{result['cost']:,.0f} VND"],
        ["Risk", result["risk_level"]],
    ]
    story.append(Paragraph("1. Executive Summary", heading_style))
    table = Table(summary_data, colWidths=[130, 360])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e3f2fd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Project Description", heading_style))
    story.append(Paragraph(project_meta.get("description", "-"), normal_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Validation Warnings", heading_style))
    if validation_messages:
        for msg in validation_messages:
            story.append(Paragraph(f"• {msg}", normal_style))
    else:
        story.append(Paragraph("Không có cảnh báo logic đáng chú ý.", normal_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Cost Drivers", heading_style))
    driver_rows = [["Driver", "Rating", "Multiplier"]]
    for driver, rating in selections.items():
        driver_rows.append([driver, rating, str(COST_DRIVERS[driver]["values"][rating])])
    driver_table = Table(driver_rows, colWidths=[80, 180, 100])
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
    fp_table = Table(fp_rows, colWidths=[130, 180])
    fp_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
    ]))
    story.append(fp_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("6. AI Analysis", heading_style))
    story.append(Paragraph((ai_text or "Chưa có AI analysis.").replace("\n", "<br/>"), normal_style))
    story.append(Spacer(1, 10))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def project_versions_df(project: dict):
    rows = []
    for version in project.get("versions", []):
        result = version.get("result", {})
        rows.append({
            "Version No": version.get("version_no", 0),
            "Title": version.get("title", ""),
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


init_state()
process_pending_actions()

with st.sidebar:
    st.title("Professional Estimation Workspace")
    st.markdown("Estimate project effort, schedule, team size, cost, risk, version history, baseline, JSON package, PDF report và AI phân tích tiếng Việt.")

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
    project_options = ["-- Chưa chọn project --"] + [f"{p['project_name']} ({p['project_id']})" for p in workspace_projects]
    selected_project_label = st.selectbox("Chọn project đã lưu", project_options)

    if selected_project_label != "-- Chưa chọn project --":
        selected_project_id = selected_project_label.split("(")[-1].replace(")", "").strip()
        project_obj = find_project_by_id(selected_project_id)
        if project_obj:
            version_options = [f"v{v['version_no']} - {v['title']} ({v['version_id']})" for v in project_obj.get("versions", [])]
            if version_options:
                selected_version_label = st.selectbox("Chọn version", version_options)
                selected_version_id = selected_version_label.split("(")[-1].replace(")", "").strip()

                if st.button("Load Version to Form", use_container_width=True):
                    ok = load_version_to_form(selected_project_id, selected_version_id)
                    if ok:
                        st.success("Đã nạp version vào form.")
                        st.rerun()

    st.markdown("---")
    st.caption("Project package JSON giúp export/import nguyên trạng project + version + cost drivers + FP + AI.")

st.title("Professional Software Effort Estimation Workspace")
st.caption("Intermediate COCOMO + Function Point + AI Analysis + Versioning + JSON Package + PDF Report + Validation + Baseline Comparison")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Estimator",
    "FP Estimator",
    "AI Optimization",
    "Validation",
    "Workspace",
    "Compare & What-if",
    "Help",
])

with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Project Information")
        st.text_input("Project Name", key="project_name")
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.text_input("Project Owner", key="project_owner")
            st.text_input("Project Domain", key="project_domain")
        with meta_col2:
            st.selectbox("Project Status", ["Draft", "Reviewed", "Approved", "Archived"], key="project_status")
            st.text_input("Version Title", key="version_title")
        st.text_area("Project Description", height=140, key="description")
        st.text_area("Version Note", height=90, key="version_note")

        st.selectbox("Development Mode", list(MODES.keys()), key="mode")
        st.selectbox("Size Input Unit", ["KLOC", "SLOC"], key="size_input_mode")

        if st.session_state.size_input_mode == "KLOC":
            st.number_input("Estimated Size (KLOC)", min_value=0.001, step=1.0, key="kloc")
        else:
            st.number_input("Estimated Size (SLOC)", min_value=1.0, step=100.0, key="sloc")

        st.number_input("Cost per Person-Month (VND)", min_value=1000000.0, step=1000000.0, key="cost_per_pm")

        if st.session_state.description.strip():
            st.info(f"Mode gợi ý theo mô tả: **{suggest_mode_rule_based(st.session_state.description)}**")

    with right:
        st.subheader("Cost Drivers")
        for group_name, drivers in COST_DRIVER_GROUPS.items():
            with st.expander(group_name, expanded=True):
                for driver in drivers:
                    meta = COST_DRIVERS[driver]
                    options = list(meta["values"].keys())
                    st.selectbox(
                        f"{driver} - {meta['label']}",
                        options,
                        key=driver,
                        help=meta["help"],
                    )

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
    validation_messages = validate_inputs(
        st.session_state.project_name,
        st.session_state.description,
        st.session_state.mode,
        effective_kloc,
        st.session_state.cost_per_pm,
        selections,
        fp_snapshot,
    )

    st.subheader("Estimation Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Effort (PM)", f"{result['effort']:.2f}")
    m2.metric("Time (Months)", f"{result['tdev']:.2f}")
    m3.metric("Team Size", f"{result['staff']:.2f}")
    m4.metric("Cost", format_currency_short(result["cost"]))
    m5.metric("Risk Level", result["risk_level"])

    st.caption(f"Effective KLOC used for calculation: {effective_kloc:.3f}")
    st.caption(f"Full estimated cost: {result['cost']:,.0f} VND")

    if validation_messages:
        st.warning(f"Hệ thống phát hiện {len(validation_messages)} cảnh báo logic. Xem tab Validation để kiểm tra chi tiết.")
    else:
        st.success("Không phát hiện cảnh báo logic đáng chú ý.")

    driver_df = build_driver_df(selections)
    top5_df = get_top_impact_drivers(driver_df)

    top_col1, top_col2 = st.columns([1.2, 1])

    with top_col1:
        fig1 = px.bar(
            driver_df,
            x="Driver",
            y="Multiplier",
            color="Group",
            hover_data=["Name", "Rating"],
            title="Cost Driver Multipliers",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with top_col2:
        st.subheader("Top 5 Driver ảnh hưởng mạnh")
        st.dataframe(
            top5_df[["Driver", "Name", "Rating", "Multiplier", "Group"]],
            use_container_width=True,
        )

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
            compare_rows.append({
                "Mode": each_mode,
                "Effort": mode_result["effort"],
                "Development Time": mode_result["tdev"],
                "Average Staff": mode_result["staff"],
            })
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

    report_text = export_text_report(result, selections, st.session_state.ai_result, fp_snapshot, validation_messages)
    pdf_bytes = make_pdf_report_bytes(
        {
            "project_name": st.session_state.project_name,
            "project_owner": st.session_state.project_owner,
            "project_status": st.session_state.project_status,
            "project_domain": st.session_state.project_domain,
            "description": st.session_state.description,
        },
        result,
        selections,
        fp_snapshot,
        validation_messages,
        st.session_state.ai_result,
    )

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
                    st.warning("AI tạm thời không khả dụng. Hệ thống đang hiển thị phân tích fallback.")
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
            payload = build_current_version_payload(result, selections, fp_snapshot, validation_messages)
            if not st.session_state.active_project_id:
                create_new_project_with_version(payload)
                st.success("Đã tạo project mới và lưu version đầu tiên.")
            else:
                save_current_as_new_version(payload)
                st.success("Đã lưu version mới cho project hiện tại.")
            save_history_row({
                "timestamp": now_text(),
                "project_name": result["project_name"],
                "project_owner": st.session_state.project_owner,
                "project_status": st.session_state.project_status,
                "mode": result["mode"],
                "kloc": result["kloc"],
                "eaf": result["eaf"],
                "effort_pm": result["effort"],
                "development_time_months": result["tdev"],
                "team_size": result["staff"],
                "estimated_cost_vnd": result["cost"],
                "risk_level": result["risk_level"],
                "version_title": st.session_state.version_title,
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
            st.info("Muốn xuất PDF cần cài thêm reportlab: pip install reportlab")

    st.subheader("AI Project Analysis")
    if st.session_state.ai_result:
        st.markdown(st.session_state.ai_result)
    else:
        st.caption("Chưa có AI analysis. Bấm Generate AI Analysis để tạo.")

with tab2:
    st.subheader("Function Point Estimator")
    st.caption("Nhập 5 nhóm FP, tính UFP / VAF / FP, chuyển sang SLOC/KLOC, phân tích AI bằng tiếng Việt và có thể đẩy sang COCOMO.")

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

    fp_mode_col1, fp_mode_col2 = st.columns(2)
    with fp_mode_col1:
        st.selectbox("Mode for FP-based COCOMO", list(MODES.keys()), key="fp_mode")
    with fp_mode_col2:
        st.number_input("FP-based Cost per Person-Month (VND)", min_value=1000000.0, step=1000000.0, key="fp_cost_per_pm")

    fp_based_result = compute_result(
        project_name=f"{st.session_state.project_name or 'FP Project'} (FP-based)",
        description=st.session_state.description or "Function Point derived estimation",
        mode=st.session_state.fp_mode,
        kloc=max(fp_snapshot["kloc"], 0.001),
        cost_per_pm=st.session_state.fp_cost_per_pm,
        selections=get_driver_selections(),
    )

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
                    st.warning("AI FP tạm thời không khả dụng. Hệ thống đang hiển thị fallback.")
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
        st.caption("Chưa có FP AI analysis. Bấm Generate FP AI Analysis để tạo.")

with tab3:
    st.subheader("AI Optimization")
    st.caption("AI đọc current project, đề xuất thay đổi có cấu trúc, cho phép so sánh Current vs Suggested và Apply.")

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
                st.warning("AI optimization tạm thời không khả dụng. Hệ thống hiển thị fallback suggestion.")
            else:
                st.success("Đã tạo AI optimization.")
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
            st.info("Không có driver change. Có thể chỉ đổi mode.")

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
        st.info("Chưa có AI optimization. Bấm Generate AI Optimization để tạo.")

with tab4:
    st.subheader("Validation & Logic Warnings")
    current_selections = get_driver_selections()
    current_result = compute_result(
        project_name=st.session_state.project_name,
        description=st.session_state.description,
        mode=st.session_state.mode,
        kloc=get_effective_kloc(),
        cost_per_pm=st.session_state.cost_per_pm,
        selections=current_selections,
    )
    fp_snapshot = get_fp_snapshot()
    validation_messages = validate_inputs(
        st.session_state.project_name,
        st.session_state.description,
        st.session_state.mode,
        current_result["kloc"],
        st.session_state.cost_per_pm,
        current_selections,
        fp_snapshot,
    )

    st.dataframe(build_validation_df(validation_messages), use_container_width=True)

    if validation_messages:
        for msg in validation_messages:
            st.warning(msg)
    else:
        st.success("Không có cảnh báo logic đáng chú ý.")

with tab5:
    st.subheader("Workspace, Versioning, JSON Package, Baseline")
    refresh_workspace()

    ws_col1, ws_col2 = st.columns([1, 1.2])

    with ws_col1:
        st.markdown("### Import Project Package JSON")
        uploaded_json = st.file_uploader("Chọn file JSON package", type=["json"])
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
        st.info("Chưa có project nào trong workspace.")
    else:
        for project in st.session_state.workspace.get("projects", []):
            with st.expander(f"{project['project_name']} - {project['project_id']}", expanded=False):
                st.write(f"**Owner:** {project['project_owner']}")
                st.write(f"**Status:** {project['project_status']}")
                st.write(f"**Domain:** {project['project_domain']}")
                st.write(f"**Description:** {project['description']}")
                st.write(f"**Created:** {project['created_at']}")
                st.write(f"**Updated:** {project['updated_at']}")

                versions_df = project_versions_df(project)
                st.dataframe(versions_df, use_container_width=True)

                version_ids = [v["version_id"] for v in project.get("versions", [])]
                selected_ver_for_action = st.selectbox(
                    f"Chọn version thao tác - {project['project_id']}",
                    version_ids,
                    key=f"action_version_{project['project_id']}"
                )

                act1, act2, act3 = st.columns(3)

                with act1:
                    if st.button(f"Set Baseline - {project['project_id']}", use_container_width=True):
                        if set_baseline(project["project_id"], selected_ver_for_action):
                            st.success("Đã đặt baseline.")
                            st.rerun()

                with act2:
                    if st.button(f"Load Form - {project['project_id']}", use_container_width=True):
                        load_version_to_form(project["project_id"], selected_ver_for_action)
                        st.success("Đã nạp version lên form.")
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
                    fig_baseline = px.bar(
                        baseline_chart_df,
                        x="Metric",
                        y="Value",
                        color="Scenario",
                        barmode="group",
                        title="Baseline vs Current"
                    )
                    st.plotly_chart(fig_baseline, use_container_width=True)
                else:
                    st.info("Project hiện tại chưa có baseline.")

    st.subheader("History CSV")
    history_df = load_history_df()
    if history_df.empty:
        st.info("Chưa có history CSV.")
    else:
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "Download History CSV",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="estimation_history.csv",
            mime="text/csv",
        )

with tab6:
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