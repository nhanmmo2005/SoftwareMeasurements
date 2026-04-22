
import os
import io
import json
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
    import plotly.io as pio
except Exception:
    pio = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

load_dotenv()

st.set_page_config(
    page_title="Professional Software Estimation Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] div.stButton > button {
    height: 42px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

APP_SCHEMA_VERSION = "3.1.0"
AI_MODEL_NAME = "gpt-4.1-mini"

MODES = {
    "Organic": {"a": 3.2, "b": 1.05, "c": 2.5, "d": 0.38},
    "Semi-detached": {"a": 3.0, "b": 1.12, "c": 2.5, "d": 0.35},
    "Embedded": {"a": 2.8, "b": 1.20, "c": 2.5, "d": 0.32},
}

COST_DRIVER_GROUPS = {
    "Product Attributes": ["RELY", "DATA", "CPLX"],
    "Hardware Attributes": ["TIME", "STOR", "VIRT", "TURN"],
    "Personnel Attributes": ["ACAP", "PCAP", "AEXP", "PEXP", "LTEX"],
    "Project Attributes": ["PCON", "TOOL", "SCED"],
}

COST_DRIVERS = {
    "RELY": {
        "label": "Required Software Reliability",
        "help": "Required level of software reliability.",
        "values": {"Very Low": 0.75, "Low": 0.88, "Nominal": 1.00, "High": 1.15, "Very High": 1.40},
    },
    "DATA": {
        "label": "Database Size",
        "help": "Amount of data the system must process.",
        "values": {"Low": 0.94, "Nominal": 1.00, "High": 1.08, "Very High": 1.16},
    },
    "CPLX": {
        "label": "Product Complexity",
        "help": "Overall product complexity.",
        "values": {"Very Low": 0.70, "Low": 0.85, "Nominal": 1.00, "High": 1.15, "Very High": 1.30, "Extra High": 1.65},
    },
    "TIME": {
        "label": "Execution Time Constraint",
        "help": "Execution time constraints.",
        "values": {"Nominal": 1.00, "High": 1.11, "Very High": 1.30, "Extra High": 1.66},
    },
    "STOR": {
        "label": "Main Storage Constraint",
        "help": "Main memory or storage constraints.",
        "values": {"Nominal": 1.00, "High": 1.06, "Very High": 1.21, "Extra High": 1.56},
    },
    "VIRT": {
        "label": "Virtual Machine Volatility",
        "help": "Volatility of the hardware or operating environment.",
        "values": {"Low": 0.87, "Nominal": 1.00, "High": 1.15, "Very High": 1.30},
    },
    "TURN": {
        "label": "Computer Turnaround Time",
        "help": "Turnaround time of the computing environment.",
        "values": {"Low": 0.87, "Nominal": 1.00, "High": 1.07, "Very High": 1.15},
    },
    "ACAP": {
        "label": "Analyst Capability",
        "help": "Capability of the systems analyst.",
        "values": {"Very Low": 1.46, "Low": 1.19, "Nominal": 1.00, "High": 0.86, "Very High": 0.71},
    },
    "PCAP": {
        "label": "Programmer Capability",
        "help": "Capability of the programmers.",
        "values": {"Very Low": 1.42, "Low": 1.17, "Nominal": 1.00, "High": 0.86, "Very High": 0.70},
    },
    "AEXP": {
        "label": "Application Experience",
        "help": "Application domain experience.",
        "values": {"Very Low": 1.29, "Low": 1.13, "Nominal": 1.00, "High": 0.91, "Very High": 0.82},
    },
    "PEXP": {
        "label": "Platform Experience",
        "help": "Experience with the target platform.",
        "values": {"Very Low": 1.21, "Low": 1.10, "Nominal": 1.00, "High": 0.90},
    },
    "LTEX": {
        "label": "Language and Tool Experience",
        "help": "Experience with the language and tools in use.",
        "values": {"Very Low": 1.14, "Low": 1.07, "Nominal": 1.00, "High": 0.95},
    },
    "PCON": {
        "label": "Personnel Continuity",
        "help": "Personnel continuity on the project.",
        "values": {"Very Low": 1.29, "Low": 1.12, "Nominal": 1.00, "High": 0.90, "Very High": 0.81},
    },
    "TOOL": {
        "label": "Use of Software Tools",
        "help": "Support level from IDEs, test tools, automation, and CI/CD.",
        "values": {"Very Low": 1.24, "Low": 1.10, "Nominal": 1.00, "High": 0.91, "Very High": 0.83},
    },
    "SCED": {
        "label": "Required Development Schedule",
        "help": "Required development schedule pressure.",
        "values": {"Very Low": 1.23, "Low": 1.08, "Nominal": 1.00, "High": 1.04, "Very High": 1.10},
    },
}

PRESET_PROJECTS = {
    "Start from Scratch": None,
    "School Management System": {
        "project_name": "School Management System",
        "description": "A school management system with student records, teachers, attendance, score reports and academic administration.",
        "mode": "Organic",
        "kloc": 18.0,
        "cost_per_pm": 12000000.0,
        "drivers": {
            "RELY": "Nominal", "DATA": "Nominal", "CPLX": "Nominal",
            "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal",
            "ACAP": "Nominal", "PCAP": "Nominal", "AEXP": "Nominal", "PEXP": "Nominal", "LTEX": "Nominal",
            "PCON": "Nominal", "TOOL": "High", "SCED": "Nominal",
        },
    },
    "Hotel Management System": {
        "project_name": "Hotel Management System",
        "description": "A medium hotel management system with booking, billing, rooms, services, reports and staff administration.",
        "mode": "Semi-detached",
        "kloc": 30.0,
        "cost_per_pm": 12000000.0,
        "drivers": {
            "RELY": "High", "DATA": "Nominal", "CPLX": "High",
            "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal",
            "ACAP": "Nominal", "PCAP": "Nominal", "AEXP": "Nominal", "PEXP": "Nominal", "LTEX": "Nominal",
            "PCON": "Nominal", "TOOL": "Nominal", "SCED": "High",
        },
    },
    "E-commerce Platform": {
        "project_name": "E-commerce Platform",
        "description": "An e-commerce platform with catalog, orders, payments, inventory, analytics and admin control.",
        "mode": "Semi-detached",
        "kloc": 45.0,
        "cost_per_pm": 13000000.0,
        "drivers": {
            "RELY": "High", "DATA": "High", "CPLX": "High",
            "TIME": "Nominal", "STOR": "Nominal", "VIRT": "Nominal", "TURN": "Nominal",
            "ACAP": "Nominal", "PCAP": "Nominal", "AEXP": "Low", "PEXP": "Low", "LTEX": "Nominal",
            "PCON": "Nominal", "TOOL": "High", "SCED": "High",
        },
    },
}

FP_COMPONENT_LABELS = {
    "EI": "External Inputs",
    "EO": "External Outputs",
    "EQ": "External Inquiries",
    "ILF": "Internal Logical Files",
    "EIF": "External Interface Files",
}

FP_MATRIX = {
    "EI": {
        "row_label": "FTR",
        "complexity": [
            ["Low", "Low", "Average"],
            ["Low", "Average", "High"],
            ["Average", "High", "High"],
        ],
    },
    "EO": {
        "row_label": "FTR",
        "complexity": [
            ["Low", "Low", "Average"],
            ["Low", "Average", "High"],
            ["Average", "High", "High"],
        ],
    },
    "EQ": {
        "row_label": "FTR",
        "complexity": [
            ["Low", "Low", "Average"],
            ["Low", "Average", "High"],
            ["Average", "High", "High"],
        ],
    },
    "ILF": {
        "row_label": "RET",
        "complexity": [
            ["Low", "Low", "Average"],
            ["Low", "Average", "High"],
            ["Average", "High", "High"],
        ],
    },
    "EIF": {
        "row_label": "RET",
        "complexity": [
            ["Low", "Low", "Average"],
            ["Low", "Average", "High"],
            ["Average", "High", "High"],
        ],
    },
}

FP_WEIGHTS = {
    "EI": {"Low": 3, "Average": 4, "High": 6},
    "EO": {"Low": 4, "Average": 5, "High": 7},
    "EQ": {"Low": 3, "Average": 4, "High": 6},
    "EIF": {"Low": 5, "Average": 7, "High": 10},
    "ILF": {"Low": 7, "Average": 10, "High": 15},
}

GSC_NAMES = [
    "Data communications", "Distributed data processing", "Performance", "Heavily used configuration",
    "Transaction rate", "Online data entry", "End-user efficiency", "Online update", "Complex processing",
    "Reusability", "Installation ease", "Operational ease", "Multiple sites", "Facilitate change",
]

LANGUAGE_LOC_PER_FP = {
    "3GL Default": 80, "4GL Default": 20, "Assembler": 320, "C": 148, "Basic": 107,
    "Pascal": 90, "C#": 59, "C++": 60, "PL/SQL": 46, "Java": 60, "Visual Basic": 50,
    "Delphi": 18, "HTML": 14, "SQL": 13, "Excel": 47,
}

FP_PRESET_TEMPLATES = {
    "Start FP from Scratch": None,
    "School Management FP Demo": {
        "language": "Java",
        "fp_mode": "Organic",
        "gsc": [2, 1, 3, 2, 3, 4, 3, 4, 3, 2, 2, 3, 2, 3],
        "items": {
            "EI": [
                {"name": "Add Student", "det": 12, "ftr_ret": 2},
                {"name": "Update Student", "det": 10, "ftr_ret": 2},
                {"name": "Record Attendance", "det": 8, "ftr_ret": 2},
                {"name": "Enter Scores", "det": 9, "ftr_ret": 2},
            ],
            "EO": [
                {"name": "Student Report Card", "det": 18, "ftr_ret": 3},
                {"name": "Attendance Summary", "det": 14, "ftr_ret": 2},
            ],
            "EQ": [
                {"name": "Search Student", "det": 8, "ftr_ret": 2},
                {"name": "View Class Schedule", "det": 7, "ftr_ret": 2},
            ],
            "ILF": [
                {"name": "Student File", "det": 35, "ftr_ret": 3},
                {"name": "Attendance File", "det": 24, "ftr_ret": 2},
                {"name": "Score File", "det": 28, "ftr_ret": 3},
            ],
            "EIF": [
                {"name": "Payment Gateway Reference", "det": 18, "ftr_ret": 1},
            ],
        },
    },
    "Hotel Management FP Demo": {
        "language": "Java",
        "fp_mode": "Semi-detached",
        "gsc": [3, 2, 3, 2, 3, 4, 3, 4, 4, 2, 2, 3, 2, 3],
        "items": {
            "EI": [
                {"name": "Create Booking", "det": 12, "ftr_ret": 2},
                {"name": "Check In Guest", "det": 9, "ftr_ret": 2},
                {"name": "Check Out Guest", "det": 10, "ftr_ret": 3},
                {"name": "Record Service Usage", "det": 8, "ftr_ret": 2},
            ],
            "EO": [
                {"name": "Invoice", "det": 18, "ftr_ret": 3},
                {"name": "Occupancy Report", "det": 15, "ftr_ret": 2},
                {"name": "Revenue Report", "det": 17, "ftr_ret": 3},
            ],
            "EQ": [
                {"name": "Room Availability Search", "det": 7, "ftr_ret": 2},
                {"name": "Guest Search", "det": 8, "ftr_ret": 2},
            ],
            "ILF": [
                {"name": "Booking File", "det": 30, "ftr_ret": 3},
                {"name": "Room File", "det": 22, "ftr_ret": 2},
                {"name": "Billing File", "det": 26, "ftr_ret": 3},
            ],
            "EIF": [
                {"name": "External Payment Provider", "det": 16, "ftr_ret": 1},
                {"name": "Travel Agency Feed", "det": 14, "ftr_ret": 1},
            ],
        },
    },
    "E-commerce FP Demo": {
        "language": "Java",
        "fp_mode": "Semi-detached",
        "gsc": [4, 3, 4, 3, 4, 5, 4, 4, 4, 3, 3, 3, 3, 4],
        "items": {
            "EI": [
                {"name": "Create Order", "det": 14, "ftr_ret": 3},
                {"name": "Update Cart", "det": 8, "ftr_ret": 2},
                {"name": "Process Payment", "det": 11, "ftr_ret": 3},
                {"name": "Manage Product", "det": 10, "ftr_ret": 2},
            ],
            "EO": [
                {"name": "Order Confirmation", "det": 16, "ftr_ret": 3},
                {"name": "Sales Dashboard", "det": 20, "ftr_ret": 4},
                {"name": "Inventory Status Report", "det": 17, "ftr_ret": 3},
            ],
            "EQ": [
                {"name": "Product Search", "det": 9, "ftr_ret": 2},
                {"name": "Track Order", "det": 7, "ftr_ret": 2},
            ],
            "ILF": [
                {"name": "Product File", "det": 40, "ftr_ret": 4},
                {"name": "Order File", "det": 34, "ftr_ret": 3},
                {"name": "Customer File", "det": 24, "ftr_ret": 2},
                {"name": "Inventory File", "det": 25, "ftr_ret": 2},
            ],
            "EIF": [
                {"name": "Payment Gateway", "det": 18, "ftr_ret": 1},
                {"name": "Shipping Carrier API", "det": 15, "ftr_ret": 1},
            ],
        },
    },
}


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


def set_form_from_package_data(data: dict):
    version = data.get("version", {})
    project = data.get("project", {})

    updates = {}
    updates["loaded_package"] = data
    updates["loaded_project_name"] = project.get("project_name", "")
    updates["project_name"] = project.get("project_name", "")
    updates["description"] = project.get("description", "")

    inp = version.get("input", {})
    updates["mode"] = inp.get("mode", "Organic")
    updates["size_input_mode"] = inp.get("size_input_mode", "KLOC")
    updates["kloc"] = safe_float(inp.get("kloc", 1.0), 1.0)
    updates["sloc"] = safe_float(inp.get("sloc", updates["kloc"] * 1000.0), updates["kloc"] * 1000.0)
    updates["cost_per_pm"] = safe_float(inp.get("cost_per_pm", 12000000.0), 12000000.0)

    drivers = version.get("cost_drivers", {})
    for driver in COST_DRIVERS:
        if driver in drivers and drivers[driver] in COST_DRIVERS[driver]["values"]:
            updates[driver] = drivers[driver]

    updates["ai_result"] = version.get("ai", {}).get("analysis", "")
    updates["ai_suggestion"] = version.get("ai", {}).get("suggestion", None)

    fp_data = version.get("fp", {})
    updates["fp_language"] = fp_data.get("language", "Java")
    updates["fp_custom_loc_per_fp"] = safe_float(fp_data.get("loc_per_fp", 60.0), 60.0)
    updates["fp_mode"] = fp_data.get("fp_mode", "Organic")
    updates["fp_cost_per_pm"] = safe_float(fp_data.get("fp_cost_per_pm", 12000000.0), 12000000.0)
    updates["ai_fp_result"] = fp_data.get("ai_fp_result", "")

    summary = fp_data.get("summary", {})
    for comp in FP_COMPONENT_LABELS:
        details = summary.get(comp, {}).get("details", [])
        items = []
        row_label = FP_MATRIX[comp]["row_label"]
        for d in details:
            items.append({
                "name": d.get("Name", ""),
                "det": safe_int(d.get("DET", 1), 1),
                "ftr_ret": safe_int(d.get(row_label, 1), 1),
            })
        updates[f"fp_items_{comp}"] = items

    gsc = fp_data.get("gsc", {})
    for idx, name in enumerate(GSC_NAMES):
        updates[f"fp_gsc_{idx}"] = safe_int(gsc.get(name, 0), 0)

    for key, value in updates.items():
        st.session_state[key] = value


def init_state():
    defaults = {
        "selected_preset": "Start from Scratch",
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
        "fp_language": "Java",
        "fp_custom_loc_per_fp": 60.0,
        "fp_mode": "Organic",
        "fp_cost_per_pm": 12000000.0,
        "loaded_package": None,
        "loaded_project_name": "",
        "pending_import_package": None,
        "pending_apply_import": False,
        "selected_fp_preset": "Start FP from Scratch",
        "link_fp_with_estimator": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    for driver, meta in COST_DRIVERS.items():
        if driver not in st.session_state:
            vals = list(meta["values"].keys())
            st.session_state[driver] = "Nominal" if "Nominal" in vals else vals[0]
    for comp in FP_COMPONENT_LABELS:
        if f"fp_items_{comp}" not in st.session_state:
            st.session_state[f"fp_items_{comp}"] = []
    for idx in range(14):
        key = f"fp_gsc_{idx}"
        if key not in st.session_state:
            st.session_state[key] = 0


def reset_form():
    st.session_state.project_name = ""
    st.session_state.description = ""
    st.session_state.ai_result = ""
    st.session_state.ai_suggestion = None
    st.session_state.ai_fp_result = ""
    st.session_state.loaded_package = None
    st.session_state.loaded_project_name = ""
    st.session_state.mode = "Organic"
    st.session_state.size_input_mode = "KLOC"
    st.session_state.kloc = 1.0
    st.session_state.sloc = 1000.0
    st.session_state.cost_per_pm = 12000000.0
    st.session_state.fp_language = "Java"
    st.session_state.fp_custom_loc_per_fp = 60.0
    st.session_state.fp_mode = "Organic"
    st.session_state.fp_cost_per_pm = 12000000.0
    st.session_state.selected_fp_preset = "Start FP from Scratch"
    st.session_state.link_fp_with_estimator = False
    for driver, meta in COST_DRIVERS.items():
        vals = list(meta["values"].keys())
        st.session_state[driver] = "Nominal" if "Nominal" in vals else vals[0]
    for comp in FP_COMPONENT_LABELS:
        st.session_state[f"fp_items_{comp}"] = []
    for idx in range(14):
        st.session_state[f"fp_gsc_{idx}"] = 0


def reset_fp_form():
    st.session_state.fp_language = "Java"
    st.session_state.fp_custom_loc_per_fp = 60.0
    st.session_state.fp_mode = "Organic"
    st.session_state.fp_cost_per_pm = 12000000.0
    st.session_state.ai_fp_result = ""
    for comp in FP_COMPONENT_LABELS:
        st.session_state[f"fp_items_{comp}"] = [{"name": "", "det": 1, "ftr_ret": 1}]
    for idx in range(14):
        st.session_state[f"fp_gsc_{idx}"] = 0


def apply_fp_preset(name: str):
    preset = FP_PRESET_TEMPLATES.get(name)
    if not preset:
        reset_fp_form()
        return
    st.session_state.fp_language = preset.get("language", "Java")
    st.session_state.fp_mode = preset.get("fp_mode", "Organic")
    for comp in FP_COMPONENT_LABELS:
        items = preset.get("items", {}).get(comp, [])
        st.session_state[f"fp_items_{comp}"] = items or [{"name": "", "det": 1, "ftr_ret": 1}]
    gsc_values = preset.get("gsc", [0] * 14)
    for idx in range(14):
        st.session_state[f"fp_gsc_{idx}"] = int(gsc_values[idx]) if idx < len(gsc_values) else 0


def apply_preset(name: str):
    preset = PRESET_PROJECTS.get(name)
    if not preset:
        reset_form()
        return
    st.session_state.project_name = preset["project_name"]
    st.session_state.description = preset["description"]
    st.session_state.mode = preset["mode"]
    st.session_state.size_input_mode = "KLOC"
    st.session_state.kloc = float(preset["kloc"])
    st.session_state.sloc = float(preset["kloc"]) * 1000.0
    st.session_state.cost_per_pm = float(preset["cost_per_pm"])
    for driver, rating in preset["drivers"].items():
        st.session_state[driver] = rating


def process_pending_actions():
    if st.session_state.get("pending_fp_transfer"):
        kloc_value = max(float(st.session_state.get("fp_transfer_kloc", 1.0)), 0.001)
        st.session_state.size_input_mode = "KLOC"
        st.session_state.kloc = kloc_value
        st.session_state.sloc = kloc_value * 1000.0
        st.session_state.pending_fp_transfer = False

    if st.session_state.get("pending_ai_apply"):
        suggestion = st.session_state.get("ai_suggestion")
        if suggestion:
            if suggestion.get("suggested_mode") in MODES:
                st.session_state.mode = suggestion["suggested_mode"]
            for driver, rating in suggestion.get("suggested_changes", {}).items():
                if driver in COST_DRIVERS and rating in COST_DRIVERS[driver]["values"]:
                    st.session_state[driver] = rating
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

    embedded_keywords = [
        "bank", "banking", "medical", "healthcare", "critical", "real-time",
        "iot", "embedded", "avionics", "telecom", "payment gateway",
        "safety", "industrial control", "firmware"
    ]
    semi_keywords = [
        "hotel", "inventory", "management", "e-commerce", "booking", "erp",
        "crm", "portal", "marketplace", "enterprise", "multi-user", "analytics"
    ]
    organic_keywords = [
        "school", "student", "library", "attendance", "simple", "small",
        "internal tool", "personal", "blog", "portfolio", "basic crud"
    ]

    if any(word in text for word in embedded_keywords):
        return "Embedded"
    if any(word in text for word in organic_keywords):
        return "Organic"
    if any(word in text for word in semi_keywords):
        return "Semi-detached"
    return "Organic"


def get_risk_level(eaf: float, selections: dict) -> str:
    high = 0
    if eaf >= 1.25:
        high += 1
    if selections["CPLX"] in ["High", "Very High", "Extra High"]:
        high += 1
    if selections["RELY"] in ["High", "Very High"]:
        high += 1
    if selections["SCED"] in ["High", "Very High"]:
        high += 1
    if selections["ACAP"] in ["Very Low", "Low"] or selections["PCAP"] in ["Very Low", "Low"]:
        high += 1
    if high >= 3:
        return "High"
    if high == 2:
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


def fp_det_bucket(component: str, det: int):
    det = max(int(det), 1)
    if component == "EI":
        return 0 if det <= 4 else 1 if det <= 15 else 2
    if component in ["EO", "EQ"]:
        return 0 if det <= 5 else 1 if det <= 19 else 2
    return 0 if det <= 19 else 1 if det <= 50 else 2


def fp_ftr_ret_bucket(component: str, val: int):
    val = max(int(val), 0)
    if component == "EI":
        return 0 if val <= 1 else 1 if val == 2 else 2
    if component == "EO":
        return 0 if val <= 1 else 1 if val <= 3 else 2
    if component == "EQ":
        return 0 if val <= 1 else 1 if val == 2 else 2
    return 0 if val <= 1 else 1 if val <= 5 else 2


def classify_fp_item(component: str, det: int, ftr_ret: int):
    r = fp_ftr_ret_bucket(component, max(int(ftr_ret), 1))
    c = fp_det_bucket(component, max(int(det), 1))
    return FP_MATRIX[component]["complexity"][r][c]


def get_fp_items(component: str):
    items = st.session_state.get(f"fp_items_{component}", [])
    cleaned = []
    for x in items:
        name = str(x.get("name", "")).strip()
        det = safe_int(x.get("det", 1), 1)
        ftr_ret = safe_int(x.get("ftr_ret", 1), 1)
        if not name and det == 1 and ftr_ret == 1:
            continue
        cleaned.append({"name": name, "det": det, "ftr_ret": ftr_ret})
    return cleaned


def calc_fp_component_summary(component: str):
    items = get_fp_items(component)
    counts = {"Low": 0, "Average": 0, "High": 0}
    detailed_rows = []
    total_ufp = 0
    row_label = FP_MATRIX[component]["row_label"]
    for idx, item in enumerate(items, start=1):
        complexity = classify_fp_item(component, item["det"], item["ftr_ret"])
        weight = FP_WEIGHTS[component][complexity]
        counts[complexity] += 1
        total_ufp += weight
        detailed_rows.append({
            "No": idx,
            "Name": item["name"] or f"{component}-{idx}",
            "DET": item["det"],
            row_label: item["ftr_ret"],
            "Complexity": complexity,
            "Weight": weight
        })
    return {"count": len(items), "counts": counts, "ufp": total_ufp, "details": detailed_rows}


def calculate_vaf():
    di = sum(int(st.session_state[f"fp_gsc_{idx}"]) for idx in range(14))
    return di, 0.65 + 0.01 * di


def get_fp_loc_per_fp():
    if st.session_state.fp_language == "Custom":
        return max(float(st.session_state.fp_custom_loc_per_fp), 1.0)
    return float(LANGUAGE_LOC_PER_FP[st.session_state.fp_language])


def get_fp_snapshot():
    summary = {}
    ufp = 0
    for comp in FP_COMPONENT_LABELS:
        s = calc_fp_component_summary(comp)
        summary[comp] = s
        ufp += s["ufp"]
    di, vaf = calculate_vaf()
    fp_value = ufp * vaf
    loc_per_fp = get_fp_loc_per_fp()
    sloc = fp_value * loc_per_fp
    kloc = sloc / 1000.0
    gsc = {GSC_NAMES[idx]: int(st.session_state[f"fp_gsc_{idx}"]) for idx in range(14)}
    return {
        "language": st.session_state.fp_language,
        "loc_per_fp": loc_per_fp,
        "summary": summary,
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


def validate_inputs(project_name, description, mode, kloc, cost_per_pm, selections, fp_snapshot, compare_fp_with_estimator=False):
    warnings = []
    if not project_name.strip():
        warnings.append("Project Name is empty.")
    if len((description or "").strip()) < 15:
        warnings.append("Project Description is too short.")
    if kloc < 1:
        warnings.append("KLOC is very small. Please re-check the selected size unit.")
    if kloc > 1000:
        warnings.append("KLOC is very large. Please re-check KLOC versus SLOC.")
    if cost_per_pm < 3000000:
        warnings.append("Cost per Person-Month is quite low and may be unrealistic.")

    suggested_mode = suggest_mode_rule_based(description)
    if description.strip() and mode != suggested_mode:
        warnings.append(f"The current mode is {mode}, but the description suggests {suggested_mode}.")

    if selections["RELY"] in ["High", "Very High"] and mode == "Organic":
        warnings.append("Reliability is high, but the selected development mode is still Organic.")
    if selections["TIME"] in ["Very High", "Extra High"] and mode == "Organic":
        warnings.append("Execution time constraint is very high, but the selected development mode is still Organic.")
    if selections["SCED"] in ["High", "Very High"] and selections["TOOL"] in ["Very Low", "Low", "Nominal"]:
        warnings.append("The schedule is tight while tool support is still limited.")
    if selections["ACAP"] in ["Very Low", "Low"] and selections["PCAP"] in ["Very Low", "Low"]:
        warnings.append("Both analyst capability and programmer capability are low.")

    if compare_fp_with_estimator and fp_snapshot["fp"] > 0 and kloc > 0:
        diff_pct = abs(fp_snapshot["kloc"] - kloc) / kloc
        if diff_pct >= 0.5:
            warnings.append(
                f"Manual KLOC ({kloc:.2f}) is significantly different from FP-derived KLOC ({fp_snapshot['kloc']:.2f})."
            )
    return warnings


def render_inline_warnings(messages):
    for msg in messages:
        st.warning(msg)


def fallback_ai_summary(description, mode, kloc, eaf, effort, tdev, staff, cost, selections, fp_snapshot):
    risks, actions = [], []
    if selections["CPLX"] in ["High", "Very High", "Extra High"]:
        risks.append("High product complexity can significantly increase effort.")
        actions.append("Break the system into smaller modules, lock scope per phase, and separate core features from optional features.")
    if selections["RELY"] in ["High", "Very High"]:
        risks.append("High reliability requirements increase the amount of testing, review, and verification work.")
        actions.append("Add more test cases, peer reviews, regression testing, and release controls.")
    if selections["TIME"] in ["Very High", "Extra High"]:
        risks.append("Strong runtime constraints usually require architectural optimization and performance testing.")
        actions.append("Design for performance early and measure it throughout the project instead of waiting until the end.")
    if selections["SCED"] in ["High", "Very High"]:
        risks.append("A tight schedule increases the risk of delay or reduced quality.")
        actions.append("Prioritize core functions, reduce non-essential features, and control late change requests.")
    if selections["PCON"] in ["Very Low", "Low"]:
        risks.append("Low personnel continuity reduces project stability.")
        actions.append("Standardize documentation, coding conventions, and internal handover practices.")
    if not risks:
        risks.append("The current configuration does not show any major high-risk signal.")
    if not actions:
        actions.append("Keep the current configuration and continue monitoring the difference between manual KLOC and FP-derived KLOC.")
    fp_note = ""
    if fp_snapshot["fp"] > 0:
        diff = abs(fp_snapshot["kloc"] - kloc)
        fp_note = f"\nFP cross-check: the current FP setup gives about {fp_snapshot['fp']:.2f} FP, equivalent to {fp_snapshot['kloc']:.2f} KLOC. The difference from manual KLOC is {diff:.2f} KLOC."

    return f"""
### 1. Overall Assessment
The project is currently estimated under **{mode} mode** with a size of **{kloc:.2f} KLOC**.
After applying all selected cost drivers, the effort adjustment factor is **EAF = {eaf:.3f}**.

### 2. Estimation Results
- Effort: **{effort:.2f} person-months**
- Development time: **{tdev:.2f} months**
- Average team size: **{staff:.2f} people**
- Estimated cost: **{cost:,.0f} VND**
- Risk level: **{get_risk_level(eaf, selections)}**

### 3. Why Effort and Cost Reach This Level
Effort is mainly driven by three factors:
1. **Software size**: larger KLOC increases effort non-linearly.
2. **Development mode**: Organic, Semi-detached, and Embedded use different coefficients.
3. **EAF**: selected cost drivers increase or decrease effort compared with the baseline.

The mode suggested by the current description is **{suggest_mode_rule_based(description)}**.{fp_note}

### 4. Main Risks
{chr(10).join([f'- {x}' for x in risks])}

### 5. Recommendations for the Project Manager
{chr(10).join([f'- {x}' for x in actions])}

### 6. Conclusion
This estimate can be used as a planning baseline and as a reference for comparing alternative project scenarios.
""".strip()


def fallback_fp_ai_summary(fp_data: dict, fp_based_result: dict):
    comp_lines = [f"- {comp}: {info['count']} item(s), UFP = {info['ufp']}" for comp, info in fp_data["summary"].items()]
    note = "The functional size is small." if fp_data["fp"] < 50 else ("The functional size is medium." if fp_data["fp"] < 200 else "The functional size is large.")
    return f"""
### 1. FP Overview
- UFP: **{fp_data['ufp']:.2f}**
- DI: **{fp_data['di']}**
- VAF: **{fp_data['vaf']:.2f}**
- FP: **{fp_data['fp']:.2f}**
- LOC/FP: **{fp_data['loc_per_fp']:.2f}**
- KLOC derived from FP: **{fp_data['kloc']:.2f}**

### 2. FP Components
{chr(10).join(comp_lines)}

### 3. Remarks
- {note}
- VAF reflects the adjustment created by the 14 general system characteristics.
- If FP-derived KLOC is far from manual KLOC, the scope or the DET/FTR/RET classification should be reviewed.

### 4. Projection to COCOMO
- Effort: **{fp_based_result['effort']:.2f} PM**
- Time: **{fp_based_result['tdev']:.2f} months**
- Team size: **{fp_based_result['staff']:.2f}**
- Cost: **{fp_based_result['cost']:,.0f} VND**
""".strip()


def call_ai_analysis(description, mode, kloc, eaf, effort, tdev, staff, cost, selections, fp_snapshot):
    api_key = get_openai_api_key()
    if not api_key or OpenAI is None:
        return None, "AI unavailable"
    try:
        client = OpenAI(api_key=api_key)
        driver_text = "\n".join([f"{k}: {v} (x{COST_DRIVERS[k]['values'][v]})" for k, v in selections.items()])
        prompt = f"""
You are a software project estimation expert.

Requirements:
- Write in English
- Use a professional technical report style
- Do not use first-person pronouns
- Do not address the reader directly
- Do not use assistant-style sentences
- Keep the explanation clear, structured, and suitable for a student project report

Project description:
{description}

Mode: {mode}
KLOC: {kloc:.2f}
EAF: {eaf:.3f}
Effort: {effort:.2f}
Time: {tdev:.2f}
Team size: {staff:.2f}
Cost: {cost:,.0f} VND

Cost drivers:
{driver_text}

FP information:
UFP={fp_snapshot['ufp']:.2f}, VAF={fp_snapshot['vaf']:.2f}, FP={fp_snapshot['fp']:.2f}, FP-derived KLOC={fp_snapshot['kloc']:.2f}

Write exactly in these 6 sections:
1. Overall Assessment
2. Estimation Results
3. Why Effort and Cost Reach This Level
4. Main Risks
5. Recommendations for the Project Manager
6. Conclusion
"""
        response = client.responses.create(model=AI_MODEL_NAME, input=prompt)
        text = response.output_text.strip()
        text = re.sub(r"(?im)^.*if needed.*$", "", text).strip()
        return (text or None), None if text else "AI unavailable"
    except Exception:
        return None, "AI unavailable"


def call_ai_fp_analysis(fp_data: dict, fp_based_result: dict):
    api_key = get_openai_api_key()
    if not api_key or OpenAI is None:
        return None, "AI unavailable"
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
You are a Function Point analysis assistant.
Data:
UFP={fp_data['ufp']:.2f}
DI={fp_data['di']}
VAF={fp_data['vaf']:.2f}
FP={fp_data['fp']:.2f}
Language={fp_data['language']}
LOC/FP={fp_data['loc_per_fp']:.2f}
SLOC={fp_data['sloc']:,.0f}
KLOC={fp_data['kloc']:.2f}

COCOMO results:
Mode={fp_based_result['mode']}
Effort={fp_based_result['effort']:.2f}
Time={fp_based_result['tdev']:.2f}
Staff={fp_based_result['staff']:.2f}
Cost={fp_based_result['cost']:,.0f} VND

Write in English using these 4 sections:
1. FP Overview
2. FP Components
3. Remarks
4. Projection to COCOMO
"""
        response = client.responses.create(model=AI_MODEL_NAME, input=prompt)
        text = response.output_text.strip()
        return (text or None), None if text else "AI unavailable"
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
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
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
    if isinstance(data.get("goals"), list):
        suggestion["goals"] = [str(x) for x in data["goals"]][:5]
    if isinstance(data.get("reasoning"), list):
        suggestion["reasoning"] = [str(x) for x in data["reasoning"]][:6]
    suggestion["expected_effect"] = str(data.get("expected_effect", ""))
    return suggestion


def fallback_ai_optimization(current_mode: str, selections: dict):
    changes = {}
    if selections["TOOL"] in ["Very Low", "Low", "Nominal"]:
        changes["TOOL"] = "High"
    if selections["PCON"] in ["Very Low", "Low", "Nominal"]:
        changes["PCON"] = "High"
    if selections["SCED"] in ["High", "Very High"]:
        changes["SCED"] = "Nominal"
    if selections["ACAP"] in ["Very Low", "Low"]:
        changes["ACAP"] = "High"
    if selections["PCAP"] in ["Very Low", "Low"]:
        changes["PCAP"] = "High"
    return {
        "suggested_mode": current_mode,
        "suggested_changes": changes,
        "goals": ["Reduce effort", "Reduce cost", "Reduce risk"],
        "reasoning": [
            "Better tool support can improve productivity.",
            "Better personnel continuity reduces rework and handover overhead.",
            "Reducing schedule pressure makes the estimate more realistic.",
        ],
        "expected_effect": "The suggested scenario is likely to reduce effort and risk compared with the current configuration.",
    }


def call_ai_optimization(project_name: str, description: str, current_result: dict, selections: dict):
    api_key = get_openai_api_key()
    fallback = fallback_ai_optimization(current_result["mode"], selections)
    if not api_key or OpenAI is None:
        return fallback, "AI unavailable"
    try:
        client = OpenAI(api_key=api_key)
        driver_text = "\n".join([f"{k}: {v} (x{COST_DRIVERS[k]['values'][v]})" for k, v in selections.items()])
        prompt = f"""
You are a software estimation optimization assistant.
Goals:
- reduce effort
- reduce cost
- reduce risk
- keep suggestions realistic
Do not change KLOC.
Change at most 5 cost drivers.

Project name: {project_name}
Description: {description}
Mode: {current_result['mode']}
KLOC: {current_result['kloc']:.2f}
EAF: {current_result['eaf']:.3f}
Effort: {current_result['effort']:.2f}
Time: {current_result['tdev']:.2f}
Cost: {current_result['cost']:,.0f} VND
Risk: {current_result['risk_level']}

Current cost drivers:
{driver_text}

Return valid JSON only:
{{
  "suggested_mode": "...",
  "suggested_changes": {{"TOOL": "High"}},
  "goals": ["..."],
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
    new_selections = current_selections.copy()
    for driver, rating in suggestion.get("suggested_changes", {}).items():
        if driver in COST_DRIVERS and rating in COST_DRIVERS[driver]["values"]:
            new_selections[driver] = rating
    new_result = compute_result(current_result["project_name"], current_result["description"], suggested_mode, current_result["kloc"], current_result["cost_per_pm"], new_selections)
    return new_selections, new_result


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
        cr, sg = current_selections[driver], suggested_selections[driver]
        if cr != sg:
            rows.append({"Driver": driver, "Current": cr, "Suggested": sg, "Current Multiplier": COST_DRIVERS[driver]["values"][cr], "Suggested Multiplier": COST_DRIVERS[driver]["values"][sg]})
    return pd.DataFrame(rows)


def build_current_package(result: dict, selections: dict, fp_snapshot: dict, validation_messages: list):
    return {
        "schema_version": APP_SCHEMA_VERSION,
        "exported_at": now_text(),
        "project": {"project_id": generate_id("PRJ"), "project_name": st.session_state.project_name or "Untitled Project", "description": st.session_state.description},
        "version": {
            "version_id": generate_id("VER"),
            "created_at": now_text(),
            "input": {"mode": st.session_state.mode, "size_input_mode": st.session_state.size_input_mode, "kloc": result["kloc"], "sloc": result["kloc"] * 1000.0, "cost_per_pm": st.session_state.cost_per_pm},
            "cost_drivers": deepcopy(selections),
            "cost_driver_multipliers": {d: COST_DRIVERS[d]["values"][r] for d, r in selections.items()},
            "result": deepcopy(result),
            "validation_messages": deepcopy(validation_messages),
            "ai": {"analysis": st.session_state.ai_result, "suggestion": deepcopy(st.session_state.ai_suggestion)},
            "fp": deepcopy(fp_snapshot),
        },
    }


def import_project_package(uploaded_file):
    try:
        data = json.load(uploaded_file)
    except Exception:
        return False, "The uploaded JSON file is invalid."
    if not isinstance(data, dict) or "project" not in data or "version" not in data:
        return False, "The JSON file is missing project or version data."

    st.session_state.pending_import_package = data
    st.session_state.pending_apply_import = True
    return True, f"Project package received: {data.get('project', {}).get('project_name', 'Imported Project')}"


def fig_to_png_bytes(fig, width=1200, height=700, scale=2):
    if pio is None:
        return None
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


def make_pdf_report_bytes(project_meta: dict, result: dict, selections: dict, fp_snapshot: dict, validation_messages: list):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=40,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.white,
        backColor=colors.HexColor("#1D4ED8"),
        spaceBefore=8,
        spaceAfter=8,
        leftIndent=6
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#111827"),
        spaceAfter=4
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=normal_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=3
    )

    def hline():
        tbl = Table([[""]], colWidths=[doc.width])
        tbl.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1"))
        ]))
        return tbl

    def make_kv_table(rows, col_widths):
        tbl = Table(rows, colWidths=col_widths, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return tbl

    def text_to_paragraphs(text: str):
        text = (text or "").strip()
        if not text:
            return [Paragraph("No AI analysis generated.", normal_style)]

        parts = []
        for block in text.split("\n"):
            line = block.strip()
            if not line:
                parts.append(Spacer(1, 4))
                continue

            if line.startswith("- "):
                safe_line = line[2:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(Paragraph(safe_line, bullet_style, bulletText="•"))
            elif re.match(r"^\d+\.", line):
                safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(Paragraph(f"<b>{safe_line}</b>", normal_style))
            elif line.startswith("### "):
                safe_line = line[4:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(Paragraph(f"<b>{safe_line}</b>", normal_style))
            elif line.startswith("## "):
                safe_line = line[3:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(Paragraph(f"<b>{safe_line}</b>", normal_style))
            elif line.startswith("# "):
                safe_line = line[2:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(Paragraph(f"<b>{safe_line}</b>", normal_style))
            else:
                safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(Paragraph(safe_line, normal_style))
        return parts

    def build_mode_comparison_chart():
        rows = []
        for each_mode in MODES:
            mode_result = compute_result(
                result["project_name"],
                result["description"],
                each_mode,
                result["kloc"],
                result["cost_per_pm"],
                selections
            )
            rows.append({
                "Mode": each_mode,
                "Effort": round(mode_result["effort"], 2)
            })
        df = pd.DataFrame(rows)
        fig = px.bar(df, x="Mode", y="Effort", title="Effort Comparison by Development Mode")
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            margin=dict(l=30, r=20, t=60, b=30),
            height=380
        )
        return fig

    def build_cost_driver_chart():
        driver_df = build_driver_df(selections)
        fig = px.bar(
            driver_df,
            x="Driver",
            y="Multiplier",
            color="Group",
            hover_data=["Name", "Rating"],
            title="Cost Driver Multiplier Analysis"
        )
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            margin=dict(l=30, r=20, t=60, b=30),
            height=400
        )
        return fig

    def build_cost_breakdown_chart():
        cost_breakdown_df = pd.DataFrame([
            {"Category": "Development", "Cost": result["cost"] * 0.55},
            {"Category": "Testing and QA", "Cost": result["cost"] * 0.20},
            {"Category": "Management", "Cost": result["cost"] * 0.15},
            {"Category": "Tools and Infrastructure", "Cost": result["cost"] * 0.10},
        ])
        fig = px.pie(
            cost_breakdown_df,
            names="Category",
            values="Cost",
            title="Estimated Cost Distribution"
        )
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            margin=dict(l=20, r=20, t=60, b=20),
            height=400
        )
        return fig

    def on_page(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(doc.leftMargin, A4[1] - 20, "Software Effort Estimation Report")
        canvas.drawRightString(A4[0] - doc.rightMargin, 18, f"Page {doc_obj.page}")
        canvas.restoreState()

    story = []

    project_name = project_meta.get("project_name", "") or "Untitled Project"
    project_description = project_meta.get("description", "") or "No description provided."
    ai_text = st.session_state.get("ai_result", "") or "No AI analysis generated."

    story.append(Paragraph("Software Effort Estimation Report", title_style))
    story.append(Paragraph("Professional project estimation summary generated from COCOMO, Function Point, and AI-based analysis.", subtitle_style))
    story.append(hline())
    story.append(Spacer(1, 10))

    overview_rows = [
        ["Field", "Value"],
        ["Project Name", project_name],
        ["Generated At", now_text()],
        ["Development Mode", result["mode"]],
        ["Estimated Size", f"{result['kloc']:.2f} KLOC"],
        ["Effort", f"{result['effort']:.2f} person-months"],
        ["Schedule", f"{result['tdev']:.2f} months"],
        ["Average Team Size", f"{result['staff']:.2f} people"],
        ["Estimated Cost", f"{result['cost']:,.0f} VND"],
        ["Risk Level", result["risk_level"]],
        ["EAF", f"{result['eaf']:.3f}"],
    ]
    story.append(make_kv_table(overview_rows, [150, 360]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Executive Summary", section_style))
    summary_text = (
        f"The current project is estimated in <b>{result['mode']}</b> mode with an effective size of "
        f"<b>{result['kloc']:.2f} KLOC</b>. The projected effort is <b>{result['effort']:.2f} person-months</b>, "
        f"with an estimated development time of <b>{result['tdev']:.2f} months</b> and an average staffing level of "
        f"<b>{result['staff']:.2f}</b>. The total estimated budget is <b>{result['cost']:,.0f} VND</b>. "
        f"Based on the current cost driver configuration, the project risk level is assessed as <b>{result['risk_level']}</b>."
    )
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Project Description", section_style))
    safe_desc = project_description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    story.append(Paragraph(safe_desc, normal_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Estimation Results", section_style))
    result_rows = [
        ["Metric", "Value"],
        ["Development Mode", result["mode"]],
        ["KLOC", f"{result['kloc']:.2f}"],
        ["EAF", f"{result['eaf']:.3f}"],
        ["Effort (PM)", f"{result['effort']:.2f}"],
        ["Development Time (Months)", f"{result['tdev']:.2f}"],
        ["Average Team Size", f"{result['staff']:.2f}"],
        ["Estimated Cost", f"{result['cost']:,.0f} VND"],
        ["Risk Level", result["risk_level"]],
    ]
    story.append(make_kv_table(result_rows, [220, 290]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4. Validation and Input Review", section_style))
    if validation_messages:
        for msg in validation_messages:
            safe_msg = str(msg).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_msg, bullet_style, bulletText="•"))
    else:
        story.append(Paragraph("No major input validation issues were detected.", normal_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. Cost Driver Configuration", section_style))
    driver_rows = [["Driver", "Description", "Rating", "Multiplier"]]
    for driver, rating in selections.items():
        driver_rows.append([
            driver,
            COST_DRIVERS[driver]["label"],
            rating,
            f"{COST_DRIVERS[driver]['values'][rating]:.2f}"
        ])
    story.append(make_kv_table(driver_rows, [55, 265, 100, 80]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6. Function Point Summary", section_style))
    fp_rows = [
        ["Metric", "Value"],
        ["Programming Language", fp_snapshot["language"]],
        ["LOC per FP", f"{fp_snapshot['loc_per_fp']:.2f}"],
        ["UFP", f"{fp_snapshot['ufp']:.2f}"],
        ["DI", f"{fp_snapshot['di']}"],
        ["VAF", f"{fp_snapshot['vaf']:.2f}"],
        ["FP", f"{fp_snapshot['fp']:.2f}"],
        ["Estimated SLOC", f"{fp_snapshot['sloc']:,.0f}"],
        ["Estimated KLOC", f"{fp_snapshot['kloc']:.2f}"],
    ]
    story.append(make_kv_table(fp_rows, [220, 290]))
    story.append(Spacer(1, 6))

    fp_component_rows = [["Component", "Count", "Low", "Average", "High", "UFP"]]
    for comp, data in fp_snapshot["summary"].items():
        fp_component_rows.append([
            comp,
            str(data["count"]),
            str(data["counts"]["Low"]),
            str(data["counts"]["Average"]),
            str(data["counts"]["High"]),
            str(data["ufp"]),
        ])
    story.append(make_kv_table(fp_component_rows, [90, 60, 60, 75, 60, 75]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("7. Visual Analysis", section_style))
    charts = [
        build_cost_driver_chart(),
        build_mode_comparison_chart(),
        build_cost_breakdown_chart()
    ]

    for fig in charts:
        png = fig_to_png_bytes(fig, width=1400, height=820, scale=2)
        if png:
            story.append(Image(io.BytesIO(png), width=520, height=300))
            story.append(Spacer(1, 10))

    story.append(Paragraph("8. AI Project Analysis", section_style))
    for item in text_to_paragraphs(ai_text):
        story.append(item)

    story.append(Spacer(1, 8))
    story.append(hline())
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "End of report. This document summarizes the current project estimation state based on the selected inputs, cost drivers, FP parameters, and AI-assisted analysis.",
        small_style
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    pdf = buffer.getvalue()
    buffer.close()
    return pdf


init_state()

if st.session_state.get("pending_apply_import") and st.session_state.get("pending_import_package"):
    set_form_from_package_data(st.session_state.pending_import_package)
    st.session_state.pending_import_package = None
    st.session_state.pending_apply_import = False

process_pending_actions()

with st.sidebar:
    st.title("Professional Estimation Workspace")
    preset = st.selectbox(
        "Project Template",
        list(PRESET_PROJECTS.keys()),
        index=list(PRESET_PROJECTS.keys()).index(st.session_state.selected_preset)
        if st.session_state.selected_preset in PRESET_PROJECTS else 0
    )

    st.markdown("### Actions")
    if st.button("Apply Template", use_container_width=True, type="primary"):
        apply_preset(preset)
        st.rerun()

    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

    if st.button("Reset Form", use_container_width=True):
        reset_form()
        st.rerun()

    enable_ai = st.checkbox("Enable AI", value=True)

    st.markdown("---")
    st.subheader("JSON Package")

    if st.session_state.project_name.strip():
        current_sel = get_driver_selections()
        current_result = compute_result(
            st.session_state.project_name,
            st.session_state.description,
            st.session_state.mode,
            get_effective_kloc(),
            st.session_state.cost_per_pm,
            current_sel
        )
        fp_snapshot_export = get_fp_snapshot()
        validation_export = validate_inputs(
            st.session_state.project_name,
            st.session_state.description,
            st.session_state.mode,
            get_effective_kloc(),
            st.session_state.cost_per_pm,
            current_sel,
            fp_snapshot_export,
            st.session_state.link_fp_with_estimator
        )
        package = build_current_package(current_result, current_sel, fp_snapshot_export, validation_export)
        st.download_button(
            "Export JSON Package",
            data=json.dumps(package, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{(st.session_state.project_name or 'project').replace(' ', '_').lower()}_package.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.info("Enter Project Name to enable project export.")

    if st.session_state.loaded_project_name:
        st.success(f"Continuing from JSON package: {st.session_state.loaded_project_name}")

st.title("Professional Software Effort Estimation Workspace")

tab1, tab2, tab3, tab4 = st.tabs(["Estimator", "FP Estimator", "AI Optimization", "Workspace"])

with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Project Information")
        st.text_input("Project Name", key="project_name")
        if not st.session_state.project_name.strip():
            st.warning("Please enter Project Name.")

        st.text_area("Project Description", height=150, key="description")
        st.selectbox("Development Mode", list(MODES.keys()), key="mode")
        st.selectbox("Size Input Unit", ["KLOC", "SLOC"], key="size_input_mode")

        if st.session_state.size_input_mode == "KLOC":
            st.number_input("Estimated Size (KLOC)", min_value=0.001, step=1.0, key="kloc")
        else:
            st.number_input("Estimated Size (SLOC)", min_value=1.0, step=100.0, key="sloc")

        st.number_input("Cost per Person-Month (VND)", min_value=1000000.0, step=1000000.0, key="cost_per_pm")

        if st.session_state.description.strip():
            st.info(f"Suggested mode from description: **{suggest_mode_rule_based(st.session_state.description)}**")

    with right:
        st.subheader("Cost Drivers")
        for group_name, drivers in COST_DRIVER_GROUPS.items():
            with st.expander(group_name, expanded=True):
                for driver in drivers:
                    meta = COST_DRIVERS[driver]
                    st.selectbox(
                        f"{driver} - {meta['label']}",
                        list(meta["values"].keys()),
                        key=driver,
                        help=meta["help"]
                    )

    selections = get_driver_selections()
    effective_kloc = get_effective_kloc()
    result = compute_result(
        st.session_state.project_name,
        st.session_state.description,
        st.session_state.mode,
        effective_kloc,
        st.session_state.cost_per_pm,
        selections
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
        st.session_state.link_fp_with_estimator
    )
    render_inline_warnings(validation_messages)

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

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig1 = px.bar(driver_df, x="Driver", y="Multiplier", color="Group", hover_data=["Name", "Rating"], title="Cost Driver Multipliers")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.subheader("Top 5 strongest driver impacts")
        st.dataframe(top5_df[["Driver", "Name", "Rating", "Multiplier", "Group"]], use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        compare_rows = []
        for each_mode in MODES:
            mode_result = compute_result(result["project_name"], result["description"], each_mode, effective_kloc, result["cost_per_pm"], selections)
            compare_rows.append({
                "Mode": each_mode,
                "Effort": mode_result["effort"],
                "Development Time": mode_result["tdev"],
                "Average Staff": mode_result["staff"]
            })
        fig2 = px.bar(pd.DataFrame(compare_rows), x="Mode", y="Effort", color="Mode", title="Effort Comparison by Mode")
        st.plotly_chart(fig2, use_container_width=True)

    with c4:
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

    pdf_bytes = make_pdf_report_bytes(
        {"project_name": st.session_state.project_name, "description": st.session_state.description},
        result,
        selections,
        fp_snapshot,
        validation_messages
    )

    a1, a2 = st.columns(2)
    with a1:
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
                    fp_snapshot
                )
                st.session_state.ai_result = ai_text or fallback_ai_summary(
                    st.session_state.description,
                    st.session_state.mode,
                    effective_kloc,
                    result["eaf"],
                    result["effort"],
                    result["tdev"],
                    result["staff"],
                    result["cost"],
                    selections,
                    fp_snapshot
                )
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
                    fp_snapshot
                )
            st.rerun()

    with a2:
        if pdf_bytes:
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"{(result['project_name'] or 'project').replace(' ', '_').lower()}_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("Add reportlab and kaleido to requirements.txt for PDF export.")

    st.subheader("AI Project Analysis")
    if st.session_state.ai_result:
        st.markdown(st.session_state.ai_result)
    else:
        st.caption("Click Generate AI Analysis to create the project analysis.")

with tab2:
    st.subheader("Function Point Estimator")

    fp_preset_col1, fp_preset_col2 = st.columns([4, 1])
    with fp_preset_col1:
        st.selectbox("FP Template", list(FP_PRESET_TEMPLATES.keys()), key="selected_fp_preset")
    with fp_preset_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Apply FP Template", use_container_width=True):
            apply_fp_preset(st.session_state.selected_fp_preset)
            st.rerun()

    st.selectbox("Programming Language", list(LANGUAGE_LOC_PER_FP.keys()) + ["Custom"], key="fp_language")
    if st.session_state.fp_language == "Custom":
        st.number_input("Custom LOC per FP", min_value=1.0, step=1.0, key="fp_custom_loc_per_fp")

    st.checkbox(
        "Compare FP-derived size with the Estimator size",
        key="link_fp_with_estimator",
        help="Leave this disabled if you want to use Estimator and FP Estimator independently."
    )
    st.caption("Estimator and FP Estimator can be used independently. Cross-check warnings only appear when this option is enabled.")

    for comp, label in FP_COMPONENT_LABELS.items():
        with st.expander(f"{comp} - {label}", expanded=False):
            items = get_fp_items(comp)
            edited = st.data_editor(
                pd.DataFrame(items, columns=["name", "det", "ftr_ret"]),
                num_rows="dynamic",
                use_container_width=True,
                key=f"editor_{comp}",
                column_config={
                    "name": st.column_config.TextColumn("Function Name"),
                    "det": st.column_config.NumberColumn("DET", min_value=1, step=1),
                    "ftr_ret": st.column_config.NumberColumn(FP_MATRIX[comp]["row_label"], min_value=1, step=1),
                }
            )
            st.session_state[f"fp_items_{comp}"] = edited.to_dict("records")

            preview_rows = []
            for row in get_fp_items(comp):
                det, fr = safe_int(row.get("det", 1), 1), safe_int(row.get("ftr_ret", 1), 1)
                cx = classify_fp_item(comp, det, fr)
                preview_rows.append({
                    "Function Name": row.get("name", ""),
                    "DET": det,
                    FP_MATRIX[comp]["row_label"]: fr,
                    "Complexity": cx,
                    "Weight": FP_WEIGHTS[comp][cx]
                })
            if preview_rows:
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

    st.markdown("### General System Characteristics (0-5)")
    st.caption("Select one score from 0 to 5 for each characteristic.")
    for idx, name in enumerate(GSC_NAMES):
        st.radio(
            name,
            options=[0, 1, 2, 3, 4, 5],
            horizontal=True,
            key=f"fp_gsc_{idx}"
        )

    fp_snapshot = get_fp_snapshot()
    fp_based_result = compute_result(
        f"{st.session_state.project_name or 'FP Project'} (FP-based)",
        st.session_state.description or "Function Point derived estimation",
        st.session_state.fp_mode,
        max(fp_snapshot["kloc"], 0.001),
        st.session_state.fp_cost_per_pm,
        get_driver_selections()
    )

    st.markdown("### FP Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("UFP", f"{fp_snapshot['ufp']:.2f}")
    m2.metric("DI", f"{fp_snapshot['di']}")
    m3.metric("VAF", f"{fp_snapshot['vaf']:.2f}")
    m4.metric("FP", f"{fp_snapshot['fp']:.2f}")
    m5.metric("KLOC", f"{fp_snapshot['kloc']:.2f}")

    st.caption(f"Estimated SLOC from FP: {fp_snapshot['sloc']:,.0f}")
    st.caption(f"LOC per FP factor used: {fp_snapshot['loc_per_fp']:.2f}")

    summary_rows = [{
        "Component": comp,
        "Count": s["count"],
        "Low": s["counts"]["Low"],
        "Average": s["counts"]["Average"],
        "High": s["counts"]["High"],
        "UFP": s["ufp"]
    } for comp, s in fp_snapshot["summary"].items()]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.selectbox("Mode for FP-based COCOMO", list(MODES.keys()), key="fp_mode")
    with mc2:
        st.number_input("FP-based Cost per Person-Month (VND)", min_value=1000000.0, step=1000000.0, key="fp_cost_per_pm")

    st.markdown("### FP-based COCOMO Projection")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Effort (PM)", f"{fp_based_result['effort']:.2f}")
    k2.metric("Time (Months)", f"{fp_based_result['tdev']:.2f}")
    k3.metric("Team Size", f"{fp_based_result['staff']:.2f}")
    k4.metric("Cost", format_currency_short(fp_based_result["cost"]))
    k5.metric("Risk Level", fp_based_result["risk_level"])

    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button("Generate FP AI Analysis", use_container_width=True):
            if enable_ai:
                ai_fp_text, _ = call_ai_fp_analysis(fp_snapshot, fp_based_result)
                st.session_state.ai_fp_result = ai_fp_text or fallback_fp_ai_summary(fp_snapshot, fp_based_result)
            else:
                st.session_state.ai_fp_result = fallback_fp_ai_summary(fp_snapshot, fp_based_result)
            st.rerun()

    with ac2:
        if st.button("Send FP Size to Estimator", use_container_width=True):
            st.session_state.fp_transfer_kloc = max(fp_snapshot["kloc"], 0.001)
            st.session_state.pending_fp_transfer = True
            st.rerun()

    st.subheader("FP AI Analysis")
    if st.session_state.ai_fp_result:
        st.markdown(st.session_state.ai_fp_result)
    else:
        st.caption("Click Generate FP AI Analysis to create the FP analysis.")

with tab3:
    st.subheader("AI Optimization")
    current_selections = get_driver_selections()
    current_result = compute_result(
        st.session_state.project_name,
        st.session_state.description,
        st.session_state.mode,
        get_effective_kloc(),
        st.session_state.cost_per_pm,
        current_selections
    )

    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        st.markdown("### Current Baseline")
        st.write(f"**Project:** {current_result['project_name'] or 'Untitled Project'}")
        st.write(f"**Mode:** {current_result['mode']}")
        st.write(f"**KLOC:** {current_result['kloc']:.2f}")
        st.write(f"**Effort:** {current_result['effort']:.2f} PM")
        st.write(f"**Cost:** {current_result['cost']:,.0f} VND")
        st.write(f"**Risk:** {current_result['risk_level']}")

    with c2:
        if st.button("Generate AI Optimization", use_container_width=True):
            suggestion, error_msg = call_ai_optimization(
                current_result["project_name"],
                current_result["description"],
                current_result,
                current_selections
            )
            st.session_state.ai_suggestion = suggestion
            if error_msg:
                st.warning("AI optimization is temporarily unavailable. The application is using the fallback suggestion.")
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
            st.dataframe(driver_change_df, use_container_width=True)

        st.markdown("### Current vs Suggested")
        st.dataframe(build_metric_compare_df(current_result, suggested_result), use_container_width=True)

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

        if st.button("Apply AI Suggestion", use_container_width=True):
            st.session_state.pending_ai_apply = True
            st.rerun()
    else:
        st.info("No AI optimization has been generated yet. Click Generate AI Optimization to create one.")

with tab4:
    st.subheader("Workspace")
    st.caption("Import a JSON package to continue working.")
    uploaded_json = st.file_uploader("Upload JSON Package", type=["json"])

    if uploaded_json is not None:
        if st.button("Load JSON Package", use_container_width=True):
            ok, msg = import_project_package(uploaded_json)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    if st.session_state.loaded_project_name:
        st.success(f"Current loaded package: {st.session_state.loaded_project_name}")
