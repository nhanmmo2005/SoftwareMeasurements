import os
import json
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

st.set_page_config(
    page_title="Software Effort Estimation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()


def format_currency_short(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B VND"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M VND"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K VND"
    return f"{value:,.0f} VND"


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
        "pending_fp_transfer": False,
        "fp_transfer_kloc": 1.0,
        "pending_ai_apply": False,
        "fp_language": "Python",
        "fp_custom_loc_per_fp": float(LANGUAGE_LOC_PER_FP["Python"]),
        "fp_mode": "Organic",
        "fp_cost_per_pm": 12000000.0,
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
    st.session_state.selected_preset = "Start from Scratch"

    for driver, meta in COST_DRIVERS.items():
        values = list(meta["values"].keys())
        st.session_state[driver] = "Nominal" if "Nominal" in values else values[0]


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
    text = description.lower()
    if any(word in text for word in ["bank", "banking", "medical", "critical", "real-time", "iot", "embedded"]):
        return "Embedded"
    if any(word in text for word in ["hotel", "inventory", "management", "school", "e-commerce", "booking"]):
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
    return driver_df.sort_values(by="Impact", ascending=False).head(3)


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
        risks.append("Độ phức tạp dự án cao, làm tăng đáng kể effort.")
        recommendations.append("Chia hệ thống thành module nhỏ hơn hoặc làm theo từng giai đoạn.")

    if selections["RELY"] in ["High", "Very High"]:
        risks.append("Yêu cầu độ tin cậy cao làm tăng effort cho kiểm thử và validation.")
        recommendations.append("Bổ sung thêm thời gian cho test, review, QA và bảo mật.")

    if selections["ACAP"] in ["Very Low", "Low"] or selections["PCAP"] in ["Very Low", "Low"]:
        risks.append("Năng lực phân tích/lập trình thấp có thể làm giảm năng suất team.")
        recommendations.append("Tăng technical guidance, code review, hoặc phân công người mạnh hơn vào phần lõi.")

    if selections["TOOL"] in ["Very Low", "Low"]:
        risks.append("Tool support còn yếu, làm giảm hiệu suất phát triển.")
        recommendations.append("Nâng cấp tool support như IDE, Git workflow, linting, testing tools, CI/CD.")

    if selections["SCED"] in ["High", "Very High"]:
        risks.append("Deadline gắt có thể làm tăng rủi ro trễ tiến độ và giảm chất lượng.")
        recommendations.append("Điều chỉnh timeline hoặc kiểm soát scope chặt hơn.")

    if not risks:
        risks.append("Không có rủi ro nổi bật từ bộ cost drivers hiện tại.")

    if not recommendations:
        recommendations.append("Cấu hình hiện tại tương đối cân bằng.")

    return f"""
### Intelligent Project Analysis

**Suggested mode:** {suggest_mode_rule_based(description)}

**Estimated summary**
- Mode: **{mode}**
- KLOC: **{kloc:.2f}**
- EAF: **{eaf:.3f}**
- Effort: **{effort:.2f} person-months**
- Development Time: **{tdev:.2f} months**
- Average Team Size: **{staff:.2f}**
- Estimated Cost: **{cost:,.0f} VND**

**Key risks**
{chr(10).join([f"- {r}" for r in risks])}

**Recommendations**
{chr(10).join([f"- {r}" for r in recommendations])}
"""


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

Hãy trả lời bằng tiếng Việt, rõ ràng, có 4 phần:
1. Đánh giá mode hiện tại có phù hợp không
2. Giải thích vì sao effort/cost ở mức này
3. Các rủi ro chính của dự án
4. Khuyến nghị để giảm effort hoặc cost

Viết ngắn gọn, thực tế, dễ hiểu cho sinh viên và project manager.
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
        "goals": ["Reduce effort", "Reduce risk"],
        "reasoning": [
            "Improving tool support and modern practices can reduce development effort.",
            "Reducing schedule pressure can lower overall project risk.",
            "Improving personnel capability can increase productivity.",
        ],
        "expected_effect": "The suggested configuration is expected to reduce effort, cost, and risk compared with the current baseline.",
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
  "goals": ["Reduce effort", "Reduce risk"],
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
        {
            "Metric": "EAF",
            "Current": round(current_result["eaf"], 3),
            "Suggested": round(suggested_result["eaf"], 3),
            "Delta": round(suggested_result["eaf"] - current_result["eaf"], 3),
        },
        {
            "Metric": "Effort (PM)",
            "Current": round(current_result["effort"], 2),
            "Suggested": round(suggested_result["effort"], 2),
            "Delta": round(suggested_result["effort"] - current_result["effort"], 2),
        },
        {
            "Metric": "Time (Months)",
            "Current": round(current_result["tdev"], 2),
            "Suggested": round(suggested_result["tdev"], 2),
            "Delta": round(suggested_result["tdev"] - current_result["tdev"], 2),
        },
        {
            "Metric": "Team Size",
            "Current": round(current_result["staff"], 2),
            "Suggested": round(suggested_result["staff"], 2),
            "Delta": round(suggested_result["staff"] - current_result["staff"], 2),
        },
        {
            "Metric": "Estimated Cost (VND)",
            "Current": round(current_result["cost"], 0),
            "Suggested": round(suggested_result["cost"], 0),
            "Delta": round(suggested_result["cost"] - current_result["cost"], 0),
        },
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


def export_text_report(result: dict, selections: dict, ai_text: str) -> str:
    lines = [
        "SOFTWARE EFFORT ESTIMATION REPORT",
        "=" * 40,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
        "COST DRIVERS",
        "-" * 20,
    ]
    for driver, rating in selections.items():
        multiplier = COST_DRIVERS[driver]["values"][rating]
        lines.append(f"{driver} - {rating} (multiplier={multiplier})")
    lines.extend([
        "",
        "ANALYSIS",
        "-" * 20,
        ai_text if ai_text else "No analysis available.",
    ])
    return "\n".join(lines)


init_state()
process_pending_actions()

with st.sidebar:
    st.title("Project Estimator")
    st.markdown("Estimate project effort, schedule, team size, cost, and risk using Intermediate COCOMO.")

    preset = st.selectbox(
        "Project Template",
        list(PRESET_PROJECTS.keys()),
        index=list(PRESET_PROJECTS.keys()).index(st.session_state.selected_preset)
        if st.session_state.selected_preset in PRESET_PROJECTS else 0,
        key="preset_selector",
    )

    if st.button("Apply Template", use_container_width=True):
        apply_preset(preset)
        st.rerun()

    if st.button("Reset Form", use_container_width=True):
        reset_form()
        st.rerun()

    enable_ai = st.checkbox("Enable Smart Analysis", value=True)

    st.markdown("---")
    st.subheader("Quick Notes")
    st.caption("Core estimation is calculated locally using Intermediate COCOMO.")
    st.caption("The AI layer explains results and suggests optimization scenarios.")

st.title("Software Effort Estimation Dashboard")
st.caption("Estimate project effort, schedule, team size, cost, and risk using Intermediate COCOMO, Function Point, and AI Optimization.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Estimator",
    "FP Estimator",
    "AI Optimization",
    "Compare & What-if",
    "History",
    "Help",
])

with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Project Information")

        st.text_input("Project Name", key="project_name")
        st.text_area("Project Description", height=140, key="description")

        st.selectbox("Development Mode", list(MODES.keys()), key="mode")

        st.selectbox("Size Input Unit", ["KLOC", "SLOC"], key="size_input_mode")

        if st.session_state.size_input_mode == "KLOC":
            st.number_input("Estimated Size (KLOC)", min_value=0.001, step=1.0, key="kloc")
        else:
            st.number_input("Estimated Size (SLOC)", min_value=1.0, step=100.0, key="sloc")

        st.number_input(
            "Cost per Person-Month (VND)",
            min_value=1000000.0,
            step=1000000.0,
            key="cost_per_pm",
        )

        if st.session_state.description.strip():
            st.info(f"Suggested mode: **{suggest_mode_rule_based(st.session_state.description)}**")

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
    top3_df = get_top_impact_drivers(driver_df)

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
        st.subheader("Top 3 Most Influential Drivers")
        st.dataframe(
            top3_df[["Driver", "Name", "Rating", "Multiplier", "Group"]],
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

    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("Generate Analysis", use_container_width=True):
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
                    st.warning("Intelligent analysis is temporarily unavailable. The system is showing a built-in estimation summary instead.")
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

    with action_col2:
        if st.button("Save Estimation", use_container_width=True):
            save_payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "project_name": result["project_name"],
                "mode": result["mode"],
                "size_input_mode": st.session_state.size_input_mode,
                "kloc": result["kloc"],
                "eaf": result["eaf"],
                "effort_pm": result["effort"],
                "development_time_months": result["tdev"],
                "team_size": result["staff"],
                "estimated_cost_vnd": result["cost"],
                "risk_level": result["risk_level"],
            }
            save_history_row(save_payload)
            st.success("Estimation saved successfully.")

    with action_col3:
        report_text = export_text_report(result, selections, st.session_state.ai_result)
        st.download_button(
            "Download Report (.txt)",
            data=report_text,
            file_name=f"{(result['project_name'] or 'project').replace(' ', '_').lower()}_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.subheader("Intelligent Project Analysis")
    if st.session_state.ai_result:
        st.markdown(st.session_state.ai_result)
    else:
        st.caption("No analysis yet. Click Generate Analysis to create one.")

with tab2:
    st.subheader("Function Point Estimator")
    st.caption("Nhập 5 nhóm Function Point, tính UFP / VAF / FP, chuyển sang SLOC/KLOC, và nếu cần thì đẩy sang COCOMO.")

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

    fp_counts = get_fp_counts()
    ufp = calculate_ufp(fp_counts)
    di, vaf = calculate_vaf()
    fp_value = ufp * vaf
    fp_sloc = fp_value * loc_per_fp
    fp_kloc = fp_sloc / 1000.0

    st.markdown("### FP Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("UFP", f"{ufp:.2f}")
    m2.metric("DI", f"{di}")
    m3.metric("VAF", f"{vaf:.2f}")
    m4.metric("FP", f"{fp_value:.2f}")
    m5.metric("KLOC", f"{fp_kloc:.2f}")

    st.caption(f"Estimated SLOC from FP: {fp_sloc:,.0f}")
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
        kloc=max(fp_kloc, 0.001),
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

    st.caption(f"Full FP-based estimated cost: {fp_based_result['cost']:,.0f} VND")

    if st.button("Send FP Size to COCOMO", use_container_width=True):
        st.session_state.fp_transfer_kloc = max(fp_kloc, 0.001)
        st.session_state.pending_fp_transfer = True
        st.rerun()

with tab3:
    st.subheader("AI Optimization")
    st.caption("AI đọc current project, đề xuất thay đổi có cấu trúc, cho phép so sánh current vs suggested và bấm apply.")

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
                st.warning("AI optimization is temporarily unavailable. A built-in optimization suggestion is shown instead.")
            else:
                st.success("AI optimization generated successfully.")

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
            st.info("No driver changes suggested. Only the mode may have changed.")

        st.markdown("### Current vs Suggested Result")
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
        fig_ai = px.bar(
            compare_chart_df,
            x="Metric",
            y="Value",
            color="Scenario",
            barmode="group",
            title="Current vs Suggested",
        )
        st.plotly_chart(fig_ai, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Cost", format_currency_short(current_result["cost"]))
        c2.metric("Suggested Cost", format_currency_short(suggested_result["cost"]))
        c3.metric("Cost Delta", f"{suggested_result['cost'] - current_result['cost']:+,.0f}")

        if st.button("Apply AI Suggestion", use_container_width=True):
            st.session_state.pending_ai_apply = True
            st.rerun()
    else:
        st.info("No AI optimization yet. Click Generate AI Optimization to create a structured improvement scenario.")

with tab4:
    st.subheader("Compare Modes")
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

    st.caption(f"Full what-if cost: {new_result['cost']:,.0f} VND")

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

with tab5:
    st.subheader("Estimation History")
    history_df = load_history_df()
    if history_df.empty:
        st.info("No saved history yet.")
    else:
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "Download History CSV",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="estimation_history.csv",
            mime="text/csv",
        )

with tab6:
    st.subheader("How to Use")
    st.markdown("""
1. Chọn một **Project Template** rồi bấm **Apply Template**, hoặc dùng **Start from Scratch**  
2. Nhập Project Name, Description, KLOC/SLOC, Development Mode và Cost per Person-Month  
3. Chọn 15 cost drivers theo **4 nhóm chính**  
4. Xem Effort, Time, Team Size, Cost và Risk Level  
5. Bấm **Generate Analysis** để nhận phân tích thông minh  
6. Vào **FP Estimator** để tính UFP / VAF / FP và chuyển sang KLOC  
7. Vào **AI Optimization** để lấy suggested changes, so sánh với current result, và bấm apply  
8. Vào **Compare & What-if** để so sánh mode và thử thay đổi cost driver  
9. Bấm **Save Estimation** để lưu lịch sử hoặc tải report  
""")

    st.subheader("Driver Groups")
    st.markdown("""
- **Product Attributes**: RELY, DATA, CPLX  
- **Computer / Platform Attributes**: TIME, STOR, VIRT, TURN  
- **Personnel Attributes**: ACAP, AEXP, PCAP, VEXP, LEXP  
- **Project Attributes**: MODP, TOOL, SCED  
""")

    st.subheader("Recommended Demo Flow")
    st.markdown("""
- Demo **Start from Scratch** để cho thấy form trống hoàn toàn  
- Demo **Personal Expense Tracker** để cho thấy effort thấp  
- Demo **Medical Record System** để cho thấy reliability cao  
- Demo **Banking Transaction System** hoặc **Real-Time IoT Monitoring System** để cho thấy risk và cost cao  
- Sang tab **FP Estimator** để cho thấy tính UFP / VAF / FP và chuyển sang KLOC  
- Sang tab **AI Optimization** để cho thấy AI đề xuất thay đổi có cấu trúc và có thể apply trực tiếp  
""")
