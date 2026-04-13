import os
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


def reset_form():
    st.session_state.project_name = ""
    st.session_state.description = ""
    st.session_state.mode = "Organic"
    st.session_state.kloc = 1.0
    st.session_state.cost_per_pm = 12000000.0
    st.session_state.ai_result = ""
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
    st.session_state.kloc = preset["kloc"]
    st.session_state.cost_per_pm = preset["cost_per_pm"]
    st.session_state.ai_result = ""
    st.session_state.selected_preset = name

    for driver, rating in preset["drivers"].items():
        st.session_state[driver] = rating


def init_state():
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = "Start from Scratch"
    if "preset_selector" not in st.session_state:
        st.session_state.preset_selector = st.session_state.selected_preset
    if "project_name" not in st.session_state:
        reset_form()
    if "ai_result" not in st.session_state:
        st.session_state.ai_result = ""

    for driver, meta in COST_DRIVERS.items():
        if driver not in st.session_state:
            values = list(meta["values"].keys())
            st.session_state[driver] = "Nominal" if "Nominal" in values else values[0]


def get_driver_group(driver: str) -> str:
    for group, drivers in COST_DRIVER_GROUPS.items():
        if driver in drivers:
            return group
    return "Other"


def get_driver_selections():
    return {driver: st.session_state[driver] for driver in COST_DRIVERS}


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

with st.sidebar:
    st.title("Project Estimator")
    st.markdown("Estimate project effort, schedule, team size, cost, and risk using COCOMO-based analysis.")

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
    st.caption("Core estimation is always calculated locally using Intermediate COCOMO.")
    st.caption("Smart analysis provides additional explanation, risk analysis, and recommendations.")

st.title("Software Effort Estimation Dashboard")
st.caption("Estimate project effort, schedule, team size, cost, and risk using COCOMO-based analysis.")

tab1, tab2, tab3, tab4 = st.tabs(["Estimator", "Compare & What-if", "History", "Help"])

with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Project Information")

        st.text_input(
            "Project Name",
            key="project_name",
        )

        st.text_area(
            "Project Description",
            height=140,
            key="description",
        )

        st.selectbox(
            "Development Mode",
            list(MODES.keys()),
            key="mode",
        )

        st.number_input(
            "Estimated Size (KLOC)",
            min_value=1.0,
            step=1.0,
            key="kloc",
        )

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
    eaf = calculate_eaf(selections)
    effort, tdev, staff = cocomo_estimate(st.session_state.mode, st.session_state.kloc, eaf)
    cost = estimate_cost(effort, st.session_state.cost_per_pm)
    risk_level = get_risk_level(eaf, selections)

    result = {
        "project_name": st.session_state.project_name,
        "description": st.session_state.description,
        "mode": st.session_state.mode,
        "kloc": st.session_state.kloc,
        "cost_per_pm": st.session_state.cost_per_pm,
        "eaf": eaf,
        "effort": effort,
        "tdev": tdev,
        "staff": staff,
        "cost": cost,
        "risk_level": risk_level,
    }

    st.subheader("Estimation Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Effort (PM)", f"{effort:.2f}")
    m2.metric("Time (Months)", f"{tdev:.2f}")
    m3.metric("Team Size", f"{staff:.2f}")
    m4.metric("Cost", format_currency_short(cost))
    m5.metric("Risk Level", risk_level)

    st.caption(f"Full estimated cost: {cost:,.0f} VND")

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
            e, t, s = cocomo_estimate(each_mode, st.session_state.kloc, eaf)
            compare_rows.append({
                "Mode": each_mode,
                "Effort": e,
                "Development Time": t,
                "Average Staff": s,
            })
        compare_df = pd.DataFrame(compare_rows)
        fig2 = px.bar(compare_df, x="Mode", y="Effort", color="Mode", title="Effort Comparison by Mode")
        st.plotly_chart(fig2, use_container_width=True)

    with chart_col2:
        cost_breakdown_df = pd.DataFrame([
            {"Category": "Development", "Cost": cost * 0.55},
            {"Category": "Testing & QA", "Cost": cost * 0.20},
            {"Category": "Management", "Cost": cost * 0.15},
            {"Category": "Tools & Infrastructure", "Cost": cost * 0.10},
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
                    st.session_state.kloc,
                    eaf,
                    effort,
                    tdev,
                    staff,
                    cost,
                    selections,
                )
                if ai_text:
                    st.session_state.ai_result = ai_text
                else:
                    st.session_state.ai_result = fallback_ai_summary(
                        st.session_state.description,
                        st.session_state.mode,
                        st.session_state.kloc,
                        eaf,
                        effort,
                        tdev,
                        staff,
                        cost,
                        selections,
                    )
                    st.warning("Intelligent analysis is temporarily unavailable. The system is showing a built-in estimation summary instead.")
            else:
                st.session_state.ai_result = fallback_ai_summary(
                    st.session_state.description,
                    st.session_state.mode,
                    st.session_state.kloc,
                    eaf,
                    effort,
                    tdev,
                    staff,
                    cost,
                    selections,
                )

    with action_col2:
        if st.button("Save Estimation", use_container_width=True):
            save_payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "project_name": result["project_name"],
                "mode": result["mode"],
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
    st.subheader("Compare Modes")
    selections = get_driver_selections()
    eaf = calculate_eaf(selections)

    mode_rows = []
    for each_mode in MODES:
        e, t, s = cocomo_estimate(each_mode, st.session_state.kloc, eaf)
        mode_rows.append({
            "Mode": each_mode,
            "Effort (PM)": round(e, 2),
            "Time (Months)": round(t, 2),
            "Team Size": round(s, 2),
            "Cost (VND)": round(estimate_cost(e, st.session_state.cost_per_pm), 0),
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

    base_eaf = calculate_eaf(selections)
    base_effort, base_tdev, base_staff = cocomo_estimate(st.session_state.mode, st.session_state.kloc, base_eaf)
    base_cost = estimate_cost(base_effort, st.session_state.cost_per_pm)

    new_eaf = calculate_eaf(scenario_selections)
    new_effort, new_tdev, new_staff = cocomo_estimate(st.session_state.mode, st.session_state.kloc, new_eaf)
    new_cost = estimate_cost(new_effort, st.session_state.cost_per_pm)

    what_if_df = pd.DataFrame([
        {"Scenario": "Current", "EAF": base_eaf, "Effort": base_effort, "Time": base_tdev, "Cost": base_cost},
        {"Scenario": "What-if", "EAF": new_eaf, "Effort": new_effort, "Time": new_tdev, "Cost": new_cost},
    ])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EAF Change", f"{new_eaf:.3f}", f"{new_eaf - base_eaf:+.3f}")
    c2.metric("Effort Change", f"{new_effort:.2f}", f"{new_effort - base_effort:+.2f}")
    c3.metric("Time Change", f"{new_tdev:.2f}", f"{new_tdev - base_tdev:+.2f}")
    c4.metric("Cost Change", format_currency_short(new_cost), f"{new_cost - base_cost:+,.0f}")

    st.caption(f"Full what-if cost: {new_cost:,.0f} VND")

    fig_what_if = px.bar(
        what_if_df.melt(id_vars="Scenario", value_vars=["Effort", "Time", "Cost"]),
        x="variable",
        y="value",
        color="Scenario",
        barmode="group",
        title="What-if Comparison",
    )
    st.plotly_chart(fig_what_if, use_container_width=True)

with tab3:
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

with tab4:
    st.subheader("How to Use")
    st.markdown("""
1. Chọn một **Project Template** rồi bấm **Apply Template**, hoặc dùng **Start from Scratch**  
2. Nhập Project Name, Description, KLOC, Development Mode và Cost per Person-Month  
3. Chọn 15 cost drivers theo **4 nhóm chính**  
4. Xem Effort, Time, Team Size, Cost và Risk Level  
5. Bấm **Generate Analysis** để nhận phân tích thông minh  
6. Bấm **Save Estimation** để lưu lịch sử  
7. Vào tab **Compare & What-if** để so sánh mode và thử thay đổi cost driver  
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
- Sang tab **What-if** và đổi `TOOL` hoặc `CPLX` để cho thấy chi phí thay đổi
""")