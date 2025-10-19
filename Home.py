from __future__ import annotations
import streamlit as st
from modules.db import list_projects

st.set_page_config(page_title="HDD Toolkit", layout="wide")
st.title("🛠️ HDD Toolkit — Главная")

st.markdown(
    """
Добро пожаловать! Выберите режим:
- 📘 **Simple Profile** — простой профиль (R–Lh–R).
- 🛠️ **Complex Profile** — конструктор сегментов.
- 💧 **Mud Calculation** — СП-341 (подбор) / API-13D (гидравлика) / Импорт XLSX.
"""
)

st.page_link("pages/1_Simple_Profile.py", label="📘 Simple Profile", icon="📘")
st.page_link("pages/2_Complex_Profile.py", label="🛠️ Complex Profile", icon="🛠️")
st.page_link("pages/4_Mud_Calculation.py", label="💧 Mud Calculation", icon="💧")

st.divider()
st.subheader("🗃️ Недавние проекты (SQLite)")
rows = list_projects(20)
if not rows:
    st.info("Пока пусто. Сохраняйте проекты со страниц.")
else:
    import pandas as pd
    df = pd.DataFrame(rows, columns=["id","name","type","created_at"])
    st.dataframe(df, use_container_width=True)
