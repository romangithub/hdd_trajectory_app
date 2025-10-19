from __future__ import annotations
import math, tempfile, os
import numpy as np
import pandas as pd
import streamlit as st

from modules.db import save_project, init_db
from modules.reports import make_engineer_report
from modules.plots import save_profile_plot, save_plan_plot

st.set_page_config(page_title="Simple Profile", layout="wide")
st.title("📘 Simple Profile — R–Lh–R")

st.caption("Участки: забуривания (R), снижение (Lh), выбуривания (R). Станции и графики, экспорт, БД.")

# ---- Inputs
c1, c2, c3 = st.columns(3)
with c1:
    L_build = st.number_input("Длина забуривания (м)", value=100.0, min_value=0.0, step=1.0)
    R_build = st.number_input("Радиус забуривания (м)", value=800.0, min_value=1.0, step=10.0)
with c2:
    L_hold = st.number_input("Длина горизонтального участка (м)", value=200.0, min_value=0.0, step=1.0)
    inc_hold_deg = st.number_input("Угол полки (°)", value=0.0, step=0.1)
with c3:
    L_drop = st.number_input("Длина выбуривания (м)", value=100.0, min_value=0.0, step=1.0)
    R_drop = st.number_input("Радиус выбуривания (м)", value=800.0, min_value=1.0, step=10.0)

step = st.number_input("Шаг станций (м)", value=10.0, min_value=0.1, step=0.1)

# ---- Calculation
def build_arc(md, L, R, inc0=0.0):
    """Круговая дуга с постоянным инклинометрическим приростом."""
    # приращение инклинометрии по дуге: dInc = L / R (в радианах)
    n = max(1, int(round(L/step)))
    md_arr = md + np.linspace(0, L, n+1)
    d_inc = (L / R) if R > 0 else 0.0
    inc = inc0 + np.linspace(0, d_inc, n+1)
    tvd = np.cumsum(np.diff(np.hstack([[0.0], np.cos(inc[:-1] + np.diff(inc)/2) * np.diff(md_arr)])))
    tvd = np.insert(tvd, 0, 0.0)
    return md_arr, inc, tvd

def hold(md, L, inc_rad):
    n = max(1, int(round(L/step)))
    md_arr = md + np.linspace(0, L, n+1)
    tvd = np.cumsum(np.diff(np.hstack([[0.0], np.cos(inc_rad)*np.diff(md_arr)])))
    tvd = np.insert(tvd, 0, 0.0)
    inc = np.full_like(md_arr, inc_rad)
    return md_arr, inc, tvd

# Соберём профиль
md0 = 0.0
md1, inc1, tvd1 = build_arc(md0, L_build, R_build, inc0=0.0)
md2, inc2, tvd2 = hold(md1[-1], L_hold, np.deg2rad(inc_hold_deg if inc_hold_deg else inc1[-1]*180/math.pi*0))
inc_mid = inc1[-1] if L_build > 0 else np.deg2rad(inc_hold_deg)
md3, inc3, tvd3 = build_arc(md2[-1], L_drop, R_drop, inc0=inc_mid)

# Склеим
md = np.concatenate([md1, md2[1:], md3[1:]])
inc = np.concatenate([inc1, inc2[1:], inc3[1:]])
tvd = np.cumsum(np.diff(np.hstack([[0.0], np.cos( (inc[:-1]+inc[1:])/2 ) * np.diff(md)])))
tvd = np.insert(tvd, 0, 0.0)

# Примитивный план (без азимута: идём по East)
east = np.cumsum(np.diff(np.hstack([[0.0], np.sin( (inc[:-1]+inc[1:])/2 ) * np.diff(md)])))
east = np.insert(east, 0, 0.0)
north = np.zeros_like(east)

# Таблица станций
df = pd.DataFrame({
    "MD": md, "TVD": tvd, "Inc (deg)": np.rad2deg(inc), "N": north, "E": east
})

st.subheader("📑 Таблица станций")
st.dataframe(df.head(200), use_container_width=True)

# Метрики
m1, m2, m3 = st.columns(3)
m1.metric("MD (конец)", f"{md[-1]:.1f} м")
m2.metric("TVD (конец)", f"{tvd[-1]:.1f} м")
m3.metric("Max Inc", f"{np.rad2deg(np.max(inc)):.1f}°")

# Графики
from modules.plots import save_profile_plot, save_plan_plot
plot_profile = save_profile_plot(md, tvd, title="Profile (TVD vs MD)")
plot_plan = save_plan_plot(north, east, title="Plan (N-E)")

st.image(plot_profile, caption="Профиль", use_column_width=True)
st.image(plot_plan, caption="План", use_column_width=True)

# Экспорт
st.download_button("⬇️ Экспорт станций (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="simple_profile_stations.csv", mime="text/csv")

# PDF
st.subheader("📄 PDF отчёт")
title = st.text_input("Название отчёта", value="Simple Profile — R–Lh–R")
notes = st.text_area("Примечания", value="Автоматически сгенерированный отчёт по простому профилю.")
if st.button("Сформировать PDF"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    make_engineer_report(
        filename=tmp.name,
        project_name="SimpleProfile",
        section_title=title,
        params={"L_build": L_build, "R_build": R_build, "L_hold": L_hold, "inc_hold_deg": inc_hold_deg, "L_drop": L_drop, "R_drop": R_drop, "step": step},
        segments=None,
        stations=df,
        mud=None,
        plot_profile_path=plot_profile,
        plot_plan_path=plot_plan,
        extra_notes=notes
    )
    with open(tmp.name, "rb") as f:
        st.download_button("⬇️ Скачать PDF", f, file_name="Simple_Profile.pdf", mime="application/pdf")
    try: os.unlink(tmp.name)
    except: pass

# Save to DB
init_db()
project_name = st.text_input("Project name", value="SimpleProfile1")
if st.button("💾 Save Project"):
    save_project(project_name, "simple_profile", {"params": {"L_build": L_build, "R_build": R_build, "L_hold": L_hold, "inc_hold_deg": inc_hold_deg, "L_drop": L_drop, "R_drop": R_drop, "step": step}}, segments=df.to_dict(orient="records"))
    st.success("Сохранено.")
