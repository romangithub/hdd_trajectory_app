from __future__ import annotations
import math, tempfile, os
import numpy as np
import pandas as pd
import streamlit as st

from modules.db import init_db, save_project
from modules.reports import make_engineer_report
from modules.plots import save_profile_plot, save_plan_plot

st.set_page_config(page_title="Complex Profile", layout="wide")
st.title("🛠️ Complex Profile — Конструктор")

st.caption("Добавляй сегменты (Build/Hold/Drop), шаг станций, получай таблицу и графики. Сохраняй и делай PDF.")

seg_types = ["BUILD", "HOLD", "DROP"]
with st.form("segments_form", clear_on_submit=False):
    st.markdown("### Сегменты")
    default = [
        {"Тип": "BUILD", "Длина, м": 100.0, "Радиус, м": 800.0, "Угол полки, °": 0.0},
        {"Тип": "HOLD",  "Длина, м": 200.0, "Радиус, м": 0.0,   "Угол полки, °": 0.0},
        {"Тип": "DROP",  "Длина, м": 100.0, "Радиус, м": 800.0, "Угол полки, °": 0.0},
    ]
    df_edit = st.data_editor(pd.DataFrame(default), use_container_width=True, num_rows="dynamic")
    step = st.number_input("Шаг станций (м)", value=10.0, min_value=0.1, step=0.1)
    submitted = st.form_submit_button("Рассчитать", type="primary")

def calc_segment(md_start, inc_start, seg):
    L = float(seg["Длина, м"])
    typ = str(seg["Тип"]).upper()
    R  = float(seg.get("Радиус, м", 0.0) or 0.0)
    hold_deg = float(seg.get("Угол полки, °", 0.0) or 0.0)
    n = max(1, int(round(L/step)))
    md_arr = md_start + np.linspace(0, L, n+1)

    if typ == "HOLD":
        inc = np.full(n+1, inc_start if hold_deg == 0 else math.radians(hold_deg))
        tvd = np.cumsum(np.diff(np.hstack([[0.0], np.cos(inc[0]) * np.diff(md_arr)])))
        tvd = np.insert(tvd, 0, 0.0)
        return md_arr, inc, tvd

    if R <= 0: R = 1e9  # почти прямая
    d_inc = L / R
    if typ == "DROP": d_inc = -d_inc
    inc = inc_start + np.linspace(0, d_inc, n+1)
    tvd = np.cumsum(np.diff(np.hstack([[0.0], np.cos( (inc[:-1]+inc[1:])/2 ) * np.diff(md_arr)])))
    tvd = np.insert(tvd, 0, 0.0)
    return md_arr, inc, tvd

if submitted:
    md_total, inc_total, tvd_total = [], [], []
    md_cur, inc_cur = 0.0, 0.0
    first = True
    for _, seg in df_edit.iterrows():
        md_arr, inc_arr, tvd_arr = calc_segment(md_cur, inc_cur, seg)
        if first:
            md_total = md_arr; inc_total = inc_arr; tvd_total = tvd_arr
            first = False
        else:
            md_total = np.concatenate([md_total, md_arr[1:]])
            inc_total = np.concatenate([inc_total, inc_arr[1:]])
            tvd_total = np.concatenate([tvd_total, tvd_arr[1:]])
        md_cur = md_total[-1]; inc_cur = inc_total[-1]

    # план (как в simple): по East с учётом инклина
    east = np.cumsum(np.diff(np.hstack([[0.0], np.sin( (inc_total[:-1]+inc_total[1:])/2 ) * np.diff(md_total)])))
    east = np.insert(east, 0, 0.0)
    north = np.zeros_like(east)

    df = pd.DataFrame({"MD": md_total, "TVD": tvd_total, "Inc (deg)": np.rad2deg(inc_total), "N": north, "E": east})
    st.subheader("📑 Таблица станций")
    st.dataframe(df.head(300), use_container_width=True)

    p1 = save_profile_plot(md_total, tvd_total, title="Complex Profile (TVD vs MD)")
    p2 = save_plan_plot(north, east, title="Plan (N-E)")
    st.image(p1, caption="Профиль", use_column_width=True)
    st.image(p2, caption="План", use_column_width=True)

    st.download_button("⬇️ Экспорт станций (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="complex_profile_stations.csv", mime="text/csv")

    # PDF
    st.subheader("📄 PDF отчёт")
    title = st.text_input("Название отчёта", value="Complex Profile — Constructor")
    notes = st.text_area("Примечания", value="Сгенерировано модулем Complex Profile.")
    if st.button("Сформировать PDF"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        make_engineer_report(
            filename=tmp.name,
            project_name="ComplexProfile",
            section_title=title,
            params={"step": step},
            segments=df_edit.to_dict(orient="records"),
            stations=df,
            mud=None,
            plot_profile_path=p1,
            plot_plan_path=p2,
            extra_notes=notes
        )
        with open(tmp.name, "rb") as f:
            st.download_button("⬇️ Скачать PDF", f, file_name="Complex_Profile.pdf", mime="application/pdf")
        try: os.unlink(tmp.name)
        except: pass

    # Save DB
    init_db()
    pname = st.text_input("Project name", value="ComplexProfile1")
    if st.button("💾 Save Project"):
        save_project(pname, "complex_profile", {"step": step}, segments=df_edit.to_dict(orient="records"))
        st.success("Сохранено.")
