from __future__ import annotations
import math, tempfile, os
import pandas as pd
import streamlit as st

from modules.reports import make_engineer_report
from modules.db import init_db, save_project

st.set_page_config(page_title="Mud Calculation", layout="wide")
st.title("💧 Mud Calculation — СП 341 / API RP 13D / Импорт XLSX")

st.caption("СП 341 — план раствора; API RP 13D — гидравлика/транспорт/оптимизация; Импорт XLSX — автозаполнение СП.")

# ---------------- Helpers (API) ----------------
def geo_from_inputs(D_pipe_mm: float, k_hole: float):
    D_pipe = D_pipe_mm / 1000.0
    D_hole = k_hole * D_pipe
    Dh = D_hole - D_pipe
    A_annulus = math.pi/4 * (D_hole**2 - D_pipe**2)
    A_hole = math.pi * (D_hole**2) / 4.0
    return D_pipe, D_hole, Dh, A_annulus, A_hole

def rheology_effective_viscosity(model: str, rho: float, V: float, Dh: float,
                                 mu_cP=None, tau_y=None, mu_p_cP=None, K=None, n=None):
    if model == "Ньютон":
        mu = (mu_cP or 1.0) * 1e-3
        Re = (rho * V * Dh) / max(mu,1e-12)
        f = 64.0 / max(Re,1e-9) if Re < 2300.0 else 0.3164 * (Re ** -0.25)
        return "Newton", f, mu
    elif model == "Bingham Plastic":
        mu_p = (mu_p_cP or 1.0) * 1e-3
        gamma = 8.0 * V / max(Dh, 1e-12)
        mu_eff = mu_p + ( (tau_y or 0.0) / max(gamma, 1e-12) )
        Re = (rho * V * Dh) / max(mu_eff,1e-12)
        f = 64.0 / max(Re,1e-9) if Re < 2300.0 else 0.3164 * (Re ** -0.25)
        return "Bingham", f, mu_eff
    else:
        K_ = (K or 0.5); n_ = (n or 0.6)
        Re_g = (rho * (V ** (2.0 - n_)) * (Dh ** n_)) / (K_ * ((3.0 * n_ + 1.0) / (4.0 * n_)))
        f = 16.0 / max(Re_g,1e-9) if Re_g < 2100.0 else 0.079 * (Re_g ** -0.25)
        gamma = 8.0 * V / max(Dh, 1e-12)
        mu_app = K_ * (max(gamma,1e-12) ** (n_ - 1.0))
        return f"PowerLaw Re_g={Re_g:.0f}", f, mu_app

def compute_all_api(Q_l_min: float, inputs: dict):
    g = 9.80665
    D_pipe_mm = inputs["D_pipe_mm"]; k_hole = inputs["k_hole"]; L_profile = inputs["L_profile"]
    rho = inputs["rho"]; model = inputs["model"]; depth_for_ecd = inputs["depth_for_ecd"]
    loc_losses_bar = inputs["loc_losses_bar"]
    alpha_h = inputs["alpha_h"]; d_cut_mm = inputs["d_cut_mm"]; rho_cut = inputs["rho_cut"]; ROP_m_h = inputs["ROP_m_h"]
    mu_cP = inputs.get("mu_cP"); tau_y = inputs.get("tau_y"); mu_p_cP = inputs.get("mu_p_cP")
    K = inputs.get("K"); n = inputs.get("n")

    D_pipe, D_hole, Dh, A_annulus, A_hole = geo_from_inputs(D_pipe_mm, k_hole)
    Q_m3_s = (Q_l_min / 1000.0) / 60.0
    V = Q_m3_s / max(A_annulus,1e-12)

    regime, f, mu_eff_for_cuttings = rheology_effective_viscosity(
        model, rho, V, Dh, mu_cP=mu_cP, tau_y=tau_y, mu_p_cP=mu_p_cP, K=K, n=n
    )
    dp_fric_Pa = f * (L_profile / max(Dh,1e-12)) * (rho * V * V / 2.0)
    dp_total_bar = dp_fric_Pa / 1e5 + loc_losses_bar

    dp_depth_Pa = (dp_total_bar * 1e5) * (depth_for_ecd / max(L_profile,1e-12))
    ecd = rho + dp_depth_Pa / g

    d_cut = d_cut_mm / 1000.0
    w_stokes = (max(rho_cut - rho,0.0) * g * d_cut * d_cut) / (18.0 * max(mu_eff_for_cuttings,1e-12))
    ang = abs(alpha_h)
    if ang <= 5.0: V_min_angle = 1.0
    elif ang <= 30.0: V_min_angle = 0.6
    else: V_min_angle = 0.3
    V_min_stokes = 1.5 * w_stokes
    V_min_required = max(V_min_angle, V_min_stokes)
    TR = V / max(V_min_required,1e-12)

    ROP_m_s = ROP_m_h / 3600.0
    A_hole = math.pi * ( (k_hole*(D_pipe_mm/1000.0))**2 ) / 4.0
    Q_cuttings_m3_s = ROP_m_s * A_hole
    A_annulus = float(A_annulus)
    Q_capacity_m3_s = V * A_annulus * 0.15

    V_total_m3 = A_annulus * L_profile

    return {
        "D_hole": D_hole, "Dh": Dh, "A_annulus": A_annulus,
        "V": V, "Q_l_min": Q_l_min, "regime": regime, "f": f,
        "dp_total_bar": dp_total_bar, "dp_fric_bar": dp_fric_Pa/1e5,
        "V_total_m3": V_total_m3, "ecd": ecd,
        "w_stokes": w_stokes, "V_min_angle": V_min_angle, "V_min_stokes": V_min_stokes,
        "V_min_required": V_min_required, "TR": TR,
        "Q_cuttings_m3_s": Q_cuttings_m3_s, "Q_capacity_m3_s": Q_capacity_m3_s
    }

def satisfies_constraints_api(res: dict, p_max_bar: float):
    return (res["TR"] >= 1.0) and (res["Q_capacity_m3_s"] >= res["Q_cuttings_m3_s"]) and (res["dp_total_bar"] <= p_max_bar)

def parse_sp341_excel(file) -> dict:
    import math as _m
    df_raw = pd.read_excel(file, sheet_name="Отходы раствора по СП 341", header=None)
    labels = {}
    for _, row in df_raw.iterrows():
        label = row[1] if len(row) > 1 else None
        v4 = row[4] if len(row) > 4 else None
        v6 = row[6] if len(row) > 6 else None
        if isinstance(label, str) and label.strip():
            val = None
            for v in (v4, v6):
                if v is not None and not (isinstance(v, float) and _m.isnan(v)):
                    val = v; break
            labels[label.strip()] = val
    return labels

# ---------------- Tabs ----------------
tab_sp, tab_api, tab_imp = st.tabs(["СП 341 — подбор", "API RP 13D — гидравлика", "Импорт XLSX (СП 341)"])

# ===== СП 341
with tab_sp:
    st.subheader("📐 Исходные данные (СП 341)")
    c0, c1, c2 = st.columns(3)
    with c0:
        L = st.number_input("Длина L, м", value=float(st.session_state.get("L_sp341", 120.0)), min_value=1.0, step=1.0)
    with c1:
        k_add = st.number_input("Надбавка (доля)", value=float(st.session_state.get("k_add_sp341", 0.30)), min_value=0.0, max_value=2.0, step=0.05)
    with c2:
        soil = st.selectbox("Тип грунта", ["Неактивная глина", "Активная глина", "Песок/галечник", "Суглинок"])

    presets = {
        "Неактивная глина": {"C_bent_kg_m3": 25.0, "C_poly_kg_m3": 0.0},
        "Активная глина":   {"C_bent_kg_m3": 30.0, "C_poly_kg_m3": 0.5},
        "Песок/галечник":   {"C_bent_kg_m3": 35.0, "C_poly_kg_m3": 1.0},
        "Суглинок":         {"C_bent_kg_m3": 30.0, "C_poly_kg_m3": 0.5},
    }
    base = presets[soil]
    c3, c4 = st.columns(2)
    with c3:
        conc_bent = st.number_input("Конц. бентонита, кг/м³", value=float(st.session_state.get("conc_bent", base["C_bent_kg_m3"])), min_value=0.0, step=1.0)
    with c4:
        conc_poly = st.number_input("Конц. полимера, кг/м³", value=float(st.session_state.get("conc_poly", base["C_poly_kg_m3"])), min_value=0.0, step=0.1)

    st.markdown("#### Этапы (пилот/расширения)")
    default = pd.DataFrame([
        {"Этап": "Пилот", "Диаметр, мм": 200.0},
        {"Этап": "Расширение 1", "Диаметр, мм": 600.0},
        {"Этап": "Расширение 2", "Диаметр, мм": 800.0},
    ])
    df_pass = st.data_editor(st.session_state.get("sp341_df_pass", default), use_container_width=True, num_rows="dynamic")

    if st.button("Рассчитать по СП 341", type="primary", key="sp_calc"):
        rows=[]
        V_total = M_b_total = M_p_total = 0.0
        for i, r in df_pass.iterrows():
            D_m = float(r["Диаметр, мм"])/1000.0
            V_hole = math.pi*(D_m**2)/4.0 * L
            V_stage = V_hole*(1.0+k_add)
            m_b = conc_bent*V_stage
            m_p = conc_poly*V_stage
            V_total += V_stage; M_b_total += m_b; M_p_total += m_p
            rows.append({"№": i+1,"Этап": r.get("Этап", f"Этап {i+1}"),"D (мм)": r["Диаметр, мм"],
                         "V_скважины, м³": round(V_hole,2),"V_расчёт, м³": round(V_stage,2),
                         "Бентонит, кг": round(m_b,1),"Полимер, кг": round(m_p,1)})
        df_out = pd.DataFrame(rows)
        st.dataframe(df_out, use_container_width=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Раствор, м³", f"{V_total:.1f}")
        m2.metric("Бентонит, кг", f"{M_b_total:.0f}")
        m3.metric("Полимер, кг", f"{M_p_total:.0f}")

        st.download_button("⬇️ CSV", df_out.to_csv(index=False).encode("utf-8"), file_name="mud_sp341.csv", mime="text/csv")

        st.subheader("📄 PDF отчёт (СП 341)")
        t = st.text_input("Название отчёта", value="Mud Planning — СП 341")
        notes = st.text_area("Примечания", value=f"Грунт: {soil}; k_add={k_add}.")
        if st.button("Сформировать PDF (СП 341)", key="sp_pdf"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            make_engineer_report(
                filename=tmp.name, project_name="Mud_SP341", section_title=t,
                params={"L": L, "k_add": k_add, "soil": soil, "C_bent": conc_bent, "C_poly": conc_poly},
                segments=rows, stations=None, mud={"V_total_m3": V_total, "bentonite_kg": M_b_total, "polymer_kg": M_p_total},
                plot_profile_path=None, plot_plan_path=None, extra_notes=notes
            )
            with open(tmp.name, "rb") as f:
                st.download_button("⬇️ Скачать PDF", f, file_name="Mud_SP341_Report.pdf", mime="application/pdf")
            try: os.unlink(tmp.name)
            except: pass

        init_db()
        pname = st.text_input("Project name (SP 341)", value="Mud_SP341_1")
        if st.button("💾 Save Project (SP 341)", key="sp_save"):
            save_project(pname, "mud_sp341",
                         {"L": L, "k_add": k_add, "soil": soil, "C_bent": conc_bent, "C_poly": conc_poly},
                         segments=rows)
            st.success("Сохранено.")

# ===== API 13D
with tab_api:
    st.subheader("Геометрия")
    a,b,c = st.columns(3)
    with a:
        D_pipe_mm = st.number_input("D трубы, мм", value=500.0, min_value=50.0, step=10.0)
        L_profile = st.number_input("Длина L, м", value=120.0, min_value=1.0, step=1.0)
    with b:
        k_hole = st.number_input("D_скв = k·D", value=1.30, min_value=1.0, max_value=3.0, step=0.05)
        depth_for_ecd = st.number_input("Глубина для ECD, м", value=10.0, min_value=0.0, step=1.0)
    with c:
        loc_losses_bar = st.number_input("Местные потери, бар", value=2.0, min_value=0.0, step=0.5)
        p_max_bar = st.number_input("Макс. ΔP, бар", value=30.0, min_value=1.0, step=1.0)

    st.subheader("Реология")
    model = st.selectbox("Модель", ["Ньютон","Bingham Plastic","Power Law"])
    if model=="Ньютон":
        rho = st.number_input("ρ, кг/м³", value=1050.0, min_value=900.0, max_value=1800.0, step=10.0); mu_cP = st.number_input("μ, cP", value=25.0, min_value=1.0, max_value=300.0, step=1.0)
        rheo={"mu_cP":mu_cP}
    elif model=="Bingham Plastic":
        rho = st.number_input("ρ, кг/м³", value=1050.0, min_value=900.0, max_value=1800.0, step=10.0); tau_y = st.number_input("τy, Па", value=5.0, min_value=0.0, step=0.5); mu_p_cP = st.number_input("μp, cP", value=20.0, min_value=1.0, max_value=300.0, step=1.0)
        rheo={"tau_y":tau_y,"mu_p_cP":mu_p_cP}
    else:
        rho = st.number_input("ρ, кг/м³", value=1050.0, min_value=900.0, max_value=1800.0, step=10.0); K = st.number_input("K, Па·сⁿ", value=0.5, min_value=0.01, step=0.01); n = st.number_input("n", value=0.6, min_value=0.1, max_value=1.5, step=0.05)
        rheo={"K":K,"n":n}

    st.subheader("Режим циркуляции")
    mode = st.selectbox("Как задать", ["По скорости","Задать Q"])
    if mode=="По скорости":
        target_v = st.number_input("Целевая V, м/с", value=1.20, min_value=0.1, max_value=5.0, step=0.1)
        Q_user=None
    else:
        Q_user = st.number_input("Q, л/мин", value=1500.0, min_value=10.0, step=10.0); target_v=None

    st.subheader("Транспорт (оценка)")
    t1,t2,t3,t4 = st.columns(4)
    with t1: alpha_h = st.number_input("α_h, °", value=0.0, min_value=-5.0, max_value=90.0, step=0.5)
    with t2: d_cut_mm = st.number_input("Размер частиц, мм", value=5.0, min_value=0.1, max_value=50.0, step=0.5)
    with t3: rho_cut = st.number_input("ρ частиц, кг/м³", value=2650.0, min_value=1000.0, max_value=4000.0, step=50.0)
    with t4: ROP_m_h = st.number_input("ROP, м/ч", value=10.0, min_value=0.1, step=0.5)

    calc, opt = st.columns(2)
    with calc: calc_click = st.button("Рассчитать (API)", type="primary")
    with opt:  opt_click  = st.button("Оптимизировать Q (мин.)")

    if calc_click or opt_click:
        try:
            D_pipe, D_hole, Dh, A_annulus, _ = geo_from_inputs(D_pipe_mm, k_hole)
            if D_hole <= D_pipe:
                st.error("D_скв должен быть больше D_трубы."); st.stop()

            inputs = {"D_pipe_mm":D_pipe_mm,"k_hole":k_hole,"L_profile":L_profile,
                      "rho":rho,"model":model,"depth_for_ecd":depth_for_ecd,"loc_losses_bar":loc_losses_bar,
                      "alpha_h":alpha_h,"d_cut_mm":d_cut_mm,"rho_cut":rho_cut,"ROP_m_h":ROP_m_h, **rheo}

            if calc_click:
                if mode=="По скорости":
                    Q_l_min = target_v*A_annulus*60.0*1000.0
                else:
                    Q_l_min = Q_user
                res = compute_all_api(Q_l_min, inputs)
                st.session_state.mud_params = inputs; st.session_state.mud_results = res
                st.success("Расчёт выполнен ✅")
                g1,g2,g3,g4=st.columns(4)
                g1.metric("Q, л/мин", f"{res['Q_l_min']:.0f}"); g2.metric("V, м/с", f"{res['V']:.2f}")
                g3.metric("ΔP, бар", f"{res['dp_total_bar']:.2f}"); g4.metric("ECD, кг/м³", f"{res['ecd']:.0f}")
                st.markdown("### Транспорт")
                tt1,tt2,tt3,tt4=st.columns(4)
                tt1.metric("wₛ, м/с", f"{res['w_stokes']:.3f}")
                tt2.metric("Vₘᵢₙ(угол), м/с", f"{res['V_min_angle']:.2f}")
                tt3.metric("Vₘᵢₙ(Стокс), м/с", f"{res['V_min_stokes']:.2f}")
                tt4.metric("Vₘᵢₙ (итог), м/с", f"{res['V_min_required']:.2f}")
                ttt1,ttt2=st.columns(2)
                ttt1.metric("TR", f"{res['TR']:.2f}")
                ttt2.metric("Q_cut vs Q_cap (м³/ч)", f"{res['Q_cuttings_m3_s']*3600:.1f} vs {res['Q_capacity_m3_s']*3600:.1f}")
                if satisfies_constraints_api(res, p_max_bar):
                    st.success("✅ Ограничения соблюдены")
                else:
                    st.warning("⚠️ Ограничения не соблюдены")

            if opt_click:
                lo, hi = 50.0, 10000.0; best=None
                for _ in range(40):
                    mid=0.5*(lo+hi)
                    res=compute_all_api(mid, inputs)
                    if satisfies_constraints_api(res, p_max_bar): best=res; hi=mid
                    else: lo=mid
                if best:
                    st.session_state.mud_params=inputs; st.session_state.mud_results=best
                    b1,b2,b3,b4=st.columns(4)
                    b1.metric("Q* (л/мин)", f"{best['Q_l_min']:.0f}")
                    b2.metric("V, м/с", f"{best['V']:.2f}")
                    b3.metric("ΔP, бар", f"{best['dp_total_bar']:.2f}")
                    b4.metric("ECD, кг/м³", f"{best['ecd']:.0f}")
                    st.success("Найден минимальный Q ✅")
                else:
                    st.error("Нет решения в диапазоне 50–10000 л/мин")

        except Exception as e:
            st.error(f"Ошибка: {e}")

    st.markdown("#### Экспорт / Отчёт / БД")
    if "mud_results" in st.session_state:
        df_res = pd.DataFrame([st.session_state.mud_results])
        st.download_button("⬇️ CSV", df_res.to_csv(index=False).encode("utf-8"), file_name="mud_api_results.csv", mime="text/csv")

    t = st.text_input("Название отчёта (API)", value="Mud — Hydraulics & Transport")
    notes = st.text_area("Примечания", value="API RP 13D расчёт + транспорт (оценка).")
    if st.button("Сформировать PDF (API)"):
        if "mud_results" not in st.session_state:
            st.error("Сначала рассчитайте/оптимизируйте.")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            make_engineer_report(filename=tmp.name, project_name="Mud_API13D", section_title=t,
                                 params=st.session_state.mud_params, segments=None,
                                 stations=df_res if "df_res" in locals() else None,
                                 mud=st.session_state.mud_results,
                                 plot_profile_path=None, plot_plan_path=None, extra_notes=notes)
            with open(tmp.name, "rb") as f:
                st.download_button("⬇️ PDF", f, file_name="Mud_API_Report.pdf", mime="application/pdf")
            try: os.unlink(tmp.name)
            except: pass

    init_db()
    pname = st.text_input("Project name (API)", value="Mud_API_1")
    if st.button("💾 Save Project (API)"):
        if "mud_params" in st.session_state and "mud_results" in st.session_state:
            save_project(pname, "mud_api", st.session_state.mud_params, segments=st.session_state.mud_results)
            st.success("Сохранено.")
        else:
            st.error("Сначала расчёт/оптимизация.")

# ===== Импорт XLSX (СП 341)
with tab_imp:
    st.subheader("Импорт Excel (лист «Отходы раствора по СП 341»)")
    uploaded = st.file_uploader("Загрузите XLS/XLSX", type=["xls","xlsx"])
    if uploaded:
        try:
            labels = parse_sp341_excel(uploaded)
            # Подхватим основные поля и запишем в session_state, чтобы СП-вкладка их увидела
            L_from = labels.get("Длина, м")
            k_from = labels.get("коэфф учета потерь раствора на техн. операции")
            if L_from is not None: st.session_state["L_sp341"] = float(L_from)
            if k_from is not None:
                k_val = float(k_from)
                st.session_state["k_add_sp341"] = (k_val-1.0) if k_val>1.0 else k_val

            # Диаметры (в мм)
            diam_list=[]
            for k in ["Диаметр пилота, м","Диаметр 1 расш, м","Диаметр 2 расш, м","Диаметр 3 расш.,м","Диаметр 4 расш.,м"]:
                if k in labels and labels[k]: diam_list.append(float(labels[k])*1000.0)
            if diam_list:
                df_pass = pd.DataFrame([{"Этап":"Пилот","Диаметр, мм":diam_list[0]}] +
                                       [{"Этап":f"Расширение {i}","Диаметр, мм":d} for i,d in enumerate(diam_list[1:], start=1)])
                st.session_state["sp341_df_pass"] = df_pass

            c_b = labels.get("Концентр. Бентонита кг/м3")
            c_p = labels.get("концентр. полимеры кг/м3")
            if c_b: st.session_state["conc_bent"] = float(c_b)
            if c_p: st.session_state["conc_poly"] = float(c_p)

            st.success("Импорт выполнен. Перейдите на вкладку «СП 341 — подбор» и нажмите «Рассчитать».")
            show=[]
            for kk in ["Диаметр скважины, м","Длина, м","Диаметр пилота, м","Диаметр 1 расш, м","Диаметр 2 расш, м",
                       "коэфф учета потерь раствора на техн. операции",
                       "Концентр. Бентонита кг/м3","концентр. полимеры кг/м3"]:
                if kk in labels: show.append({"Параметр": kk, "Значение": labels[kk]})
            if show:
                st.dataframe(pd.DataFrame(show), use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка импорта: {e}")

with st.expander("📚 Методика и источники"):
    st.markdown("""
**СП 341.1325800.2017** — плановый подбор раствора (объёмы/надбавки/концентрации).  
**API RP 13D** — реология/гидравлика (ΔP/ECD) + оценка транспорта шлама (TR, V_min, Q_cut vs Q_cap).  
Для ГНБ больших диаметров API-часть — инженерная оценка; отдельная HDD-гидравлика запланирована.
""")
