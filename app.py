import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import boxcox
import plotly.graph_objects as go

from src.data_io import generate_sample_data, parse_upload
from src.capability import (
    normality_test, boxcox_transform, process_capability,
    capability_grade, plot_boxplot, plot_histogram, plot_qq, plot_process_capability,
)
from src.control_charts import (
    xbar_r_chart, xbar_s_chart, imr_chart,
    np_chart, p_chart, c_chart, u_chart,
    detect_anomalies, remove_anomalies,
    plot_variable_control_chart, _single_control_chart, recommend_chart,
)

st.set_page_config(
    page_title="스마트제조 공정분석 대시보드",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 스마트제조 공정분석 대시보드")
st.caption("공정능력분석(Cp/Cpk/Pp/Ppk) + 통계적공정관리(관리도) 통합 웹앱")

# ── Session state init ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "LSL" not in st.session_state:
    st.session_state.LSL = None
if "USL" not in st.session_state:
    st.session_state.USL = None
if "sg_col" not in st.session_state:
    st.session_state.sg_col = None
if "val_col" not in st.session_state:
    st.session_state.val_col = None

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — 데이터 입력
# ══════════════════════════════════════════════════════════════════════════════
tab0, tab1, tab2, tab3 = st.tabs(["📂 데이터 입력", "📊 공정능력분석", "📈 통계적공정관리", "🖥️ 종합 대시보드"])

with tab0:
    st.header("데이터 입력")
    mode = st.radio("입력 방식", ["샘플 데이터 생성", "CSV/Excel 업로드", "직접 편집"], horizontal=True)

    if mode == "샘플 데이터 생성":
        col1, col2, col3 = st.columns(3)
        with col1:
            var_name = st.text_input("측정 변수명", value="thickness")
            sg_name = st.text_input("부분군 컬럼명", value="line")
        with col2:
            target = st.number_input("목표값 (target)", value=0.5, format="%.4f")
            tolerance = st.number_input("허용오차 (tolerance)", value=0.05, format="%.4f")
        with col3:
            num_sg = st.number_input("부분군 수", min_value=2, max_value=100, value=6)
            sg_size = st.number_input("부분군 크기", min_value=1, max_value=100, value=30)
            sg_std = st.number_input("부분군 표준편차", value=0.01, format="%.4f")

        col4, col5 = st.columns(2)
        with col4:
            mean_shift_min = st.number_input("평균 이동 최소", value=-0.01, format="%.4f")
        with col5:
            mean_shift_max = st.number_input("평균 이동 최대", value=0.01, format="%.4f")
        seed = st.number_input("랜덤 시드", value=42, min_value=0)

        if st.button("데이터 생성", type="primary"):
            df, LSL, USL = generate_sample_data(
                var_name=var_name, target=target, tolerance=tolerance,
                sg_name=sg_name, num_sg=int(num_sg), sg_size=int(sg_size),
                sg_std=sg_std, mean_shift=(mean_shift_min, mean_shift_max),
                seed=int(seed),
            )
            st.session_state.df = df
            st.session_state.LSL = LSL
            st.session_state.USL = USL
            st.session_state.sg_col = sg_name
            st.session_state.val_col = var_name
            st.success(f"데이터 생성 완료: {len(df)}행, LSL={LSL:.4f}, USL={USL:.4f}")

    elif mode == "CSV/Excel 업로드":
        uploaded = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])
        if uploaded:
            df_raw = parse_upload(uploaded)
            st.dataframe(df_raw.head(10), use_container_width=True)
            cols = df_raw.columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                sg_col = st.selectbox("부분군 컬럼", cols)
                val_col = st.selectbox("측정값 컬럼", cols, index=min(1, len(cols)-1))
            with col2:
                LSL = st.number_input("LSL (규격 하한)", value=0.0, format="%.4f")
                USL = st.number_input("USL (규격 상한)", value=1.0, format="%.4f")
            if st.button("적용", type="primary"):
                st.session_state.df = df_raw
                st.session_state.LSL = LSL
                st.session_state.USL = USL
                st.session_state.sg_col = sg_col
                st.session_state.val_col = val_col
                st.success("데이터 로드 완료!")

    else:  # 직접 편집
        if st.session_state.df is not None:
            edited = st.data_editor(st.session_state.df, use_container_width=True, num_rows="dynamic")
            if st.button("편집 내용 저장", type="primary"):
                st.session_state.df = edited
                st.success("저장됨.")
        else:
            st.info("먼저 다른 방식으로 데이터를 로드하세요.")

    if st.session_state.df is not None:
        st.divider()
        st.subheader("현재 데이터 미리보기")
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("전체 행 수", len(df))
        col2.metric("부분군 수", df[st.session_state.sg_col].nunique() if st.session_state.sg_col else "-")
        col3.metric("LSL", f"{st.session_state.LSL:.4f}" if st.session_state.LSL is not None else "-")
        col4.metric("USL", f"{st.session_state.USL:.4f}" if st.session_state.USL is not None else "-")
        st.dataframe(df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 공정능력분석
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("공정능력분석 (Process Capability Analysis)")

    if st.session_state.df is None:
        st.warning("데이터 입력 탭에서 데이터를 먼저 로드하세요.")
    else:
        df = st.session_state.df
        sg_col = st.session_state.sg_col
        val_col = st.session_state.val_col
        LSL = st.session_state.LSL
        USL = st.session_state.USL

        data_series = df[val_col].dropna()

        # ── 정규성 검정 ──────────────────────────────────────────────────────
        st.subheader("1. 정규성 검정")
        stat, p_val, is_normal = normality_test(data_series)
        col1, col2 = st.columns(2)
        with col1:
            status = "✅ 정규성 만족" if is_normal else "⚠️ 정규성 불만족"
            st.metric("Shapiro-Wilk p-value", f"{p_val:.4f}", delta=status)
            st.caption("p ≥ 0.05: 정규분포 따름 / p < 0.05: 정규분포 아님")
        with col2:
            st.plotly_chart(plot_qq(data_series, f"Q-Q Plot of {val_col}"), use_container_width=False)

        # ── Box-Cox 변환 (정규성 불만족 시) ──────────────────────────────────
        use_transform = False
        df_analysis = df.copy()
        LSL_analysis, USL_analysis = LSL, USL

        if not is_normal:
            st.warning("정규성 불만족 → Box-Cox 변환을 적용할 수 있습니다.")
            use_transform = st.checkbox("Box-Cox 변환 적용", value=True)
            if use_transform:
                transformed, lam, stat_t, p_t = boxcox_transform(data_series)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("최적 λ (lambda)", f"{lam:.4f}")
                    st.metric("변환 후 Shapiro-Wilk p", f"{p_t:.4f}",
                              delta="✅ 만족" if p_t >= 0.05 else "⚠️ 여전히 불만족")
                with col2:
                    st.plotly_chart(plot_qq(pd.Series(transformed), "Q-Q Plot (Box-Cox 변환 후)"),
                                    use_container_width=False)
                df_analysis = df_analysis.copy()
                df_analysis[val_col] = np.nan
                mask = df[val_col].notna()
                df_analysis.loc[mask, val_col] = transformed
                # 변환 스케일 규격
                if lam != 0:
                    LSL_analysis = (LSL * lam + 1) ** (1/lam) if False else LSL_analysis
                st.info("※ Box-Cox 변환 후 규격 한계(LSL/USL)는 동일 스케일 유지 (근사)")

        # ── 분포 시각화 ───────────────────────────────────────────────────────
        st.subheader("2. 분포 시각화")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.plotly_chart(plot_boxplot(df, sg_col, val_col, LSL, USL), use_container_width=True)
        with vcol2:
            st.plotly_chart(plot_histogram(df, sg_col, val_col, LSL, USL), use_container_width=True)

        # ── 공정능력지수 계산 ─────────────────────────────────────────────────
        st.subheader("3. 공정능력지수 (Cp / Cpk / Pp / Ppk)")
        if LSL >= USL:
            st.error("⚠️ LSL이 USL보다 크거나 같습니다. 규격 한계를 확인하세요.")
        cap = process_capability(df_analysis, sg_col, val_col, LSL_analysis, USL_analysis)

        col1, col2, col3, col4 = st.columns(4)
        for c, key, label in [
            (col1, "Cp", "Cp (단기 정밀도)"),
            (col2, "Cpk", "Cpk (단기 치우침 포함)"),
            (col3, "Pp", "Pp (장기 정밀도)"),
            (col4, "Ppk", "Ppk (장기 치우침 포함)"),
        ]:
            val = cap[key]
            grade, color = capability_grade(val)
            c.metric(label, f"{val:.4f}")
            c.markdown(f"<span style='color:{color};font-weight:bold'>{grade}</span>", unsafe_allow_html=True)

        st.caption("판정: Cp/Cpk ≥ 1.67 매우 우수 | ≥ 1.33 우수 | ≥ 1.0 보통 | < 1.0 미흡")

        # ── 통합 공정능력 그래프 ──────────────────────────────────────────────
        st.subheader("4. 공정능력분석 종합 그래프")
        st.plotly_chart(
            plot_process_capability(df, sg_col, val_col, LSL, USL, cap),
            use_container_width=True,
        )

        # ── 수치 요약 ─────────────────────────────────────────────────────────
        with st.expander("상세 통계"):
            st.write(f"**전체 평균 (x̄):** {cap['x_bar']:.4f}")
            st.write(f"**군내 표준편차 (σ_within):** {cap['sigma_within']:.4f}")
            st.write(f"**전체 표준편차 (σ_overall):** {cap['sigma_overall']:.4f}")
            sg_summary = df.groupby(sg_col)[val_col].agg(["count", "mean", "std"]).round(4)
            sg_summary.columns = ["n", "평균", "표준편차"]
            st.dataframe(sg_summary, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 통계적공정관리
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("통계적공정관리 (Statistical Process Control)")

    if st.session_state.df is None:
        st.warning("데이터 입력 탭에서 데이터를 먼저 로드하세요.")
    else:
        df = st.session_state.df
        sg_col = st.session_state.sg_col
        val_col = st.session_state.val_col

        # ── 관리도 선택 ───────────────────────────────────────────────────────
        st.subheader("관리도 설정")
        col1, col2 = st.columns(2)
        with col1:
            data_type = st.selectbox("데이터 유형", ["계량형 (연속)", "계수형 (불량품)", "계수형 (결점수)"])
        with col2:
            avg_sg_size = int(df.groupby(sg_col).size().mean())
            rec = recommend_chart(avg_sg_size, data_type)
            st.info(f"**추천 관리도:** {rec} (평균 부분군 크기: {avg_sg_size})")

        if data_type == "계량형 (연속)":
            chart_type = st.selectbox("관리도 종류", ["Xbar-R", "Xbar-S", "I-MR"])
        elif data_type == "계수형 (불량품)":
            chart_type = st.selectbox("관리도 종류", ["NP", "P"])
        else:
            chart_type = st.selectbox("관리도 종류", ["C", "U"])

        # ── 1단계: 초기 관리도 ───────────────────────────────────────────────
        st.subheader("1단계: 초기 관리도")
        try:
            if chart_type == "Xbar-R":
                charts = xbar_r_chart(df, sg_col, val_col)
                fig = plot_variable_control_chart(charts, "Xbar-R", val_col)
                st.plotly_chart(fig, use_container_width=True)
                anomalies1 = detect_anomalies(charts[0])
                anomalies2 = detect_anomalies(charts[1])
                anomaly_sgs = sorted(set(anomalies1 + anomalies2))

            elif chart_type == "Xbar-S":
                charts = xbar_s_chart(df, sg_col, val_col)
                fig = plot_variable_control_chart(charts, "Xbar-S", val_col)
                st.plotly_chart(fig, use_container_width=True)
                anomalies1 = detect_anomalies(charts[0])
                anomalies2 = detect_anomalies(charts[1])
                anomaly_sgs = sorted(set(anomalies1 + anomalies2))

            elif chart_type == "I-MR":
                charts = imr_chart(df, sg_col, val_col)
                fig = plot_variable_control_chart(charts, "I-MR", val_col)
                st.plotly_chart(fig, use_container_width=True)
                anomalies1 = detect_anomalies(charts[0])
                anomalies2 = detect_anomalies(charts[1])
                anomaly_sgs = sorted(set(anomalies1 + anomalies2))

            elif chart_type == "NP":
                chart = np_chart(df, sg_col, val_col)
                fig = _single_control_chart(chart, f"NP Control Chart of {val_col}", "NP")
                st.plotly_chart(fig, use_container_width=True)
                anomaly_sgs = detect_anomalies(chart)

            elif chart_type == "P":
                chart = p_chart(df, sg_col, val_col)
                fig = _single_control_chart(chart, f"P Control Chart of {val_col}", "P")
                st.plotly_chart(fig, use_container_width=True)
                anomaly_sgs = detect_anomalies(chart)

            elif chart_type == "C":
                chart = c_chart(df, sg_col, val_col)
                fig = _single_control_chart(chart, f"C Control Chart of {val_col}", "C")
                st.plotly_chart(fig, use_container_width=True)
                anomaly_sgs = detect_anomalies(chart)

            else:  # U
                chart = u_chart(df, sg_col, val_col)
                fig = _single_control_chart(chart, f"U Control Chart of {val_col}", "U")
                st.plotly_chart(fig, use_container_width=True)
                anomaly_sgs = detect_anomalies(chart)

            # ── 이상치 요약 ───────────────────────────────────────────────────
            if anomaly_sgs:
                sgs = df[sg_col].unique()
                bad_names = [str(sgs[i]) for i in anomaly_sgs if i < len(sgs)]
                st.warning(f"이상 부분군 감지: {len(bad_names)}개 → {', '.join(bad_names)}")
            else:
                st.success("이상 부분군 없음 — 공정 관리 상태 양호")

            # ── 2단계: 이상 제거 후 재계산 ───────────────────────────────────
            if anomaly_sgs:
                st.subheader("2단계: 이상 부분군 제거 후 관리도")
                df_clean = remove_anomalies(df, sg_col, anomaly_sgs)
                st.caption(f"제거 후 {df[sg_col].nunique()} → {df_clean[sg_col].nunique()} 부분군")
                try:
                    if chart_type == "Xbar-R":
                        charts2 = xbar_r_chart(df_clean, sg_col, val_col)
                        fig2 = plot_variable_control_chart(charts2, "Xbar-R", val_col)
                    elif chart_type == "Xbar-S":
                        charts2 = xbar_s_chart(df_clean, sg_col, val_col)
                        fig2 = plot_variable_control_chart(charts2, "Xbar-S", val_col)
                    elif chart_type == "I-MR":
                        charts2 = imr_chart(df_clean, sg_col, val_col)
                        fig2 = plot_variable_control_chart(charts2, "I-MR", val_col)
                    elif chart_type == "NP":
                        charts2 = np_chart(df_clean, sg_col, val_col)
                        fig2 = _single_control_chart(charts2, f"NP Chart (수정)", "NP")
                    elif chart_type == "P":
                        charts2 = p_chart(df_clean, sg_col, val_col)
                        fig2 = _single_control_chart(charts2, f"P Chart (수정)", "P")
                    elif chart_type == "C":
                        charts2 = c_chart(df_clean, sg_col, val_col)
                        fig2 = _single_control_chart(charts2, f"C Chart (수정)", "C")
                    else:
                        charts2 = u_chart(df_clean, sg_col, val_col)
                        fig2 = _single_control_chart(charts2, f"U Chart (수정)", "U")
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"2단계 관리도 계산 오류: {e}")

        except Exception as e:
            st.error(f"관리도 계산 오류: {e}")
            st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 종합 대시보드
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("종합 대시보드")

    if st.session_state.df is None:
        st.warning("데이터 입력 탭에서 데이터를 먼저 로드하세요.")
    else:
        df = st.session_state.df
        sg_col = st.session_state.sg_col
        val_col = st.session_state.val_col
        LSL = st.session_state.LSL
        USL = st.session_state.USL

        # ── KPI 카드 ──────────────────────────────────────────────────────────
        try:
            if LSL >= USL:
                st.error("⚠️ LSL ≥ USL — 규격 한계를 확인하세요.")
            cap = process_capability(df, sg_col, val_col, LSL, USL)
            st.subheader("공정능력 KPI")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("평균 (x̄)", f"{cap['x_bar']:.4f}")
            for c, key in [(col2, "Cp"), (col3, "Cpk"), (col4, "Pp"), (col5, "Ppk")]:
                val = cap[key]
                grade, _ = capability_grade(val)
                c.metric(key, f"{val:.4f}", delta=grade)
        except Exception as e:
            st.error(f"공정능력 계산 오류: {e}")

        st.divider()

        # ── 관리도 + 능력 통합 뷰 ────────────────────────────────────────────
        st.subheader("공정 현황 통합 뷰")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**공정능력 분석 그래프**")
            try:
                cap = process_capability(df, sg_col, val_col, LSL, USL)
                fig_cap = plot_process_capability(df, sg_col, val_col, LSL, USL, cap)
                st.plotly_chart(fig_cap, use_container_width=True, key="dash_cap")
            except Exception as e:
                st.error(str(e))

        with col_right:
            st.markdown("**Xbar-R 관리도**")
            try:
                avg_n = int(df.groupby(sg_col).size().mean())
                if avg_n == 1:
                    charts = imr_chart(df, sg_col, val_col)
                    fig_spc = plot_variable_control_chart(charts, "I-MR", val_col)
                elif avg_n < 10:
                    charts = xbar_r_chart(df, sg_col, val_col)
                    fig_spc = plot_variable_control_chart(charts, "Xbar-R", val_col)
                else:
                    charts = xbar_s_chart(df, sg_col, val_col)
                    fig_spc = plot_variable_control_chart(charts, "Xbar-S", val_col)
                st.plotly_chart(fig_spc, use_container_width=True, key="dash_spc")
            except Exception as e:
                st.error(str(e))

        # ── 공정 상태 판정 ────────────────────────────────────────────────────
        st.divider()
        st.subheader("공정 상태 종합 판정")
        try:
            cap = process_capability(df, sg_col, val_col, LSL, USL)
            cpk = cap["Cpk"]
            grade, color = capability_grade(cpk)

            # 관리 상태 (Xbar 기준)
            if avg_n >= 1:
                try:
                    if avg_n == 1:
                        c1, c2 = imr_chart(df, sg_col, val_col)
                    elif avg_n < 10:
                        c1, c2 = xbar_r_chart(df, sg_col, val_col)
                    else:
                        c1, c2 = xbar_s_chart(df, sg_col, val_col)
                    anom = detect_anomalies(c1)
                    spc_status = "관리 이탈" if anom else "관리 상태"
                    spc_color = "red" if anom else "green"
                except:
                    spc_status, spc_color = "계산 불가", "gray"

            col1, col2 = st.columns(2)
            col1.markdown(
                f"**공정능력 (Cpk={cpk:.4f}):** "
                f"<span style='color:{color};font-size:18px;font-weight:bold'>{grade}</span>",
                unsafe_allow_html=True,
            )
            col2.markdown(
                f"**SPC 관리 상태:** "
                f"<span style='color:{spc_color};font-size:18px;font-weight:bold'>{spc_status}</span>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(str(e))
