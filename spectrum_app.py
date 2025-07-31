import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 核心計算函數 ---

def solve_sdof_newmark(accel_g, dt, period, damping_ratio):
    """
    使用 Newmark-β 方法 (線性加速度法) 求解單自由度系統的動力反應。
    返回相對位移、相對速度和相對加速度的完整歷時。
    """
    omega = 2 * np.pi / period
    c = 2 * damping_ratio * omega
    k = omega**2
    gamma = 0.5
    beta = 1.0 / 6.0
    g = 9.81
    p = -accel_g * g
    n_steps = len(p)
    u, v, a = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    a[0] = p[0]
    k_hat = k + (gamma / (beta * dt)) * c + 1 / (beta * dt**2)
    for i in range(n_steps - 1):
        dp_hat = (p[i+1] - p[i]) + (1/(beta*dt) + (gamma/beta)*c)*v[i] + (1/(2*beta) + (dt*gamma/(2*beta) - dt)*c)*a[i]
        du = dp_hat / k_hat
        dv = (gamma/(beta*dt))*du - (gamma/beta)*v[i] + dt*(1 - gamma/(2*beta))*a[i]
        da = (1/(beta*dt**2))*du - (1/(beta*dt))*v[i] - (1/(2*beta))*a[i]
        u[i+1], v[i+1], a[i+1] = u[i]+du, v[i]+dv, a[i]+da
    return u, v, a

def calculate_response_spectrum(time, accel_g, periods, damping_ratio):
    """計算 Sa, PSv, Sd 反應譜。"""
    dt = time[1] - time[0]
    g = 9.81
    sa_list, psv_list, sd_list = [], [], []
    accel_g_ms2 = accel_g * g
    progress_bar = st.progress(0, text="計算進度")
    for i, T in enumerate(periods):
        if T == 0:
            sa_g, psv_m_s, sd_m = np.max(np.abs(accel_g)), 0, 0
        else:
            omega = 2 * np.pi / T
            u_history, v_history, a_rel_history_ms2 = solve_sdof_newmark(accel_g, dt, T, damping_ratio)
            sd_m = np.max(np.abs(u_history))
            psv_m_s = sd_m * omega
            a_abs_history_ms2 = accel_g_ms2 + a_rel_history_ms2
            sa_m_s2 = np.max(np.abs(a_abs_history_ms2))
            sa_g = sa_m_s2 / g
        sa_list.append(sa_g); psv_list.append(psv_m_s); sd_list.append(sd_m)
        progress_bar.progress((i + 1) / len(periods), text=f"計算進度：{((i + 1) / len(periods))*100:.0f}%")
    progress_bar.empty()
    return np.array(sa_list), np.array(psv_list), np.array(sd_list)

def get_design_spectrum_values(periods, SDS, SD1):
    """計算並返回法規反應譜的數值陣列"""
    if SDS <= 0 or SD1 <= 0: return None
    Ts = SD1 / SDS
    T0 = 0.2 * Ts
    sa_design = [SDS * (0.4 + 0.6 * T / T0) if T < T0 else (SDS if T <= Ts else SD1 / T) for T in periods]
    return np.array(sa_design)

def plot_design_spectrum(ax, periods, SDS, SD1):
    """在給定的 axes 上繪製法規反應譜"""
    sa_design = get_design_spectrum_values(periods, SDS, SD1)
    if sa_design is not None:
        ax.plot(periods, sa_design, color='red', linestyle='-', lw=2, label=f'Design Spectrum (SDS={SDS}, SD1={SD1})')

def calculate_scaling_factor(periods, sa_unscaled, sa_design, target_period):
    """計算縮放係數"""
    T_lower, T_upper = 0.2 * target_period, 1.5 * target_period
    indices = np.where((periods >= T_lower) & (periods <= T_upper))
    if len(indices[0]) == 0: return None
    
    sa_unscaled_in_range = sa_unscaled[indices]
    sa_design_in_range = sa_design[indices]
    
    ratios = np.divide(0.9 * sa_design_in_range, sa_unscaled_in_range, out=np.zeros_like(sa_design_in_range), where=sa_unscaled_in_range != 0)
    factor_1 = np.max(ratios) if len(ratios) > 0 else 1.0
    
    mean_sa_unscaled = np.mean(sa_unscaled_in_range)
    mean_sa_design = np.mean(sa_design_in_range)
    factor_2 = (mean_sa_design / mean_sa_unscaled) if mean_sa_unscaled > 0 else 1.0
    
    return max(factor_1, factor_2)

def main_app():
    """
    主應用程式介面，在成功登入後顯示。
    """
    st.set_page_config(layout="wide", page_title="地震反應譜產生器")
    st.title("地震反應譜產生器 (SDOF)")
    st.markdown("上傳地震歷時檔案，繪製**絕對加速度 (Sa)**、**譜速度 (Sv)** 與 **譜位移 (Sd)** 反應譜，並可選用**反應譜縮放**功能。")

    # --- 初始化 Session State ---
    for key, default_val in [('analysis_dt', 0.005), ('last_file_id', None), ('results_ready', False), ('results_data', {})]:
        if key not in st.session_state: st.session_state[key] = default_val

    def invalidate_results():
        st.session_state.results_ready = False

    # --- 檔案上傳與 DT 建議邏輯 ---
    uploaded_file = st.file_uploader("請上傳您的地震歷時檔案 (加速度單位: gal)", type=['txt', 'csv', 'dat'])
    if uploaded_file:
        current_file_id = (uploaded_file.name, uploaded_file.size)
        if current_file_id != st.session_state.last_file_id:
            st.session_state.last_file_id = current_file_id
            invalidate_results()
            try:
                df_for_dt = pd.read_csv(uploaded_file, header=None, sep=r'\s+|,', engine='python')
                df_for_dt.columns = ['time', 'accel_gal']
                df_for_dt['time'] = pd.to_numeric(df_for_dt['time'], errors='coerce')
                df_for_dt.dropna(inplace=True)
                if not df_for_dt.empty:
                    time_orig_check = df_for_dt['time'].to_numpy()
                    original_dt_check = np.mean(np.diff(time_orig_check))
                    if original_dt_check > 0.001:
                        st.session_state.analysis_dt = max(0.001, min(original_dt_check / 2.0, 0.005))
                    st.rerun()
            except Exception: pass

    # --- 側邊欄 ---
    with st.sidebar:
        st.header("核心參數")
        st.caption("改變這些參數後需要重新計算")
        damping_ratio = st.slider("阻尼比 (Damping Ratio, ξ)", 0.0, 0.3, 0.05, 0.01, on_change=invalidate_results)
        T_min = st.number_input("最小週期 (Min Period, T)", 0.01, 10.0, 0.02, 0.01, on_change=invalidate_results)
        T_max = st.number_input("最大週期 (Max Period, T)", 0.1, 10.0, 4.0, 0.1, on_change=invalidate_results)
        T_steps = st.number_input("週期點數 (Period Points)", 50, 2000, 400, 10, on_change=invalidate_results)
        st.number_input("分析用時間間隔 (Analysis dt)", 0.001, 0.02, step=0.001, key='analysis_dt', format="%.4f", on_change=invalidate_results)
        use_log_sampling = st.checkbox("使用對數週期取樣", value=True, on_change=invalidate_results)

        st.header("顯示設定")
        st.caption("改變這些參數會即時更新圖表")
        x_scale = st.radio("X座標軸尺度", ("一般座標", "對數座標"), index=0)
        with st.expander("法規反應譜設定"):
            overlay_design_spec = st.checkbox("疊加法規反應譜 (Sa圖)")
            sds_val = st.number_input("設計譜加速度參數 SDS (g)", 0.0, value=1.0, step=0.01, disabled=not overlay_design_spec)
            sd1_val = st.number_input("設計譜加速度參數 SD1 (g-sec)", 0.0, value=0.55, step=0.01, disabled=not overlay_design_spec)

        st.header("地震歷時縮放 (選用)")
        st.caption("輸入目標週期以即時計算與更新")
        target_period = st.number_input("目標週期 T (sec)", 0.0, value=0.0, step=0.1, help="輸入大於0的目標週期以啟用縮放計算。需同時啟用上方的法規反應譜疊加。")
        
        st.divider()
        if st.button("登出"):
            st.session_state["logged_in"] = False
            st.rerun()

    if T_min >= T_max: st.sidebar.error("錯誤：最小週期必須小於最大週期。"); st.stop()

    # --- 產生週期陣列 ---
    if use_log_sampling:
        if T_min <= 0: T_min = 0.01
        periods = np.logspace(np.log10(T_min), np.log10(T_max), int(T_steps))
    else:
        periods = np.linspace(T_min, T_max, int(T_steps))

    # --- 主內容顯示區 ---
    if uploaded_file:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None, sep=r'\s+|,', engine='python')
            df.columns = ['time', 'accel_gal']
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df['accel_gal'] = pd.to_numeric(df['accel_gal'], errors='coerce')
            df.dropna(inplace=True)

            if df.shape[1] != 2 or df.empty: st.error("檔案讀取錯誤：請確保檔案為兩欄的數值資料。")
            else:
                if not st.session_state.get('initial_load_success', False):
                    st.success("檔案上傳並解析成功！")
                    st.session_state.initial_load_success = True

                time_orig, accel_orig_gal = df['time'].to_numpy(), df['accel_gal'].to_numpy()
                original_dt = np.mean(np.diff(time_orig))
                
                st.header("輸入地震資料")
                st.info(f"原始檔案 dt: **{original_dt:.4f} sec** | 分析用 dt: **{st.session_state.analysis_dt:.4f} sec**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(df.head(), height=210)
                    pga_gal = df['accel_gal'].abs().max()
                    pga_g = pga_gal / 981.0
                    st.metric("地表加速度峰值 (PGA)", f"{pga_gal:.2f} gal", f"{pga_g:.4f} g", delta_color="off")
                with col2:
                    fig_th, ax_th = plt.subplots(figsize=(10, 4))
                    ax_th.plot(time_orig, accel_orig_gal, lw=0.8); ax_th.set_xlabel("Time (sec)"); ax_th.set_ylabel("Acceleration (gal)"); ax_th.set_title("Ground Acceleration Time History"); ax_th.grid(True)
                    st.pyplot(fig_th)

                if st.button("計算並繪製反應譜", type="primary", use_container_width=True):
                    new_time_vector = np.arange(time_orig[0], time_orig[-1], st.session_state.analysis_dt)
                    accel_interp_gal = np.interp(new_time_vector, time_orig, accel_orig_gal)
                    accel_interp_g = accel_interp_gal / 981.0
                    with st.spinner('正在進行動力分析計算，請稍候...'):
                        sa_values, psv_values, sd_values = calculate_response_spectrum(new_time_vector, accel_interp_g, periods, damping_ratio)
                    
                    st.session_state.results_data = {
                        'sa': sa_values, 'psv': psv_values, 'sd': sd_values, 
                        'periods': periods, 'time_vector': new_time_vector, 
                        'accel_gal': accel_interp_gal
                    }
                    st.session_state.results_ready = True
                    st.rerun()

        except Exception as e: st.error(f"檔案處理時發生錯誤: {e}")

    # --- 繪圖區 ---
    if st.session_state.results_ready:
        st.header("反應譜分析結果")
        results = st.session_state.results_data
        res_periods = results['periods']
        
        scaling_factor = 1.0
        if target_period > 0 and overlay_design_spec:
            sa_design = get_design_spectrum_values(res_periods, sds_val, sd1_val)
            if sa_design is not None:
                factor = calculate_scaling_factor(res_periods, results['sa'], sa_design, target_period)
                if factor:
                    scaling_factor = factor
                    st.metric("計算縮放係數 (Scaling Factor)", f"{scaling_factor:.4f}", f"目標週期 T = {target_period} sec")
                    
                    scaled_accel_gal = results['accel_gal'] * scaling_factor
                    scaled_df = pd.DataFrame({'time': results['time_vector'], 'scaled_accel_gal': scaled_accel_gal})
                    csv_string = scaled_df.to_csv(index=False, sep='\t', header=False, float_format='%.6f')
                    st.download_button("下載縮放後地震歷時 (.txt)", data=csv_string, file_name=f"scaled_th_{uploaded_file.name}", mime="text/plain")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. 絕對加速度反應譜 (Sa)")
            fig_sa, ax_sa = plt.subplots(figsize=(7, 5))
            ax_sa.plot(res_periods, results['sa'], lw=1.5, color='blue', label='Response Spectrum')
            if scaling_factor != 1.0:
                ax_sa.plot(res_periods, results['sa'] * scaling_factor, lw=2, color='orange', linestyle='--', label='Scaled Spectrum')
            if overlay_design_spec:
                plot_design_spectrum(ax_sa, res_periods, sds_val, sd1_val)
            if target_period > 0 and overlay_design_spec:
                T_lower, T_upper = 0.2 * target_period, 1.5 * target_period
                ax_sa.axvspan(T_lower, T_upper, color='lightgreen', alpha=0.3, zorder=0, label='Scaling Range')
            ax_sa.set_xlabel("Period T (sec)"); ax_sa.set_ylabel("Sa (g)")
            ax_sa.set_title(f"Absolute Acceleration Spectrum (ξ={damping_ratio:.2f})")
            ax_sa.grid(True, which='both', linestyle='--')
            max_y = 0
            if len(ax_sa.get_lines()) > 0:
                for line in ax_sa.get_lines():
                    y_data = line.get_ydata();
                    if len(y_data) > 0: max_y = max(max_y, np.max(y_data))
            ax_sa.set_ylim(bottom=0, top=max_y * 1.05 if max_y > 0 else 1)
            if x_scale == "對數座標": ax_sa.set_xscale('log'); ax_sa.set_xlim(left=T_min if T_min > 0 else 0.01)
            else: ax_sa.set_xscale('linear'); ax_sa.set_xlim(left=0)
            ax_sa.legend()
            st.pyplot(fig_sa)
            buf = io.BytesIO(); fig_sa.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("下載圖表 (.png)", data=buf, file_name="spectrum_Sa.png", mime="image/png", key="download_sa")

        with col2:
            st.subheader("2. 譜速度反應譜 (Sv)")
            fig_sv, ax_sv = plt.subplots(figsize=(7, 5))
            sv_cm_s = results['psv'] * 100
            ax_sv.plot(res_periods, sv_cm_s, lw=1.5, color='blue', label='Response Spectrum')
            ax_sv.set_xlabel("Period T (sec)"); ax_sv.set_ylabel("Sv (cm/s)")
            ax_sv.set_title(f"Spectral Velocity (ξ={damping_ratio:.2f})")
            ax_sv.grid(True, which='both', linestyle='--')
            max_y = np.max(sv_cm_s) if len(sv_cm_s) > 0 else 0
            ax_sv.set_ylim(bottom=0, top=max_y * 1.05 if max_y > 0 else 1)
            if x_scale == "對數座標": ax_sv.set_xscale('log'); ax_sv.set_xlim(left=T_min if T_min > 0 else 0.01)
            else: ax_sv.set_xscale('linear'); ax_sv.set_xlim(left=0)
            st.pyplot(fig_sv)
            buf = io.BytesIO(); fig_sv.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("下載圖表 (.png)", data=buf, file_name="spectrum_PSv.png", mime="image/png", key="download_psv")

        col3, _ = st.columns(2)
        with col3:
            st.subheader("3. 譜位移反應譜 (Sd)")
            fig_sd, ax_sd = plt.subplots(figsize=(7, 5))
            ax_sd.plot(res_periods, results['sd'], lw=1.5, color='blue', label='Response Spectrum')
            ax_sd.set_xlabel("Period T (sec)"); ax_sd.set_ylabel("Sd (m)")
            ax_sd.set_title(f"Spectral Displacement (ξ={damping_ratio:.2f})")
            ax_sd.grid(True, which='both', linestyle='--')
            max_y = np.max(results['sd']) if len(results['sd']) > 0 else 0
            ax_sd.set_ylim(bottom=0, top=max_y * 1.05 if max_y > 0 else 1)
            if x_scale == "對數座標": ax_sd.set_xscale('log'); ax_sd.set_xlim(left=T_min if T_min > 0 else 0.01)
            else: ax_sd.set_xscale('linear'); ax_sd.set_xlim(left=0)
            st.pyplot(fig_sd)
            buf = io.BytesIO(); fig_sd.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("下載圖表 (.png)", data=buf, file_name="spectrum_Sd.png", mime="image/png", key="download_sd")

    else:
        if uploaded_file:
            st.warning("核心參數已變更或已上傳新檔案，請點擊「計算並繪製反應譜」按鈕以更新結果。")

def check_password():
    """顯示登入頁面並驗證密碼"""
    st.set_page_config(layout="centered", page_title="登入")
    st.title("🔐 登入")
    st.write("請輸入密碼以使用本應用程式。")
    
    password_input = st.text_input("密碼", type="password")
    
    # *** CHANGE FOR DEPLOYMENT ***
    # 從寫死密碼改為讀取 st.secrets
    # 在本機測試時，需建立 .streamlit/secrets.toml 檔案
    # 在雲端部署時，需在 Streamlit Community Cloud 的設定中新增 Secret
    try:
        correct_password = st.secrets["APP_PASSWORD"]
    except FileNotFoundError:
        st.error("錯誤：找不到密碼設定檔 (.streamlit/secrets.toml)。請參考部署指南建立它。")
        return
    except KeyError:
        st.error("錯誤：密碼設定檔中找不到 'APP_PASSWORD'。請確認您的設定。")
        return

    if st.button("登入"):
        if password_input == correct_password:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("密碼錯誤，請重新輸入。")

# --- 主程式執行邏輯 ---
if not st.session_state.get("logged_in", False):
    check_password()
else:
    main_app()
