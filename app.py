import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="HW Case Checker (3 cases)", layout="wide")
st.title("📈 Prediction Case Checker")

# ===================== 1. 配置文件 =====================
CASE1_FILE = Path("case1_test.xlsx")   # 1天 = 24
CASE2_FILE = Path("case2_test.xlsx")   # 7天 = 168
CASE3_FILE = Path("case3_test.xlsx")   # 30天 ≈ 720
USER_FILE  = Path("users.xlsx")

# ===================== 2. 检查标准答案 =====================
missing = [f.name for f in [CASE1_FILE, CASE2_FILE, CASE3_FILE] if not f.exists()]
if missing:
    st.error(f"❗ 没找到标准答案文件: {', '.join(missing)}，请放到 app.py 同目录下。")
    st.stop()

# 只取每个文件的第一列作为真值
df_case1_truth = pd.read_excel(CASE1_FILE, index_col=0).iloc[:, 0].to_frame("truth")
df_case2_truth = pd.read_excel(CASE2_FILE, index_col=0).iloc[:, 0].to_frame("truth")
df_case3_truth = pd.read_excel(CASE3_FILE, index_col=0).iloc[:, 0].to_frame("truth")

# ===================== 3. 用户表 =====================
if not USER_FILE.exists():
    st.error("❗ 没找到 users.xlsx，请先创建一个包含 student_id, password 的 Excel。")
    st.stop()


def load_users() -> pd.DataFrame:
    dfu = pd.read_excel(USER_FILE)
    if "student_id" not in dfu.columns or "password" not in dfu.columns:
        st.error("users.xlsx 必须至少有 student_id, password 两列。")
        st.stop()
    # 成绩列没有就补
    for col in ["best_case1", "best_case2", "best_case3"]:
        if col not in dfu.columns:
            dfu[col] = np.nan
    return dfu


def save_users(dfu: pd.DataFrame):
    dfu.to_excel(USER_FILE, index=False)


users_df = load_users()

# ===================== 4. 工具函数 =====================
def read_uploaded(file):
    suf = Path(file.name).suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(file, index_col=0)
    elif suf == ".csv":
        return pd.read_csv(file, index_col=0)
    else:
        raise ValueError("只支持 .xlsx / .xls / .csv")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def update_user_best(student_id: str, col_name: str, score: float):
    dfu = load_users()
    idx = dfu[dfu["student_id"].astype(str) == str(student_id)].index
    if len(idx) != 1:
        st.error("当前用户不在 users.xlsx 中")
        return
    old = dfu.loc[idx[0], col_name]
    if pd.isna(old) or score < old:
        dfu.loc[idx[0], col_name] = score
        save_users(dfu)
        st.success("🎉 成绩已更新！")
    else:
        st.info(f"本次 RMSE = {score:.4f}，没有超过你当前最好成绩 {old:.4f}")


def show_leaderboard_html(case_col: str, title: str):
    users_df = load_users()
    rank_df = users_df.dropna(subset=[case_col]).copy()
    if rank_df.empty:
        st.info("还没有同学提交这个 case。")
        return

    rank_df = rank_df.sort_values(case_col, ascending=True).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    rank_df = rank_df.rename(columns={"student_id": "学号", case_col: "得分"})

    html_table = rank_df[["学号", "得分"]].to_html(
        classes="styled-table",
        justify="center",
        border=0
    )

    st.subheader(title)
    st.markdown(
        """
        <style>
        .styled-table {
            font-size: 22px;
            text-align: center;
            margin: 0 auto;
            border-collapse: collapse;
            width: 60%;
        }
        .styled-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            font-size: 24px;
            padding: 5px 8px;
        }
        .styled-table td {
            padding: 5px 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(html_table, unsafe_allow_html=True)


# ===================== 5. 登录 =====================
col1, col2, col3 = st.columns([2, 4, 2])
with col2:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    if not st.session_state.logged_in:
        st.subheader("🔐 登录")
        sid = st.text_input("学号")
        pwd = st.text_input("密码", type="password")
        if st.button("登录"):
            match = users_df[
                (users_df["student_id"].astype(str) == str(sid).strip()) &
                (users_df["password"].astype(str) == str(pwd).strip())
            ]
            if len(match) == 1:
                st.session_state.logged_in = True
                st.session_state.current_user = str(sid).strip()
                st.success(f"登录成功，欢迎 {sid}！")
            else:
                st.error("学号或密码错误。")
        st.stop()

st.info(f"当前登录：{st.session_state.current_user}")

# ===================== 6. 三列页面 =====================
c1, c2, c3 = st.columns(3)

# ---------- Case 1 ----------
with c1:
    st.subheader("📅 Case 1：预测 1 天（24条）")
    st.caption("上传：第一列 index，第二列你的预测；必须 24 行。")
    up1 = st.file_uploader("上传 Case 1 预测", type=["xlsx", "xls", "csv"], key="up_case1")

    if up1 is not None:
        try:
            df_stu = read_uploaded(up1)
        except Exception as e:
            st.error(f"读取失败：{e}")
        else:
            if df_stu.shape[1] != 1:
                st.error(f"你上传了 {df_stu.shape[1]} 列，我只要 1 列预测值。")
            else:
                truth = df_case1_truth["truth"].reset_index(drop=True)
                stu = df_stu.iloc[:, 0].reset_index(drop=True)

                if len(stu) != len(truth):
                    st.error(f"你上传了 {len(stu)} 行，但 Case 1 需要 {len(truth)} 行（1天=24条）。")
                else:
                    y_true = truth.values
                    y_pred = stu.values
                    r = rmse(y_true, y_pred)
                    st.success(f"✅ Case 1 RMSE = {r:.4f}")
                    update_user_best(st.session_state.current_user, "best_case1", r)

                    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=140)
                    ax.plot(y_true, label="Truth", linewidth=2, color="#1f77b4")
                    ax.plot(y_pred, label="Your Pred", linewidth=1.8, color="#d62728")
                    ax.set_title(f"Case 1 (RMSE={r:.3f})", pad=5)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout(pad=0.6)
                    st.pyplot(fig, use_container_width=False)

                    show_leaderboard_html("best_case1", "🏆 Case 1 排行榜")

# ---------- Case 2 ----------
with c2:
    st.subheader("📅 Case 2：预测 7 天（168条）")
    st.caption("上传：第一列 index，第二列你的预测；必须 168 行。")
    up2 = st.file_uploader("上传 Case 2 预测", type=["xlsx", "xls", "csv"], key="up_case2")

    if up2 is not None:
        try:
            df_stu = read_uploaded(up2)
        except Exception as e:
            st.error(f"读取失败：{e}")
        else:
            if df_stu.shape[1] != 1:
                st.error(f"你上传了 {df_stu.shape[1]} 列，我只要 1 列预测值。")
            else:
                truth = df_case2_truth["truth"].reset_index(drop=True)
                stu = df_stu.iloc[:, 0].reset_index(drop=True)

                if len(stu) != len(truth):
                    st.error(f"你上传了 {len(stu)} 行，但 Case 2 需要 {len(truth)} 行（7天=168条）。")
                else:
                    y_true = truth.values
                    y_pred = stu.values
                    r = rmse(y_true, y_pred)
                    st.success(f"✅ Case 2 RMSE = {r:.4f}")
                    update_user_best(st.session_state.current_user, "best_case2", r)

                    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=140)
                    ax.plot(y_true, label="Truth", linewidth=2, color="#1f77b4")
                    ax.plot(y_pred, label="Your Pred", linewidth=1.8, color="#d62728")
                    ax.set_title(f"Case 2 (RMSE={r:.3f})", pad=5)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout(pad=0.6)
                    st.pyplot(fig, use_container_width=False)

                    show_leaderboard_html("best_case2", "🏆 Case 2 排行榜")

# ---------- Case 3 ----------
with c3:
    st.subheader("📅 Case 3：预测 30 天（720条）")
    st.caption("上传：第一列 index，第二列你的预测；必须 720 行。")
    up3 = st.file_uploader("上传 Case 3 预测", type=["xlsx", "xls", "csv"], key="up_case3")

    if up3 is not None:
        try:
            df_stu = read_uploaded(up3)
        except Exception as e:
            st.error(f"读取失败：{e}")
        else:
            if df_stu.shape[1] != 1:
                st.error(f"你上传了 {df_stu.shape[1]} 列，我只要 1 列预测值。")
            else:
                truth = df_case3_truth["truth"].reset_index(drop=True)
                stu = df_stu.iloc[:, 0].reset_index(drop=True)

                if len(stu) != len(truth):
                    st.error(f"你上传了 {len(stu)} 行，但 Case 3 需要 {len(truth)} 行（30天≈720条）。")
                else:
                    y_true = truth.values
                    y_pred = stu.values
                    r = rmse(y_true, y_pred)
                    st.success(f"✅ Case 3 RMSE = {r:.4f}")
                    update_user_best(st.session_state.current_user, "best_case3", r)

                    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=140)
                    ax.plot(y_true, label="Truth", linewidth=2, color="#1f77b4")
                    ax.plot(y_pred, label="Your Pred", linewidth=1.8, color="#d62728")
                    ax.set_title(f"Case 3 (RMSE={r:.3f})", pad=5)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout(pad=0.6)
                    st.pyplot(fig, use_container_width=False)

                    show_leaderboard_html("best_case3", "🏆 Case 3 排行榜")
