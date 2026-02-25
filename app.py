import streamlit as st
import pandas as pd
import time
from streamlit_gsheets import GSheetsConnection

# --- 0. 数据库连接与初始化 ---
# 在 Streamlit Cloud 的 Secrets 中配置表格链接
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data():
    # ttl=0 确保每次读取都是最新的云端数据
    return conn.read(ttl=0)

def save_data(df):
    conn.update(data=df)
    st.cache_data.clear()

# --- 初始化全局状态 ---
def init_state():
    if 'page' not in st.session_state:
        st.session_state.page = "login"
    if 'user' not in st.session_state:
        st.session_state.user = ""
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'learned_modules' not in st.session_state:
        st.session_state.learned_modules = set()
    if 'step' not in st.session_state:
        st.session_state.step = 0

init_state()

# --- 样式美化 ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; }
    .algo-card { 
        padding: 20px; border-radius: 15px; background: white; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
        border: 1px solid #e0e0e0; margin-bottom: 20px;
    }
    .rank-1 { color: #FFD700; font-weight: bold; font-size: 20px; }
    .rank-2 { color: #C0C0C0; font-weight: bold; }
    .rank-3 { color: #CD7F32; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 教师后台管理 (侧边栏) ---
with st.sidebar:
    st.title("⚙️ 管理面板")
    admin_pwd = st.text_input("管理员密码", type="password")
    if admin_pwd == "666888": # 你可以修改自己的密码
        st.subheader("👨‍🏫 教师后台数据管理")
        all_data = get_data()
        edited_df = st.data_editor(all_data, num_rows="dynamic")
        if st.button("💾 保存修改到云端"):
            save_data(edited_df)
            st.success("云端数据同步成功！")

# --- 1. 登录页面 ---
if st.session_state.page == "login":
    st.title("🌟 智能课堂互动系统")
    name = st.text_input("请输入姓名以登录")
    if st.button("进入教室"):
        if name:
            st.session_state.user = name
            # 登录时从云端同步该学生的旧积分
            df = get_data()
            if name in df["学生"].values:
                st.session_state.score = int(df[df["学生"] == name]["总积分"].iloc[0])
            else:
                # 新学生自动注册
                new_user = pd.DataFrame([{"学生": name, "总积分": 0}])
                save_data(pd.concat([df, new_user], ignore_index=True))
            st.session_state.page = "dashboard"
            st.rerun()

# --- 2. 仪表盘 ---
elif st.session_state.page == "dashboard":
    st.title(f"👋 你好, {st.session_state.user}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("我的当前积分", st.session_state.score)
    with col2:
        if st.button("🏆 查看班级排行榜"):
            st.session_state.page = "leaderboard"
            st.rerun()

    st.subheader("📚 课程知识地图")
    with st.expander("📍 路径规划算法板块", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="algo-card"><h3>Dijkstra 算法</h3></div>', unsafe_allow_html=True)
            if st.button("进入学习", key="dij"):
                st.session_state.current_algo = "Dijkstra"; st.session_state.page = "learning"; st.session_state.step = 0; st.rerun()
        with c2:
            st.markdown('<div class="algo-card"><h3>A* 算法</h3></div>', unsafe_allow_html=True)
            if st.button("进入学习", key="astar"):
                st.session_state.current_algo = "AStar"; st.session_state.page = "learning"; st.session_state.step = 0; st.rerun()

    st.divider()
    st.warning("🔔 限时随堂测试已发布")
    if st.button("🚀 开始进入答题模式"):
        st.session_state.page = "quiz"; st.session_state.quiz_step = 1; st.session_state.quiz_score = 0; st.session_state.start_time = time.time(); st.rerun()

# --- 3. 教学模式 ---
elif st.session_state.page == "learning":
    algo = st.session_state.current_algo
    steps = {
        "AStar": [
            {"t": "核心概念：贪心算法", "c": "贪心算法选择当前最优路径...", "img": "💡"},
            {"t": "启发式搜索", "c": "A* 引入了 h(n) 预估代价。", "img": "🔍"}
        ],
        "Dijkstra": [
            {"t": "算法简介", 
             "c": ("迪杰斯特拉算法（Dijkstra's Algorithm）是由荷兰计算机科学家艾兹赫尔·戴克斯特拉在 1956 年提出的一种单源最短路径算法。\n\n"
                  "该算法的核心思想是贪心策略，每次都选择当前已知距离源点最近的一个节点，并以此节点为基准去更新它相邻节点的距离，从而在一个包含多个节点和带有非负权重边的图中，找到从一个指定的“源点”到图中所有其他节点的最短距离。\n\n"
                  "我们将以下图为例，学习应用该算法。"), 
             "img": "assets/dijkstra_demo1.png"},
             {"t": "启发式搜索", "c": "A* 引入了 h(n) 预估代价。", "img": "🔍"}
        ]
    }

    if "step" not in st.session_state:
        st.session_state.step = 0
        
    data = steps[algo][st.session_state.step]

    # 3. 渲染当前步骤
    st.subheader(f"正在学习: {algo}")
    st.divider()

    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header(data['t'])
        st.write(data['c'])
    
    with col2:
        img_path = data['img']
        # 优化判断逻辑
        if "/" in img_path or img_path.endswith(('.png', '.jpg', '.jpeg')):
            try:
                st.image(img_path, use_container_width=True)
            except Exception as e:
                st.error(f"图片加载失败，请检查 GitHub 仓库中是否存在: {img_path}")
        else:
            # 如果是表情符号
            st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{img_path}</h1>", unsafe_allow_html=True)

    st.divider()

    # 4. 底部导航按钮
    col_l, col_m, col_r = st.columns([1, 1, 1])
    with col_l:
        if st.session_state.step > 0:
            if st.button("⬅️ 上一步"):
                st.session_state.step -= 1
                st.rerun()
    
    with col_r:
        if st.session_state.step < len(steps[algo]) - 1:
            if st.button("下一步 ➡️"):
                st.session_state.step += 1
                st.rerun()
        elif algo not in st.session_state.learned_modules:
            if st.button("🏁 知识检验"):
                st.session_state.page = "learning_test"
                st.rerun()
        else:
            if st.button("🏠 返回首页"):
                st.session_state.page = "dashboard"
                st.rerun()
# --- 4. 知识检验 ---
elif st.session_state.page == "learning_test":
    st.header("🎯 知识检验")
    q = st.radio("A* 公式中 h 代表什么？", ["起点距离", "预估终点距离"])
    if st.button("提交答案"):
        if "预估" in q:
            st.session_state.score += 50
            st.session_state.learned_modules.add(st.session_state.current_algo)
            # 学习完立刻同步积分到云端
            df = get_data()
            df.loc[df["学生"] == st.session_state.user, "总积分"] = st.session_state.score
            save_data(df)
            st.success("获得 50 积分！已保存到云端。")
        time.sleep(1); st.session_state.page = "dashboard"; st.rerun()

# --- 5. 课堂答题 (锁定模式) ---
elif st.session_state.page == "quiz":
    elapsed = time.time() - st.session_state.start_time
    remaining = max(0, int(60 - elapsed))
    st.error(f"⏱️ 剩余时间: {remaining} 秒")
    if remaining <= 0: st.session_state.page = "result"; st.rerun()

    if st.session_state.quiz_step == 1:
        ans = st.selectbox("Dijkstra 一定能找到最短路径？", ["请选择", "是", "否"])
        if st.button("下一题") and ans != "请选择":
            if ans == "是": st.session_state.quiz_score += int(20 + remaining/2)
            st.session_state.quiz_step = 2; st.rerun()
    else:
        ans = st.text_input("A* 公式？")
        if st.button("提交结果"):
            if "f=g+h" in ans.lower().replace(" ",""): st.session_state.quiz_score += int(20 + remaining/2)
            st.session_state.page = "result"; st.rerun()

# --- 6. 结果与排行榜 ---
elif st.session_state.page == "result":
    st.title("📊 答题报告")
    st.metric("本次得分", st.session_state.quiz_score)
    st.session_state.score += st.session_state.quiz_score
    # 答题结束同步总分到云端
    df = get_data()
    df.loc[df["学生"] == st.session_state.user, "总积分"] = st.session_state.score
    save_data(df)
    if st.button("返回大厅"): st.session_state.page = "dashboard"; st.rerun()

elif st.session_state.page == "leaderboard":
    st.title("🏆 班级荣誉榜")
    df = get_data().sort_values(by="总积分", ascending=False).reset_index(drop=True)
    for i, row in df.iterrows():
        style = f"rank-{i+1}" if i < 3 else ""
        st.markdown(f'<div style="display:flex; justify-content:space-between; padding:10px;">'
                    f'<span class="{style}">第 {i+1} 名: {row["学生"]}</span>'
                    f'<span>{row["总积分"]} pts</span></div>', unsafe_allow_html=True)
    if st.button("返回"): st.session_state.page = "dashboard"; st.rerun()
        # D:\conda\Scripts\streamlit.exe run .\app.py