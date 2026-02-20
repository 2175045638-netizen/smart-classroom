import streamlit as st
import pandas as pd
import time
import datetime

# --- 初始化全局状态 ---
def init_state():
    if 'page' not in st.session_state:
        st.session_state.page = "login"  # login, dashboard, learning, quiz, result
    if 'user' not in st.session_state:
        st.session_state.user = ""
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'learned_modules' not in st.session_state:
        st.session_state.learned_modules = set() # 记录已学完的板块
    if 'quiz_active' not in st.session_state:
        st.session_state.quiz_active = False # 锁定模式
    if 'step' not in st.session_state:
        st.session_state.step = 0 # 教学步骤

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

# --- 模拟排行榜数据 ---
@st.cache_data
def get_mock_leaderboard():
    return pd.DataFrame({
        "学生": ["王小明", "李华", "张三", "赵敏"],
        "总积分": [120, 110, 95, 80]
    })

# --- 1. 登录页面 ---
if st.session_state.page == "login":
    st.title("🌟 智能课堂互动系统")
    with st.container():
        name = st.text_input("请输入姓名以登录")
        if st.button("进入教室"):
            if name:
                st.session_state.user = name
                st.session_state.page = "dashboard"
                st.rerun()

# --- 2. 仪表盘 (知识板块选择) ---
elif st.session_state.page == "dashboard":
    st.title(f"👋 你好, {st.session_state.user}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("我的总积分", st.session_state.score)
    with col2:
        if st.button("🏆 查看班级排行榜"):
            st.session_state.page = "leaderboard"
            st.rerun()

    st.subheader("📚 课程知识地图")
    
    # 路径规划板块
    with st.expander("📍 路径规划算法板块", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="algo-card"><h3>Dijkstra 算法</h3><p>从单点到所有点的最短路径</p></div>', unsafe_allow_html=True)
            if st.button("进入学习", key="dij"):
                st.session_state.current_algo = "Dijkstra"
                st.session_state.page = "learning"
                st.session_state.step = 0
                st.rerun()
        with c2:
            st.markdown('<div class="algo-card"><h3>A* 算法</h3><p>启发式搜索：更快、更智能</p></div>', unsafe_allow_html=True)
            if st.button("进入学习", key="astar"):
                st.session_state.current_algo = "AStar"
                st.session_state.page = "learning"
                st.session_state.step = 0
                st.rerun()

    # 老师发布的任务区
    st.divider()
    st.warning("🔔 老师发布了新任务：路径规划随堂测试 (限时 60s)")
    if st.button("🚀 开始进入答题模式 (进入后无法退出)"):
        st.session_state.page = "quiz"
        st.session_state.quiz_step = 1
        st.session_state.quiz_score = 0
        st.session_state.start_time = time.time()
        st.rerun()

# --- 3. 教学模式 (分步走) ---
elif st.session_state.page == "learning":
    algo = st.session_state.current_algo
    st.title(f"📖 正在学习: {algo}")
    
    steps = {
        "AStar": [
            {"t": "核心概念：贪心算法", "c": "贪心算法每一步都选择当前看起来最优的路径...", "img": "💡"},
            {"t": "启发式搜索 (Heuristic)", "c": "A*引入了 h(n)，即预测到终点的距离。公式：f=g+h", "img": "🔍"},
            {"t": "搜索迭代可视化", "c": "看！A* 优先探索朝向终点的方格，而不是像水波一样扩散。", "img": "🗺️"},
            {"t": "小结", "c": "A* 是带了GPS的 Dijkstra。", "img": "✅"}
        ],
        "Dijkstra": [
            {"t": "核心概念：广度优先", "c": "Dijkstra 确保找到最短路径，它不放过任何一个可能的节点。", "img": "🌊"},
            {"t": "迭代过程", "c": "不断更新起点到邻居节点的距离...", "img": "🔢"}
        ]
    }
    
    current_step_data = steps[algo][st.session_state.step]
    
    st.info(f"第 {st.session_state.step + 1} 步 / 共 {len(steps[algo])} 步")
    st.header(current_step_data['t'])
    st.write(current_step_data['c'])
    st.title(current_step_data['img']) # 模拟图像/动图
    
    cols = st.columns([1,1,1])
    with cols[0]:
        if st.session_state.step > 0:
            if st.button("上一步"):
                st.session_state.step -= 1
                st.rerun()
    with cols[2]:
        if st.session_state.step < len(steps[algo]) - 1:
            if st.button("下一步"):
                st.session_state.step += 1
                st.rerun()
        else:
            if algo not in st.session_state.learned_modules:
                if st.button("🏁 完成学习并进入知识检验"):
                    st.session_state.page = "learning_test"
                    st.rerun()
            else:
                st.success("本模块已学完，积分已领取。")
                if st.button("返回首页"):
                    st.session_state.page = "dashboard"
                    st.rerun()

# --- 4. 知识检验 (学完后的测试) ---
elif st.session_state.page == "learning_test":
    st.header("🎯 知识检验")
    q = st.radio("A* 算法中，f = g + h，h 代表什么？", ["起点距离", "预估终点距离", "随机值"])
    if st.button("提交结果"):
        if "预估" in q:
            st.session_state.score += 50
            st.session_state.learned_modules.add(st.session_state.current_algo)
            st.success("回答正确！获得 50 积分奖励！")
        else:
            st.error("回答错误，请重新回顾知识点。")
        time.sleep(2)
        st.session_state.page = "dashboard"
        st.rerun()

# --- 5. 课堂答题模式 (锁定模式) ---
elif st.session_state.page == "quiz":
    # 隐藏侧边栏逻辑 (在Streamlit中通过不渲染侧边栏内容实现)
    st.empty() 
    
    # 倒计时逻辑
    limit = 60 # 老师设置的60秒
    elapsed = time.time() - st.session_state.start_time
    remaining = max(0, int(limit - elapsed))
    
    st.error(f"⏱️ 剩余时间: {remaining} 秒")
    if remaining <= 0:
        st.session_state.page = "result"
        st.rerun()

    st.subheader(f"第 {st.session_state.quiz_step} 题 / 共 2 题")
    
    if st.session_state.quiz_step == 1:
        ans = st.selectbox("Dijkstra 算法是否一定能找到最短路径？", ["请选择", "是", "否"])
        if st.button("提交答案并下一题"):
            if ans == "是":
                # 根据时间给分，越快分越高
                st.session_state.quiz_score += int(20 + (remaining/2))
            st.session_state.quiz_step = 2
            st.rerun()
            
    elif st.session_state.quiz_step == 2:
        ans = st.text_input("请输入 A* 算法的核心公式 (例如 a=b+c)")
        if st.button("提交并结算"):
            if "f=g+h" in ans.lower().replace(" ", ""):
                st.session_state.quiz_score += int(20 + (remaining/2))
            st.session_state.page = "result"
            st.rerun()

# --- 6. 结果与排行榜 ---
elif st.session_state.page == "result":
    st.balloons()
    st.title("📊 答题报告")
    st.metric("本次得分", st.session_state.quiz_score)
    
    # 额外奖励逻辑
    bonus = 0
    if st.session_state.quiz_score > 40: # 模拟前三名逻辑
        bonus = 30
        st.success(f"🎊 表现优异！获得额外排名奖励 {bonus} 积分！")
    
    st.session_state.score += (st.session_state.quiz_score + bonus)
    
    if st.button("返回大厅"):
        st.session_state.page = "dashboard"
        st.rerun()

elif st.session_state.page == "leaderboard":
    st.title("🏆 班级荣誉榜")
    df = get_mock_leaderboard()
    # 加入当前用户
    new_row = pd.DataFrame({"学生": [st.session_state.user], "总积分": [st.session_state.score]})
    df = pd.concat([df, new_row]).sort_values(by="总积分", ascending=False).reset_index(drop=True)
    
    for i, row in df.iterrows():
        rank_style = f"rank-{i+1}" if i < 3 else ""
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee;">
            <span class="{rank_style}">第 {i+1} 名: {row['学生']}</span>
            <span style="font-weight: bold;">{row['总积分']} pts</span>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("返回"):
        st.session_state.page = "dashboard"
        st.rerun()
        # D:\conda\Scripts\streamlit.exe run .\app.py