import streamlit as st
import pandas as pd
import time
from streamlit_gsheets import GSheetsConnection
import networkx as nx
import matplotlib.pyplot as plt

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

def generate_dijkstra_steps():
    # 图结构定义 (与你图片一致)
    edges = [(0,1,4),(0,7,8),(1,7,11),(1,2,8),(7,8,7),(7,6,1),(2,8,2),(8,6,6),(2,3,7),(2,5,4),(6,5,2),(3,5,14),(3,4,9),(5,4,10)]
    
    # 初始化
    dist = {i: float('inf') for i in range(9)}; dist[0] = 0
    # 新增：用于存储计算痕迹的字典，初始化为 "∞" 或 "0"
    dist_formula = {i: "∞" for i in range(9)}; dist_formula[0] = "0"
    prev = {i: "-" for i in range(9)}
    visited = {i: False for i in range(9)}
    unvisited = list(range(9))
    
    all_steps = []
    
    # 初始状态快照
    all_steps.append({
        "t": "准备阶段",
        "c": "算法开始，起点 0 距离设为 0，其余设为无穷大。",
        "explanation": "此时尚未开始探索，Visit 集合为空。",
        "type": "interactive_demo",
        "snapshot": {"dist_form": dist_formula.copy(), "prev": prev.copy(), "visited": visited.copy(), "curr": None}
    })

    while unvisited:
        curr = min(unvisited, key=lambda n: dist[n])
        if dist[curr] == float('inf'): break
        
        step_explanation = f"**当前步骤**：从所有未访问节点中，选择距离最小的节点 **{curr}**（当前距离为 {dist[curr]}）。"
        update_logs = []

        # 遍历邻居进行松弛操作
        for nbr in range(9):
            # 获取边权重 (支持无向图)
            weight = next((e[2] for e in edges if (e[0]==curr and e[1]==nbr) or (e[0]==nbr and e[1]==curr)), None)
            
            if weight is not None and not visited[nbr]:
                new_val = dist[curr] + weight
                # 无论是否更新，我们都可以展示这个比较过程
                if new_val < dist[nbr]:
                    old_dist_str = str(dist[nbr]) if dist[nbr] != float('inf') else "∞"
                    # 关键修改：记录计算式
                    dist_formula[nbr] = f"{dist[curr]} + {weight} = {new_val}"
                    dist[nbr] = new_val
                    prev[nbr] = curr
                    update_logs.append(f"节点 {nbr}: 发现更短路径！ {old_dist_str} > {dist_formula[nbr]}")
                else:
                    update_logs.append(f"节点 {nbr}: 维持现状。现有距离 {dist[nbr]} <= 尝试路径 ({dist[curr]} + {weight})")

        visited[curr] = True
        unvisited.remove(curr)

        all_steps.append({
            "t": f"处理节点 {curr}",
            "c": f"正在从节点 {curr} 向外探索邻居。",
            "explanation": step_explanation + "\n\n" + ("\n".join([f"- {log}" for log in update_logs])),
            "type": "interactive_demo",
            "snapshot": {"dist_form": dist_formula.copy(), "prev": prev.copy(), "visited": visited.copy(), "curr": curr}
        })
        
    return all_steps

def render_dijkstra_snapshot(snapshot):
    import networkx as nx
    import matplotlib.pyplot as plt
    import pandas as pd

    # 1. 设置布局
    c1, c2 = st.columns([1.2, 1])

    # 2. 左侧：图表可视化 (利用 Matplotlib)
    with c1:
        # 这里复用之前的绘图代码...
        # 红色表示当前正在处理的节点，绿色表示已确定的最短路径点
        pass 

    # 3. 右侧：详细步骤表 (对应你要求的 4+8=12 样式)
    with c2:
        st.write("**实时路径状态表**")
        df = pd.DataFrame({
            "节点": [f"点 {i}" for i in range(9)],
            "确定 (√)": ["✅" if snapshot["visited"][i] else "" for i in range(9)],
            "计算过程 / 距离": [snapshot["dist_form"][i] for i in range(9)],
            "前驱点": [snapshot["prev"][i] for i in range(9)]
        })
        
        # 使用 st.table 展示，因为它更像静态表格，不会有滚动条干扰
        st.table(df)

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
    
    # 1. 预定义 AStar 步骤（完全保留你原来的内容）
    # ---------------------------------------------------------
    astar_steps = [
        {"t": "核心概念：贪心算法", "c": "贪心算法选择当前最优路径...", "img": "💡"},
        {"t": "启发式搜索", "c": "A* 引入了 h(n) 预估代价。", "img": "🔍"}
    ]

    # 2. 动态生成 Dijkstra 步骤（将其展开为多步演示）
    # ---------------------------------------------------------
    # 只有当 algo 是 Dijkstra 时，才生成这组长列表
    if algo == "Dijkstra":
        if "dijkstra_full_steps" not in st.session_state:
            # 这里调用我们之前讨论的 generate_dijkstra_steps() 函数
            # 它会返回一个包含 10 步左右的列表，每一步都有 snapshot
            st.session_state.dijkstra_full_steps = generate_dijkstra_steps() 
        dijkstra_steps = st.session_state.dijkstra_full_steps
    else:
        dijkstra_steps = []

    # 3. 汇总所有算法的 steps 字典
    # ---------------------------------------------------------
    steps = {
        "AStar": astar_steps,
        "Dijkstra": dijkstra_steps
    }

    # 4. 初始化 step
    if "step" not in st.session_state:
        st.session_state.step = 0
        
    # 获取当前步的数据
    data = steps[algo][st.session_state.step]

    # --- 渲染逻辑 (保持你原来的代码不变) ---
    st.subheader(f"📖 正在学习: {algo}")
    st.divider()

    st.header(data['t'])
    # 如果有详细讲解文字，显示出来
    if 'explanation' in data:
        st.info(data['explanation'])
    st.write(data['c'])

    # 内容展示区
    if data.get("type") == "interactive_demo":
        # 传入当前步的 snapshot 进行绘图
        render_dijkstra_snapshot(data['snapshot'])
    else:
        # 原有的图片/表情渲染（A* 会走这里）
        img_path = data.get('img', "💡")
        if "/" in img_path or img_path.endswith(('.png', '.jpg', '.jpeg')):
            _, center_col, _ = st.columns([1, 6, 1]) 
            with center_col:
                try: st.image(img_path, use_container_width=True)
                except: st.error(f"图片加载失败: {img_path}")
        else:
            st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{img_path}</h1>", unsafe_allow_html=True)

    st.divider()

    # --- 底部导航按钮 (完全控制 step) ---
    col_l, col_m, col_r = st.columns([1, 1, 1])
    with col_l:
        if st.session_state.step > 0:
            if st.button("⬅️ 上一步", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()
    
    with col_r:
        # 这里会自动根据 steps[algo] 的长度来判断是翻页还是去考试
        if st.session_state.step < len(steps[algo]) - 1:
            if st.button("下一步 ➡️", use_container_width=True):
                st.session_state.step += 1
                st.rerun()
        else:
            # 走到最后一步了
            if st.button("🏁 知识检验", use_container_width=True):
                st.session_state.page = "learning_test"
                st.rerun()
        # ... 这里的知识检验/返回首页逻辑保持不变 ...
                
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
       