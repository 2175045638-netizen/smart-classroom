import streamlit as st
import pandas as pd
import time
from streamlit_gsheets import GSheetsConnection
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --- 题目数据库定义 ---
QUIZ_BANK = {
    "迪杰斯特拉算法": [
        {"type": "choice", "q": "Dijkstra 算法的核心思想是什么？", "options": ["贪心", "动态规划", "回溯"], "a": "贪心", "pts": 30},
        {"type": "choice", "q": "Dijkstra 能处理含有负权边的图吗？", "options": ["能", "不能"], "a": "不能", "pts": 30},
        {"type": "input", "q": "若起点到A距离为5，A到B边权为3，则更新后起点到B距离为？", "a": "8", "pts": 40}
    ],
    "A*算法": [
        {"type": "choice", "q": "A* 算法中的 h(n) 代表什么？", "options": ["实际代价", "启发式预估代价", "总代价"], "a": "启发式预估代价", "pts": 30},
        {"type": "input", "q": "A* 算法的公式是 f = g + ?", "a": "h", "pts": 30},
        {"type": "choice", "q": "如果 h(n) 始终为 0，A* 退化为什么算法？", "options": ["BFS", "Dijkstra", "DFS"], "a": "Dijkstra", "pts": 40}
    ]
}

# --- 0. 数据库连接与初始化 ---
# 在 Streamlit Cloud 的 Secrets 中配置表格链接
conn_data = st.connection("gsheets_data", type=GSheetsConnection)

# 连接2：答题状态控制表
conn_control = st.connection("gsheets_control", type=GSheetsConnection)

def get_student_data():
    # 默认读取该文件的第一个工作表
    return conn_data.read(ttl=10)

def save_student_data(df):
    conn_data.update(data=df)
    st.cache_data.clear()

# 操作【答题状态控制】表的函数
def get_system_state():
    # 假设你的状态数据在名为 "Sheet1" 的工作表里
    return conn_control.read(ttl=10)

def update_system_state(df):
    conn_control.update(data=df)
    # 无需清除整个 cache，因为这个表变动频繁

def update_system_state(df):
    try:
        conn_control.update(data=df)
    except Exception as e:
        if "429" in str(e):
            st.error("操作太快啦！Google 正在排队，请 5 秒后重试。")
            time.sleep(5)

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

    all_steps.append({
        "t": "算法简介", 
        "c": ("迪杰斯特拉算法（Dijkstra's Algorithm）是由荷兰计算机科学家艾兹赫尔·戴克斯特拉在 1956 年提出的一种单源最短路径算法。\n\n"
              "该算法该算法既适用于无向加权图，也适用于有向加权图。它的核心思想是贪心策略，即每次都选择当前已知距离源点最近的一个节点，并以此为基准更新其邻居的距离。\n\n"
              "接下来，我们将以下面的无向加权图为例，通过分步演示来学习这一算法。"), 
        "img": "assets/dijkstra_demo1.png" # 这里放你原本的简介图片路径
    })
    
    # 初始状态快照
    all_steps.append({
        "t": "准备阶段",
        "c": "算法开始，起点 0 距离设为 0，其余设为无穷大。",
        "type": "interactive_demo",
        "snapshot": {"dist_form": dist_formula.copy(), "prev": prev.copy(), "visited": visited.copy(), "curr": None}
    })

    while unvisited:
        curr = min(unvisited, key=lambda n: dist[n])
        if dist[curr] == float('inf'): break
        
        step_explanation = f"从所有未访问节点中，选择距离最小的节点 **{curr}**（当前距离为 {dist[curr]}）。"
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
                    update_logs.append(f"节点 {nbr}: 更新表格，因为发现更短路径： {old_dist_str} > {dist_formula[nbr]}")
                else:
                    update_logs.append(f"节点 {nbr}: 维持现状，因为现有距离 {dist[nbr]} <= 尝试路径 ({dist[curr]} + {weight})")

        visited[curr] = True
        unvisited.remove(curr)

        all_steps.append({
            "t": f"分步学习--处理节点 {curr}",
            "explanation": f"正在从节点 {curr} 向外探索邻居。",
            "c": step_explanation + "\n\n" + ("\n".join([f"- {log}" for log in update_logs])),
            "type": "interactive_demo",
            "snapshot": {"dist_form": dist_formula.copy(), "prev": prev.copy(), "visited": visited.copy(), "curr": curr}
        })

    all_steps.append({
        "t": "注意事项", 
        "c": ("Dijkstra算法虽然复杂度非常优秀（单源最短路中基本上最优），但是它不能用来计算带有负权边的图，即必须保证图中所有边的权值为非负数。\n\n"
              "请大家思考一下为什么。\n\n"
              "接下来，请完成知识检验考察大家的学习成果吧。"), 
    })    
        
    return all_steps

def render_dijkstra_snapshot(snapshot):
    # --- 1. 定义图结构与坐标 (确保与你图片中的位置一致) ---
    edges = [
        (0, 1, 4), (0, 7, 8), (1, 7, 11), (1, 2, 8), (7, 8, 7), (7, 6, 1),
        (2, 8, 2), (8, 6, 6), (2, 3, 7), (2, 5, 4), (6, 5, 2), (3, 5, 14),
        (3, 4, 9), (5, 4, 10)
    ]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    
    # 手动固定节点坐标，还原图片布局
    pos = {
        0: (0, 1), 1: (1, 2), 7: (1, 0), 2: (2, 2), 8: (2, 1), 
        6: (2, 0), 3: (3, 2), 5: (3, 0), 4: (4, 1)
    }

    # --- 2. 创建 Streamlit 分栏 ---
    col1, col2 = st.columns([1.2, 1])

    # --- 3. 左侧：绘制 NetworkX 图 ---
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 节点颜色逻辑：当前考察点红色，已确定点绿色，其余灰色
        node_colors = []
        for n in G.nodes():
            if n == snapshot["curr"]:
                node_colors.append('#FF4B4B') # 红色
            elif snapshot["visited"][n]:
                node_colors.append('#2E7D32') # 绿色
            else:
                node_colors.append('#BDBDBD') # 灰色

        # 绘图
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=1000, font_color='white', font_weight='bold', ax=ax)
        
        # 绘制边权重
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
        
        plt.axis('off') # 隐藏坐标轴
        st.pyplot(fig)
        plt.close()

    # --- 4. 右侧：绘制实时状态表 (包含详细计算过程) ---
    with col2:
        st.write("**实时路径状态表**")
        df = pd.DataFrame({
            "节点": [f"点 {i}" for i in range(9)],
            "确定 (√)": ["√" if snapshot["visited"][i] else "" for i in range(9)],
            "计算过程 / 距离": [snapshot["dist_form"][i] for i in range(9)],
            "前驱点": [snapshot["prev"][i] for i in range(9)]
        })
        st.table(df)

def generate_grid_map():
    """生成一个10x10的网格地图，0为平地，1为障碍"""
    grid = np.zeros((10, 10))
    # 设置障碍物 (模仿 U 型障碍)
    grid[3:7, 3] = 1
    grid[3, 3:7] = 1
    grid[7, 3:7] = 1
    return grid

# --- 新增：A* 分步逻辑生成 ---
def generate_Astar_full_steps():
    grid = generate_grid_map()
    start = (2, 2)
    goal = (8, 8)
    
    def heuristic(a, b):
        # 使用曼哈顿距离
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_list = {start: 0 + heuristic(start, goal)}
    g_score = {start: 0}
    parent = {}
    closed_list = set()

    all_steps = []

    all_steps.append({
        "t": "算法简介", 
        "c": ("A-star（A*）算法是一种经典的启发式搜索算法，用来在图或状态空间中找到从起点到终点的代价最小路径。它结合了Dijkstra算法和贪心算法的优点，通过启发式函数在保证最优解的同时提高搜索效率。\n\n"
              "A*算法的目标是找到从起点到终点的最短路径。其通过维护一个优先队列（最小堆），根据估价函数$f(n)$来选择下一步要探索的节点，其中估价函数由两部分组成：\n\n"
              "实际代价$g(n)$：从起点到当前节点$n$的已知路径代价（已经走了多少步）。\n\n"
              "启发式代价$h(n)$：从当前节点$n$到终点的估计代价（预测还要多少步到达终点）。\n\n"
              "因此，总估价函数表达为： $f(n) = g(n) + h(n)$\n\n"
              "A*算法每次选择$f(n)$最小的节点进行扩展，直到找到终点。\n\n"), 
    })
    all_steps.append({
        "t":"启发式代价$h(n)$",
        "c":("启发式函数的选择决定了 A* 算法的效率，但它必须满足**可接受性（Admissibility）**：\n"
        "即对于图中任何节点 n，其预估代价 $h(n)$ 必须不大于实际最短路径代价 $h^*(n)$，即：$h(n) \le h^*(n)$。\n\n"
        "如果 $h(n)$ 是可接受的，A* 算法保证能找到最优解。如果 $h(n)$ 大于实际代价，算法可能运行更快，但无法保证最短路径。\n\n"
        "常见的启发函数选择：\n\n"
        "1. **曼哈顿距离 (Manhattan Distance)**：适用于只能在网格中水平或垂直移动的场景。\n"
        "公式：$h(n) = |x_n - x_{goal}| + |y_n - y_{goal}|$\n\n"
        "2. **欧几里得距离 (Euclidean Distance)**：适用于可以沿任意角度直线移动的场景。\n"
        "公式：$h(n) = \sqrt{(x_n - x_{goal})^2 + (y_n - y_{goal})^2}$\n\n"),
    })

    all_steps.append({
        "t": "A* 算法准备阶段",
        "c": f"起点设为 {start}，终点为 {goal}。我们将使用曼哈顿距离作为 $h(n)$。",
        "type": "astar_visual",
        "snapshot": {
            "grid": grid.tolist(),
            "curr": None,
            "open": list(open_list.keys()),
            "closed": list(closed_list),
            "g_score": g_score.copy(),
            "goal": goal
        }
    })

    while open_list:
        # 获取 f 值最小的节点
        curr = min(open_list, key=open_list.get)
        
        if curr == goal:
            break
            
        del open_list[curr]
        closed_list.add(curr)
        
        update_logs = []
        # 探索 4 个方向
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (curr[0] + dx, curr[1] + dy)
            
            if 0 <= neighbor[0] < 10 and 0 <= neighbor[1] < 10:
                if grid[neighbor[0], neighbor[1]] == 1 or neighbor in closed_list:
                    continue
                
                tentative_g = g_score[curr] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_val = tentative_g + heuristic(neighbor, goal)
                    open_list[neighbor] = f_val
                    parent[neighbor] = curr
                    update_logs.append(f"发现节点 {neighbor}: $g={tentative_g}, h={heuristic(neighbor, goal)}, f={f_val}$")

        # 记录当前步骤快照
        all_steps.append({
            "t": f"正在探索节点 {curr}",
            "explanation": f"从 Open List 中选择了 $f(n)$ 最小的节点 {curr}。",
            "c": "\n".join([f"· {log}" for log in update_logs]) if update_logs else "当前节点邻居已全部探索或不可达。",
            "type": "astar_visual",
            "snapshot": {
                "grid": grid.tolist(),
                "curr": curr,
                "open": list(open_list.keys()),
                "closed": list(closed_list),
                "g_score": g_score.copy(),
                "goal": goal
            }
        })

    all_steps.append({
        "t": "与迪杰斯特拉算法对比", 
        "c": ("相比于 Dijkstra 算法，A* 算法由于其启发式搜索，通常能更快地找到路径，尤其是在大型图中。\n\n"
              "但是，它要存储开放列表和关闭列表中的所有节点，当图非常大时，可能会占用大量内存。此外，它的性能高度依赖于启发式函数的质量。一个糟糕的启发式函数可能导致算法性能下降，甚至退化为 Dijkstra 算法。\n\n"
              ), 
    })

    return all_steps

def render_astar_snapshot(snapshot):
    grid = np.array(snapshot["grid"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid, cmap='Greys', origin='upper')
    
    goal = snapshot["goal"]
    curr_g = snapshot["g_score"]

    for r in range(10):
        for c in range(10):
            pos = (r, c)
            # 只有在 Open 或 Closed 列表中的点才显示数值，避免画面太乱
            if pos in snapshot["open"] or pos in snapshot["closed"]:
                g = curr_g.get(pos, 0)
                h = abs(r - goal[0]) + abs(c - goal[1]) # 曼哈顿距离
                f = g + h
                # 在方块中心标注 f 值
                ax.text(c, r, f'f:{f}={g}+{h}', ha='center', va='center', 
                        color='blue', fontsize=6, fontweight='bold')
            
            # 绘制节点颜色
            if pos in snapshot["closed"]:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#2E7D32', alpha=0.3))
            elif pos in snapshot["open"]:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#FFD600', alpha=0.4))

    # 绘制当前考察点
    if snapshot["curr"]:
        ax.plot(snapshot["curr"][1], snapshot["curr"][0], 'X', color='#FF4B4B', markersize=12)

    ax.set_title("A* Grid Search (Yellow: Open, Green: Visited)", fontsize=10)
    st.pyplot(fig)
    plt.close()

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
        all_data = get_student_data()
        edited_df = st.data_editor(all_data, num_rows="dynamic")
        if st.button("💾 保存修改到云端"):
            save_student_data(edited_df)
            st.success("云端数据同步成功！")

        st.subheader("📢 课堂答题同步控制")
    
        # 读取当前的全局状态表
        # 注意：这里需要指定对应的 worksheet 名称
        state_df = conn_control.read(ttl=60)

        # 选择主题
        selected_topic = st.selectbox("选择本次答题主题", list(QUIZ_BANK.keys()))
    
        col_admin1, col_admin2, col_admin3 = st.columns(3)
        with col_admin1:
            if st.button("🚩 发布主题"):
                state_df.loc[state_df['Key'] == 'quiz_status', 'Value'] = 'ready'
                state_df.loc[state_df['Key'] == 'current_topic', 'Value'] = selected_topic
                update_system_state(state_df)
                st.success(f"已发布: {selected_topic}")
            
        with col_admin2:
            if st.button("🚀 开始答题"):
                state_df.loc[state_df['Key'] == 'quiz_status', 'Value'] = 'started'
                state_df.loc[state_df['Key'] == 'start_time', 'Value'] = str(time.time())
                update_system_state(state_df)
                st.toast("全员计时开始！")

        with col_admin3:
            if st.button("🛑 结束答题", use_container_width=True):
                # 将状态设为 idle (闲置)
                state_df.loc[state_df['Key'] == 'quiz_status', 'Value'] = 'idle'
                # 清空当前主题
                state_df.loc[state_df['Key'] == 'current_topic', 'Value'] = 'None'
                update_system_state(state_df)
                st.toast("答题通道已关闭")
                st.rerun()

# --- 1. 登录页面 ---
if st.session_state.page == "login":
    st.title("🌟 智能课堂互动系统")
    name = st.text_input("请输入姓名以登录")
    if st.button("进入教室"):
        if name:
            st.session_state.user = name
            # 登录时从云端同步该学生的旧积分
            df = get_student_data()
            if name in df["学生"].values:
                user_row = df[df["学生"] == name].iloc[0]
                st.session_state.score = int(user_row["总积分"])
                learned = set()
                if user_row.get("Dijkstra_已完成") == True: learned.add("Dijkstra")
                if user_row.get("AStar_已完成") == True: learned.add("AStar")
                st.session_state.learned_modules = learned
            else:
                # 新学生自动注册
                new_user = pd.DataFrame([{"学生": name, 
                    "总积分": 0, 
                    "Dijkstra_已完成": False, 
                    "AStar_已完成": False}])
                save_student_data(pd.concat([df, new_user], ignore_index=True))
            st.session_state.page = "dashboard"
            st.rerun()

# --- 2. 仪表盘 ---
elif st.session_state.page == "dashboard":
    sys_state = get_system_state()
    # 容错处理：确保能读取到状态
    try:
        current_status = sys_state.loc[sys_state['Key'] == 'quiz_status', 'Value'].values[0]
    except:
        current_status = 'idle'
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
            is_done = "AStar" in st.session_state.learned_modules
            label = "【✅ 已掌握】" if is_done else ""
            st.markdown('<div class="algo-card"><h3>Dijkstra 算法</h3></div>', unsafe_allow_html=True)
            if st.button("进入学习", key="dij"):
                st.session_state.current_algo = "Dijkstra"; st.session_state.page = "learning"; st.session_state.step = 0; st.rerun()
        with c2:
            is_done = "Dijkstra" in st.session_state.learned_modules
            label = "【✅ 已掌握】" if is_done else ""
            st.markdown('<div class="algo-card"><h3>A* 算法</h3></div>', unsafe_allow_html=True)
            if st.button("进入学习", key="astar"):
                st.session_state.current_algo = "AStar"; st.session_state.page = "learning"; st.session_state.step = 0; st.rerun()

    st.divider()
    if current_status in ["ready", "started"]:
        # 只有在老师发布了题目（ready）或者正在答题（started）时才显示按钮
        st.warning("🔔 限时随堂测试已发布")
        if st.button("🚀 开始进入答题模式", use_container_width=True):
            # 初始化答题状态
            st.session_state.page = "quiz"
            st.session_state.quiz_step = 0
            st.session_state.quiz_score = 0
            st.rerun()
    else:
        # 当状态为 idle 或 ended 时
        st.info("💡 限时随堂测试暂未发布")
        # 这里不放置 st.button，按钮就会自然消失

# --- 3. 教学模式 ---
elif st.session_state.page == "learning":
    algo = st.session_state.current_algo
    
    # 1. 确保数据源已初始化
    if algo == "AStar":
        if "astar_full_steps" not in st.session_state:
            st.session_state.astar_full_steps = generate_Astar_full_steps()
        current_steps_source = st.session_state.astar_full_steps
    else:
        if "dijkstra_full_steps" not in st.session_state:
            st.session_state.dijkstra_full_steps = generate_dijkstra_steps()
        current_steps_source = st.session_state.dijkstra_full_steps

    # 2. 越界保护：确保 step 不超过数据长度
    if st.session_state.step >= len(current_steps_source):
        st.session_state.step = 0
    
    data = current_steps_source[st.session_state.step]

    # --- 标题栏 ---
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.subheader(f"📖 正在学习: {algo} 算法")
    with head_col2:
        if st.button("🏠 返回首页", key="back_home_btn"):
            st.session_state.page = "dashboard"
            st.session_state.step = 0
            st.rerun()
    st.divider()

    # --- 内容讲解区 ---
    st.header(data['t'])
    if 'explanation' in data:
        st.info(data['explanation'])
    
    # --- 交互演示区 (分算法渲染) ---
    if data.get("type") == "interactive_demo":
        # Dijkstra 渲染：调用你定义的 render_dijkstra_snapshot
        render_dijkstra_snapshot(data['snapshot'])
        st.write(data['c'])
        
    elif data.get("type") == "astar_visual":
        # A* 增强渲染：左图右表
        col_viz, col_data = st.columns([1.5, 1])
        with col_viz:
            render_astar_snapshot(data['snapshot'])
        with col_data:
            st.markdown("🔍 **节点代价分析**")
            curr_node = data['snapshot']['curr']
            if curr_node:
                g = data['snapshot']['g_score'].get(curr_node, 0)
                goal = data['snapshot']['goal']
                h = abs(curr_node[0] - goal[0]) + abs(curr_node[1] - goal[1])
                st.metric("当前处理", f"({curr_node[0]}, {curr_node[1]})")
                st.write(f"- $g(n)$ (已走): `{g}`")
                st.write(f"- $h(n)$ (预估): `{h}`")
                st.write(f"- $f(n)$ (总计): **{g+h}**")
            else:
                st.write("等待算法开始...")
            st.divider()
            st.write("**算法日志:**")
            st.write(data['c'])
            
    else:
        st.write(data['c'])

    st.divider()

    # --- 底部导航控制 ---
    col_prev, col_mid, col_next = st.columns([1, 1, 1])
    with col_prev:
        if st.session_state.step > 0:
            if st.button("⬅️ 上一步", use_container_width=True, key="prev_btn"):
                st.session_state.step -= 1
                st.rerun()
    
    with col_mid:
        st.write(f"<p style='text-align:center; color:gray; padding-top:10px;'>步数: {st.session_state.step + 1} / {len(current_steps_source)}</p>", unsafe_allow_html=True)

    with col_next:
        if st.session_state.step < len(current_steps_source) - 1:
            if st.button("下一步 ➡️", use_container_width=True, key="next_btn"):
                st.session_state.step += 1
                st.rerun()
        else:
            # 学习完成阶段
            is_learned = algo in st.session_state.learned_modules
            btn_label = "✅ 测验通过 (查看)" if is_learned else "🚀 开始知识检验"
            if st.button(btn_label, use_container_width=True, type="primary", key="go_test_btn"):
                st.session_state.page = "learning_test"
                st.rerun()

# --- 4. 知识检验 ---
elif st.session_state.page == "learning_test":
    algo = st.session_state.current_algo
    is_completed = algo in st.session_state.learned_modules
    
    st.header(f"{'查看题目' if is_completed else '知识检验'}: {algo}")
    if is_completed:
        st.success("提示：你已通过此项测验，当前为查看模式（已显示正确答案）。")

    user_ans = ""
    correct_ans = [] # 统一初始化为列表，方便后续 any() 遍历
    is_text_input = False 
    
    with st.container():
        if algo == "Dijkstra":
            st.write("如图，这是一个有向加权图，权重代表两点之间的距离。请使用 Dijkstra 算法，计算出从A点到F点的最短路径。")
            
            # 修正点1：统一正确答案变量
            # 这里的正确答案既用于填入输入框，也用于后续判定
            ans_str = "A->B->D->F"
            correct_ans = [ans_str] 
            
            q = st.text_input(
                "请输入路径 (示例: D->F->E):", # 添加 Label
                value=ans_str if is_completed else "", 
                disabled=is_completed
            )
            user_ans = q
            
            # 图片居中显示
            st.write("") 
            img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
            with img_col2:
                st.image("assets/d_test1.png", caption="题目示意图", use_container_width=True)
            
            is_text_input = True
            
        elif algo == "AStar":
            options = [
                "从起点到当前节点的实际代价", 
                "从当前节点到终点的预估代价", 
                "算法运行的总步数"
            ]
            # 修正点2：查看模式下自动选中正确项
            correct_str = "从当前节点到终点的预估代价"
            correct_ans = [correct_str]
            
            default_index = options.index(correct_str) if is_completed else 0
            
            q = st.radio(
                "A* 算法的代价函数 f(n) = g(n) + h(n) 中，h(n) 代表什么？",
                options,
                index=default_index,
                disabled=is_completed
            )
            user_ans = q
            is_text_input = False

    st.divider()

    # ... 前接 user_ans 和 correct_ans 的定义 ...

    # 提交逻辑
    if is_completed:
        if st.button("返回主页", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    else:
        if st.button("确认提交", use_container_width=True):
            # 这里的比较逻辑要严谨（去除空格和转大小写）
            is_correct = any(ans.strip().lower() == user_ans.strip().lower() for ans in correct_ans)
            
            if is_correct:
                st.success("🎉 正确！积分 +50")
                st.session_state.learned_modules.add(algo)
                st.session_state.score += 50  # 假设给 50 分
                # 同步到云端
                df = get_student_data()
                idx = df[df["学生"] == st.session_state.user].index
                if not idx.empty:
                    df.loc[idx, "总积分"] = st.session_state.score
                    # 动态更新对应的算法列
                    column_name = f"{algo}_已完成"
                    if column_name in df.columns:
                        df.loc[idx, column_name] = True
                    save_student_data(df)
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.last_result = "wrong"
                st.rerun() # 必须 rerun 才能看到错误提示

elif st.session_state.page == "quiz":
    # 1. 获取云端最新状态
    sys_state = get_system_state()
    status = sys_state.loc[sys_state['Key'] == 'quiz_status', 'Value'].values[0]
    topic = sys_state.loc[sys_state['Key'] == 'current_topic', 'Value'].values[0]
    
    # 2. 获取当前主题的题目列表
    questions = QUIZ_BANK.get(topic, [])
    total_q = len(questions)

    st.title(f"✍️ 课堂测试：{topic}")

    if status == "ready":
        st.info("🎯 答题主题已就绪，请等待老师点击『开始答题』...")
        if st.button("刷新状态"): st.rerun()

    elif status == "started":
        # 计算统一时间
        global_start = float(sys_state.loc[sys_state['Key'] == 'start_time', 'Value'].values[0])
        elapsed = time.time() - global_start
        remaining = max(0, int(120 - elapsed)) # 假设总时长120秒
        
        if remaining <= 0:
            st.warning("⏳ 时间到！正在自动结算...")
            st.session_state.page = "result"; st.rerun()

        st.error(f"⏱️ 全班统一倒计时：{remaining} 秒")
        
        # 3. 动态渲染当前题目
        current_q_idx = st.session_state.get('quiz_step', 0)
        
        if current_q_idx < total_q:
            q_data = questions[current_q_idx]
            st.markdown(f"### 第 {current_q_idx + 1} 题 / 共 {total_q} 题")
            st.write(q_data['q'])

            # 根据题目类型显示不同组件
            if q_data['type'] == "choice":
                ans = st.radio("选择答案", q_data['options'], key=f"q_{current_q_idx}")
            else:
                ans = st.text_input("填写答案", key=f"q_{current_q_idx}")

            if st.button("确认提交本题"):
                # 判定对错
                if str(ans).strip().lower() == str(q_data['a']).strip().lower():
                    st.session_state.quiz_score += q_data['pts']
                
                # 下一步
                if current_q_idx + 1 < total_q:
                    st.session_state.quiz_step = current_q_idx + 1
                else:
                    # 全部答完，记录完成时间
                    st.session_state.finish_time = elapsed
                    st.session_state.page = "result"
                st.rerun()
        else:
            st.session_state.page = "result"; st.rerun()

# --- 6. 结果与排行榜 ---
elif st.session_state.page == "result":
    st.title("📊 答题报告")
    st.metric("本次得分", st.session_state.quiz_score)
    st.session_state.score += st.session_state.quiz_score
    # 答题结束同步总分到云端
    df = get_student_data()
    df.loc[df["学生"] == st.session_state.user, "总积分"] = st.session_state.score
    save_student_data(df)
    if st.button("返回大厅"): st.session_state.page = "dashboard"; st.rerun()

elif st.session_state.page == "leaderboard":
    st.title("🏆 班级荣誉榜")
    df = get_student_data().sort_values(by="总积分", ascending=False).reset_index(drop=True)
    for i, row in df.iterrows():
        style = f"rank-{i+1}" if i < 3 else ""
        st.markdown(f'<div style="display:flex; justify-content:space-between; padding:10px;">'
                    f'<span class="{style}">第 {i+1} 名: {row["学生"]}</span>'
                    f'<span>{row["总积分"]} pts</span></div>', unsafe_allow_html=True)
    if st.button("返回"): st.session_state.page = "dashboard"; st.rerun()
       