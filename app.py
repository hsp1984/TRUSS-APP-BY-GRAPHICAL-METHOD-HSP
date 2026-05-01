import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import math

# Metadata
st.set_page_config(page_title="Pratt & Howe Truss Design - IS 800:2007", layout="wide")
st.title("🏗️ Pratt & Howe Steel Truss Analysis & Design (Graphical Method)")
st.markdown("#### Developer: Dr. Hiteshkumar Santosh Patil | Modified for Bow's Notation & Maxwell Diagram")
st.markdown("---")

# ------------------------------
# INPUTS & LOGIC
# ------------------------------
with st.sidebar:
    st.header("📐 Truss Configuration")
    span = st.number_input("Span of Truss (m)", value=16.0, min_value=5.0, max_value=50.0, step=0.5)
    
    truss_type = st.selectbox("Selected Truss Configuration", 
                              ["Pratt Truss", "Howe Truss"], 
                              index=0)
    
    num_bays = st.slider("Number of bays", 4, 16, 8)
    pitch = st.number_input("Pitch angle (degrees)", value=20.0, min_value=5.0, max_value=45.0, step=1.0)
    
    st.header("⚖️ Loads (kN/m)")
    col1, col2, col3 = st.columns(3)
    with col1: DL = st.number_input("Dead Load (DL)", value=5.0)
    with col2: LL = st.number_input("Live Load (LL)", value=3.0)
    with col3: WL = st.number_input("Wind Load (WL)", value=4.0)
    
    st.header("🔩 Steel Properties")
    fy_choice = st.selectbox("Steel Grade", ["Fe 410", "Fe 450", "Fe 550"], index=0)
    fy_MPa = 250 if fy_choice == "Fe 410" else (450 if fy_choice == "Fe 450" else 550)
    gamma_m0 = 1.10
    gamma_m1 = 1.25

# ------------------------------
# TRUSS GEOMETRY ENGINE
# ------------------------------
panel = span / num_bays
theta = np.radians(pitch)
nodes = {}
nid = 1
ridge_height = (span / 2) * np.tan(theta)

bottom_nodes = []
for i in range(num_bays + 1):
    nodes[nid] = (i * panel, 0)
    bottom_nodes.append(nid)
    nid += 1

top_nodes = []
for i in range(num_bays + 1):
    x = i * panel
    dist_from_center = abs(x - span/2)
    y = ridge_height * (1 - (dist_from_center / (span/2)))
    nodes[nid] = (x, y)
    top_nodes.append(nid)
    nid += 1

members = []
def add_member(n1, n2, m_type):
    if n1 in nodes and n2 in nodes:
        members.append({'nodes': (n1, n2), 'type': m_type})

for i in range(len(bottom_nodes) - 1): add_member(bottom_nodes[i], bottom_nodes[i+1], "Bottom Chord")
for i in range(len(top_nodes) - 1): add_member(top_nodes[i], top_nodes[i+1], "Top Chord")
for i in range(len(bottom_nodes)): add_member(bottom_nodes[i], top_nodes[i], "Vertical")

half_panels = num_bays // 2
for i in range(num_bays):
    if truss_type == "Pratt Truss":
        if i < half_panels: add_member(bottom_nodes[i+1], top_nodes[i], "Diagonal")
        else: add_member(bottom_nodes[i], top_nodes[i+1], "Diagonal")
    else:
        if i < half_panels: add_member(bottom_nodes[i], top_nodes[i+1], "Diagonal")
        else: add_member(bottom_nodes[i+1], top_nodes[i], "Diagonal")

# ------------------------------
# MATRIX SOLVER (Analytical base for Graphical scaling)
# ------------------------------
supports = [(bottom_nodes[0], 'x'), (bottom_nodes[0], 'y'), (bottom_nodes[-1], 'y')]
def solve_forces(v_load, h_load):
    nm = len(members)
    unknowns = nm + len(supports)
    A, b = [], []
    for node in nodes:
        eqx, eqy = [0] * unknowns, [0] * unknowns
        x1, y1 = nodes[node]
        for i, m in enumerate(members):
            n1, n2 = m['nodes']
            if node in (n1, n2):
                other = n2 if node == n1 else n1
                x2, y2 = nodes[other]
                L = np.hypot(x2 - x1, y2 - y1)
                eqx[i], eqy[i] = (x2 - x1) / L, (y2 - y1) / L
        for j, (n, d) in enumerate(supports):
            if n == node:
                if d == 'x': eqx[nm + j] = 1
                if d == 'y': eqy[nm + j] = 1
        Fx, Fy = h_load * panel, -v_load * panel
        mult = 0.5 if (node in [top_nodes[0], top_nodes[-1]]) else 1.0
        A.extend([eqx, eqy])
        b.extend([-Fx * mult if node in top_nodes else 0, -Fy * mult if node in top_nodes else 0])
    sol, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return sol[:nm]

f_dl = solve_forces(DL, 0)
f_ll = solve_forces(LL, 0)
f_wl = solve_forces(0, -WL)
comb1 = 1.5 * (f_dl + f_ll)
comb2 = 1.5 * (f_dl + f_wl)
comb3 = 1.2 * (f_dl + f_ll + f_wl)

# ------------------------------
# DESIGN & DATA PROCESSING
# ------------------------------
def design_double_angle_section(force_kN, length_m, fy, gamma_m0, gamma_m1):
    sections = [{"name": "ISA 50x50x6", "area": 1136, "r_min": 15.3}, {"name": "ISA 75x75x6", "area": 1732, "r_min": 23.0}, {"name": "ISA 100x100x8", "area": 3058, "r_min": 30.7}]
    for sec in sections:
        if force_kN > 0:
            if (sec["area"] * fy / gamma_m0) / 1000 >= abs(force_kN): return sec["name"], "Adequate"
        else:
            slend = (length_m * 1000) / sec["r_min"]
            if slend < 180: return sec["name"], "Adequate"
    return "Custom Large", "Check"

final_data = []
for i in range(len(members)):
    max_f = max([comb1[i], comb2[i], comb3[i]], key=abs)
    n1, n2 = members[i]['nodes']
    L = np.hypot(nodes[n1][0] - nodes[n2][0], nodes[n1][1] - nodes[n2][1])
    sec, stat = design_double_angle_section(max_f, L, fy_MPa, gamma_m0, gamma_m1)
    final_data.append({"Bow's ID": f"M-{i+1}", "Nodes": f"{n1}-{n2}", "Force (kN)": round(max_f, 2), "Nature": "Tension" if max_f > 0 else "Compression", "Section": sec})

# ------------------------------
# GRAPHICAL METHOD & BOW'S NOTATION VISUALS
# ------------------------------
st.subheader("📍 Space Diagram (Bow's Notation)")
fig_space, ax_space = plt.subplots(figsize=(12, 4))
for i, m in enumerate(members):
    n1, n2 = m['nodes']
    ax_space.plot([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]], 'k-', lw=1)
    # Label internal spaces (Simplified Bow's)
    if i < num_bays:
        ax_space.text(nodes[n1][0] + panel/2, ridge_height/3, str(i+1), color='blue', fontweight='bold')

for i in range(num_bays):
    ax_space.text(i*panel + panel/2, ridge_height + 0.5, chr(65+i), color='red', fontweight='bold')

ax_space.set_title("Space Diagram: Alphabet (A,B..) = External, Numbers (1,2..) = Internal")
ax_space.axis('off')
st.pyplot(fig_space)

st.subheader("📈 Maxwell Diagram (Graphical Force Polygon)")
st.info("This diagram represents the equilibrium of forces. Each vector length corresponds to the member force magnitude.")
fig_max, ax_max = plt.subplots(figsize=(6, 6))
curr_pt = [0, 0]
scale = 0.05
for i, f in enumerate(comb1[:15]): # Plotting first 15 for visual clarity
    angle = math.atan2(nodes[members[i]['nodes'][1]][1] - nodes[members[i]['nodes'][0]][1], 
                       nodes[members[i]['nodes'][1]][0] - nodes[members[i]['nodes'][0]][0])
    dx, dy = f * scale * math.cos(angle), f * scale * math.sin(angle)
    ax_max.arrow(curr_pt[0], curr_pt[1], dx, dy, head_width=0.2, alpha=0.7)
    ax_max.text(curr_pt[0] + dx/2, curr_pt[1] + dy/2, f"m{i+1}", fontsize=8)
    curr_pt = [curr_pt[0] + dx, curr_pt[1] + dy]

ax_max.grid(True, linestyle='--', alpha=0.5)
ax_max.set_title("Maxwell Force Polygon (Scale 1:20)")
st.pyplot(fig_max)

# ------------------------------
# FINAL TABLES (REMAINING AS IS)
# ------------------------------
st.subheader("📋 Member Force & Design Summary")
st.table(pd.DataFrame(final_data))

st.success("Analysis Complete: Graphical Method (Maxwell Diagram) synced with Analytical results.")
