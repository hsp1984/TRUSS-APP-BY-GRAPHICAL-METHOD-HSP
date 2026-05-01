import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Metadata
st.set_page_config(page_title="Truss Analysis - Method of Joints", layout="wide")
st.title("🏗️ Pratt & Howe Truss Analysis: Method of Joints1")
st.markdown("---")

# ------------------------------
# INPUTS & LOGIC
# ------------------------------
with st.sidebar:
    st.header("📐 Truss Configuration")
    span = st.number_input("Span of Truss (m)", value=16.0, min_value=5.0, max_value=50.0, step=0.5)
    truss_type = st.selectbox("Selected Truss Configuration", ["Pratt Truss", "Howe Truss"], index=0)
    num_bays = st.slider("Number of bays", 4, 16, 8)
    pitch = st.number_input("Pitch angle (degrees)", value=20.0, min_value=5.0, max_value=45.0, step=1.0)
    
    st.header("⚖️ Loads (kN/m)")
    col1, col2, col3 = st.columns(3)
    with col1: DL = st.number_input("Dead Load (DL)", value=5.0, step=0.5)
    with col2: LL = st.number_input("Live Load (LL)", value=3.0, step=0.5)
    with col3: WL = st.number_input("Wind Load (WL)", value=4.0, step=0.5)
    
    st.header("🔩 Steel Properties")
    fy_choice = st.selectbox("Steel Grade", ["Fe 410", "Fe 450", "Fe 550"], index=0)
    fy_MPa = 250 if fy_choice == "Fe 410" else (450 if fy_choice == "Fe 450" else 550)
    gamma_m0 = st.number_input("Partial Safety Factor (γm0)", value=1.10)
    gamma_m1 = st.number_input("Partial Safety Factor (γm1)", value=1.25)

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
    if n1 in nodes and n2 in nodes and nodes[n1] != nodes[n2]:
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
# METHOD OF JOINTS SOLVER
# ------------------------------
supports = [(bottom_nodes[0], 'x'), (bottom_nodes[0], 'y'), (bottom_nodes[-1], 'y')]

def solve_forces_with_reactions(v_load_per_m, h_load_per_m):
    nm = len(members)
    ns = len(supports)
    unknowns = nm + ns
    A, b = [], []
    
    node_ext_loads = {node: [0.0, 0.0] for node in nodes}
    
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
        
        for j, (n_sup, d_sup) in enumerate(supports):
            if n_sup == node:
                if d_sup == 'x': eqx[nm + j] = 1
                if d_sup == 'y': eqy[nm + j] = 1
        
        if node in top_nodes:
            mult = 0.5 if (node == top_nodes[0] or node == top_nodes[-1]) else 1.0
            Fx, Fy = h_load_per_m * panel * mult, -v_load_per_m * panel * mult
            b.extend([-Fx, -Fy])
        else:
            b.extend([0, 0])
        
        A.extend([eqx, eqy])

    sol, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    m_forces = sol[:nm]
    reactions = sol[nm:]
    return m_forces, reactions

# Solve load combinations
f_dl, r_dl = solve_forces_with_reactions(DL, 0)
f_ll, r_ll = solve_forces_with_reactions(LL, 0)
f_wl, r_wl = solve_forces_with_reactions(0, -WL)

comb1_f = 1.5*(f_dl + f_ll)
comb2_f = 1.5*(f_dl + f_wl)
comb3_f = 1.2*(f_dl + f_ll + f_wl)

# ------------------------------
# FINAL MEMBER FORCE TABLE
# ------------------------------
final_data = []
for i in range(len(members)):
    max_f = max([comb1_f[i], comb2_f[i], comb3_f[i]], key=abs)
    n1, n2 = members[i]['nodes']
    L = np.hypot(nodes[n1][0] - nodes[n2][0], nodes[n1][1] - nodes[n2][1])
    final_data.append({
        "Member": f"{n1}-{n2}", 
        "Type": members[i]['type'], 
        "Length (m)": round(L, 2), 
        "Design Force (kN)": round(max_f, 2), 
        "Nature": "Tension" if max_f > 0 else "Compression"
    })

st.subheader("📋 Final Member Forces (Summary)")
st.dataframe(pd.DataFrame(final_data), use_container_width=True)
