import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Metadata
st.set_page_config(page_title="Pratt & Howe Truss Design - IS 800:2007", layout="wide")
st.title("🏗️ Pratt & Howe Steel Truss Analysis & Design as per IS 800:2007")
st.markdown("#### Developer: Dr. Hiteshkumar Santosh Patil, Assistant Professor, RCPIT Shirpur")
st.markdown("---")

# ------------------------------
# INPUTS & LOGIC
# ------------------------------
with st.sidebar:
    st.header("📐 Truss Configuration")
    span = st.number_input("Span of Truss (m)", value=16.0, min_value=5.0, max_value=50.0, step=0.5)
    
    # Truss type selection
    truss_type = st.selectbox("Selected Truss Configuration", 
                              ["Pratt Truss", "Howe Truss"], 
                              index=0,
                              help="Pratt: Diagonals slope towards center (tension) | Howe: Diagonals slope away from center (compression)")
    
    num_bays = st.slider("Number of bays", 4, 16, 8, help="Even number recommended for symmetry")
    pitch = st.number_input("Pitch angle (degrees)", value=20.0, min_value=5.0, max_value=45.0, step=1.0)
    
    st.header("⚖️ Loads (kN/m along plan length)")
    col1, col2, col3 = st.columns(3)
    with col1:
        DL = st.number_input("Dead Load (DL)", value=5.0, step=0.5, help="Self weight + finishes in kN/m")
    with col2:
        LL = st.number_input("Live Load (LL)", value=3.0, step=0.5, help="Live load in kN/m")
    with col3:
        WL = st.number_input("Wind Load (WL)", value=4.0, step=0.5, help="+ve = downward, -ve = uplift in kN/m")
    
    st.header("🔩 Steel Properties (IS 800:2007)")
    fy_choice = st.selectbox("Steel Grade", ["Fe 410", "Fe 450", "Fe 550"], index=0)
    if fy_choice == "Fe 410":
        fy_MPa = 250
    elif fy_choice == "Fe 450":
        fy_MPa = 450
    else:
        fy_MPa = 550
    
    gamma_m0 = st.number_input("Partial Safety Factor (γm0)", value=1.10, step=0.05, help="For yielding")
    gamma_m1 = st.number_input("Partial Safety Factor (γm1)", value=1.25, step=0.05, help="For rupture/buckling")
    
    st.caption("Load combinations as per IS 800:2007 Clause 5.3")
    st.caption("1. 1.5(DL + LL) | 2. 1.5(DL + WL) | 3. 1.2(DL + LL + WL)")

# ------------------------------
# TRUSS GEOMETRY ENGINE (Integrated custom logic)
# ------------------------------
panel = span / num_bays
theta = np.radians(pitch)
nodes = {}
nid = 1

# Calculate ridge height
ridge_height = (span / 2) * np.tan(theta)

# Bottom nodes
bottom_nodes = []
for i in range(num_bays + 1):
    nodes[nid] = (i * panel, 0)
    bottom_nodes.append(nid)
    nid += 1

# Top nodes
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

# Chords
for i in range(len(bottom_nodes) - 1):
    add_member(bottom_nodes[i], bottom_nodes[i+1], "Bottom Chord")
for i in range(len(top_nodes) - 1):
    add_member(top_nodes[i], top_nodes[i+1], "Top Chord")

# Verticals
for i in range(len(bottom_nodes)):
    add_member(bottom_nodes[i], top_nodes[i], "Vertical")

# Diagonals logic based on Choice
half_panels = num_bays // 2
for i in range(num_bays):
    if truss_type == "Pratt Truss":
        # Pratt: Diagonals slope towards center (tension)
        if i < half_panels:
            add_member(bottom_nodes[i+1], top_nodes[i], "Diagonal")
        else:
            add_member(bottom_nodes[i], top_nodes[i+1], "Diagonal")
    else:  # Howe Truss
        # Howe: Diagonals slope away from center (compression)
        if i < half_panels:
            add_member(bottom_nodes[i], top_nodes[i+1], "Diagonal")
        else:
            add_member(bottom_nodes[i+1], top_nodes[i], "Diagonal")

# ------------------------------
# MATRIX SOLVER FOR TRUSS FORCES
# ------------------------------
supports = [(bottom_nodes[0], 'x'), (bottom_nodes[0], 'y'), (bottom_nodes[-1], 'y')]

def solve_forces(vertical_load, horizontal_load):
    nm = len(members)
    if nm == 0: return np.zeros(0)
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
                if L > 1e-9:
                    eqx[i], eqy[i] = (x2 - x1) / L, (y2 - y1) / L
        for j, (n, d) in enumerate(supports):
            if n == node:
                if d == 'x': eqx[nm + j] = 1
                if d == 'y': eqy[nm + j] = 1
        Fx, Fy = horizontal_load * panel, -vertical_load * panel
        if node in top_nodes:
            mult = 0.5 if (node == top_nodes[0] or node == top_nodes[-1]) else 1.0
            A.extend([eqx, eqy])
            b.extend([-Fx * mult, -Fy * mult])
        else:
            A.extend([eqx, eqy])
            b.extend([0, 0])
    sol, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return sol[:nm]

f_dl = solve_forces(DL, 0)
f_ll = solve_forces(LL, 0)
f_wl = solve_forces(0, -WL)
min_len = len(members)
comb1 = 1.5 * (f_dl + f_ll)
comb2 = 1.5 * (f_dl + f_wl)
comb3 = 1.2 * (f_dl + f_ll + f_wl)

# ------------------------------
# SECTION DESIGN LOGIC
# ------------------------------
def design_double_angle_section(force_kN, length_m, fy, gamma_m0, gamma_m1):
    force_abs = abs(force_kN) * 1000
    sections = [
        {"name": "ISA 40x40x5", "area": 758, "r_min": 12.2},
        {"name": "ISA 50x50x6", "area": 1136, "r_min": 15.3},
        {"name": "ISA 65x65x6", "area": 1488, "r_min": 19.9},
        {"name": "ISA 75x75x6", "area": 1732, "r_min": 23.0},
        {"name": "ISA 100x100x8", "area": 3058, "r_min": 30.7},
    ]
    for sec in sections:
        if force_kN > 0: # Tension
            cap = (sec["area"] * fy / gamma_m0) / 1000
            if cap >= abs(force_kN): return sec["name"], sec["area"], "✓ Adequate"
        else: # Compression
            slenderness = (length_m * 1000) / sec["r_min"]
            if slenderness > 180: continue
            f_cc = (math.pi**2 * 2e5) / (slenderness**2)
            lambda_e = math.sqrt(fy / f_cc)
            phi = 0.5 * (1 + 0.34 * (lambda_e - 0.2) + lambda_e**2)
            chi = min(1.0, 1 / (phi + math.sqrt(phi**2 - lambda_e**2)))
            cap = (sec["area"] * chi * fy / gamma_m0) / 1000
            if cap >= abs(force_kN): return sec["name"], sec["area"], "✓ Adequate"
    return sections[-1]["name"] + " (Limit)", sections[-1]["area"], "⚠️ High Stress"

# ------------------------------
# DATA PROCESSING & VISUALIZATION
# ------------------------------
final_data = []
for i in range(len(members)):
    max_f = max([comb1[i], comb2[i], comb3[i]], key=abs)
    n1, n2 = members[i]['nodes']
    L = np.hypot(nodes[n1][0] - nodes[n2][0], nodes[n1][1] - nodes[n2][1])
    sec, area, stat = design_double_angle_section(max_f, L, fy_MPa, gamma_m0, gamma_m1)
    final_data.append({"Member": f"{n1}-{n2}", "Type": members[i]['type'], "Length (m)": round(L, 2), "Design Force (kN)": round(max_f, 2), "Nature": "Tension" if max_f > 0 else "Compression", "Section": sec, "Status": stat})

df = pd.DataFrame(final_data)

# Visualization with Node Numbering
st.subheader(f"📊 {truss_type} Structural View - Member Forces")
fig, ax = plt.subplots(figsize=(12, 5))

# Draw members
for i, m in enumerate(members):
    n1, n2 = m['nodes']
    f = final_data[i]['Design Force (kN)']
    color = '#e74c3c' if f > 0 else '#3498db'  # Red for tension, Blue for compression
    ax.plot([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]], color=color, lw=2, alpha=0.8)

# Draw nodes with numbers
for node_id, (x, y) in nodes.items():
    # Plot node point
    ax.plot(x, y, 'ko', markersize=6, zorder=5)
    # Add node number label
    ax.annotate(str(node_id), (x, y), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
                zorder=6)

ax.set_aspect('equal')
ax.set_title(f"{truss_type} Geometry with Node Numbers and Member Forces", fontsize=14, fontweight='bold')
ax.set_xlabel("Span (m)", fontsize=12)
ax.set_ylabel("Height (m)", fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.5, span + 0.5)
ax.set_ylim(-0.5, ridge_height + 0.5)

# Add legend for color coding
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', edgecolor='#e74c3c', label='Tension Members', alpha=0.8),
                   Patch(facecolor='#3498db', edgecolor='#3498db', label='Compression Members', alpha=0.8)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

st.pyplot(fig)

# Display node information summary
st.subheader("📍 Node Information")
node_df = pd.DataFrame([(node_id, f"{nodes[node_id][0]:.2f}", f"{nodes[node_id][1]:.2f}") 
                         for node_id in sorted(nodes.keys())], 
                        columns=["Node No.", "X-Coordinate (m)", "Y-Coordinate (m)"])
st.dataframe(node_df, use_container_width=True)

# Table of design forces in members
st.subheader("📋 Design Forces in Members (IS 800:2007)")
st.dataframe(df, use_container_width=True)

# Download option for member forces
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df)
st.download_button(
    label="📥 Download Member Forces Table (CSV)",
    data=csv,
    file_name=f"{truss_type.replace(' ', '_')}_member_forces.csv",
    mime="text/csv",
)

# Additional note about color coding
st.info("📌 **Node Numbering:** Each node is labeled with its ID number. "
        "**Color Coding:** 🔴 Red members are in TENSION | 🔵 Blue members are in COMPRESSION")
