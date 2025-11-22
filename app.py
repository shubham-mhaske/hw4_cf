import streamlit as st
import plotly.graph_objects as go
from geometry import Mesh
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SubSurf Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧊"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (The "Sleek Aesthetic")
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Theme Overrides */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e6edf3 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Metric Containers */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.5rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }
    
    /* Custom Card Containers */
    .css-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3);
    }
    
    /* Secondary Buttons */
    [data-testid="stExpander"] button {
        border: 1px solid #30363d;
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def mesh_from_uploaded(uploaded) -> Mesh:
    """Cache the heavy parsing of OBJ files."""
    data = uploaded.getvalue().decode('utf-8')
    return Mesh.load_obj_text(data)

@st.cache_data
def mesh_from_sample(filename) -> Mesh:
    with open(f'assets/{filename}', 'r') as f:
        return Mesh.load_obj_text(f.read())

def plot_mesh(mesh: Mesh, 
              show_shaded=True, 
              show_wireframe=True, 
              opacity=1.0, 
              colorscale='Viridis',
              wireframe_color='rgba(255, 255, 255, 0.3)',
              bg_color='rgba(0,0,0,0)'):
    
    # 1. Compute geometry data
    V, tris = mesh.to_plotly_mesh()
    
    # 2. Compute Smooth Normals (The Secret to high-quality rendering)
    # vertex_normals = mesh.compute_vertex_normals()
    
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    i = [t[0] for t in tris]
    j = [t[1] for t in tris]
    k = [t[2] for t in tris]
    
    data = []

    # 3. Shaded Surface Trace
    if show_shaded:
        mesh3d = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            opacity=opacity,
            colorscale=colorscale,
            intensity=z, # Color based on height
            flatshading=False, # CRITICAL: False enables smooth interpolation
            lighting=dict(
                ambient=0.4,
                diffuse=0.7,
                specular=0.5, # Plastic/Glossy look
                roughness=0.1,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=200, z=500),
            name='Surface',
            showscale=False
        )
        data.append(mesh3d)

    # 4. Wireframe Trace (Overlay)
    if show_wireframe:
        edges = mesh.get_edge_list()
        crease_ids = {he.edge_id for he in mesh.halfedges if he.is_crease}
        
        # Batch lines for performance
        reg_x, reg_y, reg_z = [], [], []
        crease_x, crease_y, crease_z = [], [], []
        
        for eid, a, b in edges:
            p1, p2 = V[a], V[b]
            if eid in crease_ids:
                crease_x.extend([p1[0], p2[0], None])
                crease_y.extend([p1[1], p2[1], None])
                crease_z.extend([p1[2], p2[2], None])
            else:
                reg_x.extend([p1[0], p2[0], None])
                reg_y.extend([p1[1], p2[1], None])
                reg_z.extend([p1[2], p2[2], None])

        # Regular Edges
        if reg_x:
            data.append(go.Scatter3d(
                x=reg_x, y=reg_y, z=reg_z,
                mode='lines',
                line=dict(color=wireframe_color, width=2),
                name='Edges', hoverinfo='none'
            ))
        
        # Crease Edges (Highlighted)
        if crease_x:
            data.append(go.Scatter3d(
                x=crease_x, y=crease_y, z=crease_z,
                mode='lines',
                line=dict(color='#ff4b4b', width=5),
                name='Creases', hoverinfo='none'
            ))

    # 5. Layout Configuration
    layout = go.Layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(projection=dict(type='perspective')),
            bgcolor=bg_color
        ),
        # UIREVISION: Keeps camera angle stable during updates
        uirevision='constant',
        modebar=dict(bgcolor='rgba(0,0,0,0)', color='white')
    )
    
    return go.Figure(data=data, layout=layout)

# -----------------------------------------------------------------------------
# 4. STATE MANAGEMENT
# -----------------------------------------------------------------------------
if 'mesh' not in st.session_state:
    st.session_state.mesh = None
    st.session_state.base_mesh = None
    st.session_state.history = []
    st.session_state.crease_edges = set()

# -----------------------------------------------------------------------------
# 5. SIDEBAR UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧊 SubSurf Pro")
    
    # Top-level tabs for better organization
    tab_model, tab_edit, tab_view = st.tabs(["📂 Model", "⚡ Edit", "🎨 View"])
    
    # --- TAB 1: MODEL (Import & Generate) ---
    with tab_model:
        st.markdown("### Source")
        source_type = st.radio("Input Type", ["Upload", "Primitive", "Library"], label_visibility="collapsed")
        
        if source_type == "Upload":
            uploaded = st.file_uploader("Drop OBJ file", type=['obj'])
            if uploaded:
                if st.button("Load Uploaded File", type="primary"):
                    mesh = mesh_from_uploaded(uploaded)
                    st.session_state.mesh = st.session_state.base_mesh = mesh
                    st.session_state.history = [mesh]
                    st.session_state.crease_edges = set()
                    st.rerun()

        elif source_type == "Primitive":
            prim_type = st.selectbox("Shape", ["Cube", "Torus", "Icosphere", "Cylinder", "Grid"])
            
            if prim_type == "Cube":
                size = st.slider("Size", 0.1, 5.0, 1.0)
                if st.button("Generate Cube"):
                    mesh = Mesh.create_cube(size)
                    st.session_state.mesh = st.session_state.base_mesh = mesh
                    st.session_state.history = [mesh]
                    st.rerun()
                    
            elif prim_type == "Torus":
                r_maj = st.slider("Major Radius", 0.5, 5.0, 1.0)
                r_min = st.slider("Minor Radius", 0.1, 2.0, 0.3)
                seg_maj = st.slider("Major Segments", 3, 64, 32)
                seg_min = st.slider("Minor Segments", 3, 32, 16)
                if st.button("Generate Torus"):
                    mesh = Mesh.create_torus(r_maj, r_min, seg_maj, seg_min)
                    st.session_state.mesh = st.session_state.base_mesh = mesh
                    st.session_state.history = [mesh]
                    st.rerun()
                    
            elif prim_type == "Icosphere":
                rad = st.slider("Radius", 0.1, 5.0, 1.0)
                sub = st.slider("Subdivisions", 0, 5, 2)
                if st.button("Generate Icosphere"):
                    mesh = Mesh.create_icosphere(rad, sub)
                    st.session_state.mesh = st.session_state.base_mesh = mesh
                    st.session_state.history = [mesh]
                    st.rerun()
            
            elif prim_type == "Cylinder":
                rad = st.slider("Radius", 0.1, 5.0, 1.0)
                h = st.slider("Height", 0.1, 10.0, 2.0)
                seg = st.slider("Segments", 3, 64, 16)
                if st.button("Generate Cylinder"):
                    mesh = Mesh.create_cylinder(rad, h, seg)
                    st.session_state.mesh = st.session_state.base_mesh = mesh
                    st.session_state.history = [mesh]
                    st.rerun()
                    
            elif prim_type == "Grid":
                w = st.slider("Width", 1.0, 10.0, 2.0)
                h = st.slider("Height", 1.0, 10.0, 2.0)
                seg = st.slider("Segments", 1, 20, 4)
                if st.button("Generate Grid"):
                    mesh = Mesh.create_grid(w, h, seg, seg)
                    st.session_state.mesh = st.session_state.base_mesh = mesh
                    st.session_state.history = [mesh]
                    st.rerun()

        elif source_type == "Library":
            samples = ["Cube", "Octahedron", "Icosahedron", "Pyramid", "Triangle"]
            selected_sample = st.selectbox("Select Sample", samples)
            if st.button(f"Load {selected_sample}"):
                try:
                    mesh = mesh_from_sample(f"{selected_sample.lower()}.obj")
                    st.session_state.mesh = st.session_state.base_mesh = mesh
                    st.session_state.history = [mesh]
                    st.session_state.crease_edges = set()
                    st.rerun()
                except:
                    st.toast(f"{selected_sample} not found in assets", icon="⚠️")

    # --- TAB 2: EDIT (Subdivide & Crease) ---
    with tab_edit:
        if st.session_state.mesh:
            st.markdown("### Subdivision")
            
            is_tri = st.session_state.mesh.is_triangle_mesh()
            algos = ['Catmull-Clark', 'Doo-Sabin']
            if is_tri: algos = ['Loop', 'Sqrt(3)'] + algos
            
            selected_algo = st.selectbox("Algorithm", algos)
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("⚡ Subdivide", type="primary"):
                    with st.spinner("Processing..."):
                        m = st.session_state.mesh
                        creases = st.session_state.crease_edges
                        try:
                            if selected_algo == 'Catmull-Clark':
                                nm = m.subdivide_catmull_clark(creases)
                            elif selected_algo == 'Loop':
                                nm = m.subdivide_loop(creases)
                            elif selected_algo == 'Sqrt(3)':
                                nm = m.subdivide_sqrt3()
                            else:
                                nm = m.subdivide_doo_sabin()
                            
                            st.session_state.mesh = nm
                            st.session_state.history.append(nm)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            with c2:
                if st.button("↩ Undo", disabled=len(st.session_state.history) <= 1):
                    if len(st.session_state.history) > 1:
                        st.session_state.history.pop()
                        st.session_state.mesh = st.session_state.history[-1]
                        st.rerun()
            
            if st.button("↺ Reset to Original", use_container_width=True):
                st.session_state.mesh = st.session_state.base_mesh
                st.session_state.history = [st.session_state.base_mesh]
                st.rerun()

            st.markdown("---")
            st.markdown("### Feature Edges")
            
            edges = st.session_state.mesh.get_edge_list()
            opts = [f"{eid}: {a}-{b}" for eid, a, b in edges]
            defaults = [f"{i}: {a}-{b}" for i,a,b in edges if i in st.session_state.crease_edges]
            
            sel = st.multiselect("Select Hard Edges", opts, default=defaults)
            new_creases = {int(s.split(':')[0]) for s in sel}
            
            if new_creases != st.session_state.crease_edges:
                if st.button("Apply Creases"):
                    st.session_state.crease_edges = new_creases
                    st.session_state.mesh.mark_crease_edges(new_creases)
                    st.rerun()
        else:
            st.info("Load a model first")

    # --- TAB 3: VIEW (Visualization & Export) ---
    with tab_view:
        if st.session_state.mesh:
            st.markdown("### Display Settings")
            show_shaded = st.toggle("Surface", True)
            show_wire = st.toggle("Wireframe", True)
            opacity = st.slider("Opacity", 0.0, 1.0, 1.0)
            theme = st.selectbox("Color Theme", ["Viridis", "Plasma", "Blues", "Magma", "Inferno", "Cividis"])
            wire_color = st.color_picker("Wireframe Color", "#FFFFFF")
            bg_color = st.color_picker("Background Color", "#0e1117")
            
            st.markdown("---")
            st.markdown("### Export")
            obj_data = st.session_state.mesh.to_obj_text()
            st.download_button(
                label="💾 Download OBJ",
                data=obj_data,
                file_name="subdivided_mesh.obj",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Load a model first")

# -----------------------------------------------------------------------------
# 6. MAIN AREA
# -----------------------------------------------------------------------------

# Top Metrics Row
if st.session_state.mesh:
    m = st.session_state.mesh
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vertices", len(m.vertices))
    c2.metric("Faces", len(m.faces))
    c3.metric("Type", "Triangles" if m.is_triangle_mesh() else "Polygons")
    c4.metric("Level", len(st.session_state.history)-1)

# Main Viewport
if st.session_state.mesh:
    fig = plot_mesh(st.session_state.mesh, show_shaded, show_wire, opacity, theme, wire_color, bg_color)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
else:
    st.info("👈 Load a model to begin")