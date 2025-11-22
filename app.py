import streamlit as st
import plotly.graph_objects as go
from geometry import Mesh
import numpy as np
from typing import Set

st.set_page_config(
    page_title="Subdivision Surface Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background-color: #1a202c;
    }
    
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    [data-testid="stSidebar"] {
        background-color: #2d3748;
        border-right: 1px solid #4a5568;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #4a5568;
        color: white;
        border: 1px solid #4a5568;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #2d3748;
        border: 1px solid #4a5568;
    }
    
    h1, h2, h3 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown, p, label {
        color: #cbd5e0 !important;
    }
</style>
""", unsafe_allow_html=True)

def plot_mesh(mesh: Mesh, show_shaded=True, show_wireframe=False, opacity=1.0, ambient=0.3, diffuse=0.8, specular=0.2, roughness=0.4, colorscale='Blues'):
    V, tris = mesh.to_plotly_mesh()
    fig = go.Figure()
    
    if show_shaded:
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        i = [t[0] for t in tris]
        j = [t[1] for t in tris]
        k = [t[2] for t in tris]
        
        mesh3d = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            opacity=opacity,
            colorscale=colorscale,
            intensity=z,
            flatshading=False,
            lighting=dict(
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                roughness=roughness,
                fresnel=0.2
            ),
            lightposition=dict(x=2000, y=2000, z=3000),
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        )
        fig.add_trace(mesh3d)

    if show_wireframe:
        edges = mesh.get_edge_list()
        crease_edges = {he.edge_id for he in mesh.halfedges if he.is_crease}
        
        regular_x, regular_y, regular_z = [], [], []
        crease_x, crease_y, crease_z = [], [], []
        
        for eid, a, b in edges:
            if eid in crease_edges:
                crease_x.extend([V[a][0], V[b][0], None])
                crease_y.extend([V[a][1], V[b][1], None])
                crease_z.extend([V[a][2], V[b][2], None])
            else:
                regular_x.extend([V[a][0], V[b][0], None])
                regular_y.extend([V[a][1], V[b][1], None])
                regular_z.extend([V[a][2], V[b][2], None])
        
        # Hologram style if only wireframe is shown
        is_hologram = not show_shaded
        
        if is_hologram:
            # Outer glow
            fig.add_trace(go.Scatter3d(
                x=regular_x, y=regular_y, z=regular_z,
                mode='lines',
                line=dict(color='rgba(0, 255, 255, 0.2)', width=10),
                hoverinfo='none',
                name='Glow'
            ))
        
        if regular_x:
            line_color = 'rgba(0, 255, 255, 0.8)' if is_hologram else 'rgba(255, 255, 255, 0.6)'
            fig.add_trace(go.Scatter3d(
                x=regular_x, y=regular_y, z=regular_z,
                mode='lines',
                line=dict(color=line_color, width=2.5),
                hoverinfo='skip',
                name='Edges'
            ))
        
        if crease_x:
            fig.add_trace(go.Scatter3d(
                x=crease_x, y=crease_y, z=crease_z,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.8)', width=4),
                hoverinfo='skip',
                name='Crease Edges'
            ))

    fig.update_layout(
        title=dict(text="3D Mesh Visualization", font=dict(size=20, color='white')),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            aspectmode='data',
            bgcolor='rgba(10, 20, 40, 0.9)' if not show_shaded and show_wireframe else 'rgba(0,0,0,0)',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )
    return fig

def mesh_from_uploaded(uploaded) -> Mesh:
    data = uploaded.getvalue().decode('utf-8')
    return Mesh.load_obj_text(data)

def init_state():
    if 'mesh' not in st.session_state:
        st.session_state.mesh = None
        st.session_state.base_mesh = None
        st.session_state.history = []
        st.session_state.crease_edges = set()

init_state()

st.sidebar.title("Controls")

with st.sidebar.expander("Load Model", expanded=True):
    uploaded = st.file_uploader("Upload OBJ", type=['obj'])
    if uploaded:
        mesh = mesh_from_uploaded(uploaded)
        st.session_state.mesh = st.session_state.base_mesh = mesh
        st.session_state.history = [mesh]
        st.session_state.crease_edges = set()

    st.markdown("**Sample Models**")
    
    samples = ["Cube", "Triangle", "Octahedron", "Pyramid", "Icosahedron"]
    for sample in samples:
        if st.button(sample, use_container_width=True):
            try:
                with open(f'assets/{sample.lower()}.obj', 'r') as f:
                    mesh = Mesh.load_obj_text(f.read())
                    st.session_state.mesh = st.session_state.base_mesh = mesh
                    st.session_state.history = [mesh]
                    st.session_state.crease_edges = set()
            except FileNotFoundError:
                st.error(f"Sample model '{sample.lower()}.obj' not found.")

with st.sidebar.expander("Generate Model"):
    if st.button("Torus", use_container_width=True):
        mesh = Mesh.create_torus()
        st.session_state.mesh = st.session_state.base_mesh = mesh
        st.session_state.history = [mesh]
    if st.button("Icosphere", use_container_width=True):
        mesh = Mesh.create_icosphere()
        st.session_state.mesh = st.session_state.base_mesh = mesh
        st.session_state.history = [mesh]

with st.sidebar.expander("Subdivision", expanded=True):
    is_triangle_mesh = st.session_state.mesh.is_triangle_mesh() if st.session_state.mesh else False
    
    algo_options = ['Catmull-Clark', 'Doo-Sabin']
    if is_triangle_mesh:
        algo_options.extend(['Loop', 'Sqrt(3)'])
    
    algo = st.selectbox("Algorithm", algo_options)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Subdivide', use_container_width=True):
            if st.session_state.mesh:
                try:
                    with st.spinner('Subdividing...'):
                        if algo == 'Catmull-Clark':
                            new_mesh = st.session_state.mesh.subdivide_catmull_clark(st.session_state.crease_edges)
                        elif algo == 'Loop':
                            new_mesh = st.session_state.mesh.subdivide_loop(st.session_state.crease_edges)
                        elif algo == "Sqrt(3)":
                            new_mesh = st.session_state.mesh.subdivide_sqrt3()
                        else:
                            new_mesh = st.session_state.mesh.subdivide_doo_sabin()
                        st.session_state.mesh = new_mesh
                        st.session_state.history.append(new_mesh)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning('Load a mesh first.')
    with col2:
        if st.button('Reset', use_container_width=True):
            if st.session_state.base_mesh:
                st.session_state.mesh = st.session_state.base_mesh
                st.session_state.history = [st.session_state.base_mesh]
                st.session_state.crease_edges = set()

with st.sidebar.expander("Creases & Export"):
    if st.session_state.mesh:
        edges = st.session_state.mesh.get_edge_list()
        edge_options = [f"{eid}: {a}-{b}" for eid, a, b in edges]
        selected = st.multiselect('Crease edges', edge_options, default=[f"{i}: {a}-{b}" for i,a,b in edges if i in st.session_state.crease_edges])
        
        ids = {int(s.split(':')[0]) for s in selected}
        if st.button('Apply Creases', use_container_width=True):
            st.session_state.crease_edges = ids
            st.session_state.mesh.mark_crease_edges(ids)
            st.rerun()
        
        obj_text = st.session_state.mesh.to_obj_text()
        st.download_button('Download OBJ', obj_text, file_name='mesh.obj', mime='text/plain', use_container_width=True)

st.title("Subdivision Surface Viewer")

col1, col2 = st.columns([3, 1])

with col1:
    st.header("3D Visualization")
    
    with st.expander("Visualization Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            show_shaded = st.checkbox('Shaded', value=True)
        with c2:
            show_wireframe = st.checkbox('Wireframe', value=True)

        if show_shaded:
            c1, c2 = st.columns(2)
            with c1:
                opacity = st.slider("Opacity", 0.0, 1.0, 1.0)
                ambient = st.slider("Ambient", 0.0, 1.0, 0.3)
            with c2:
                diffuse = st.slider("Diffuse", 0.0, 1.0, 0.8)
                specular = st.slider("Specular", 0.0, 1.0, 0.2)
            
            roughness = st.slider("Roughness", 0.0, 1.0, 0.4)
            colorscale = st.selectbox("Colorscale", ['Blues', 'Greys', 'Viridis', 'Cividis', 'Plasma', 'Turbo', 'Rainbow', 'Ice', 'Electric'])
        else:
            opacity = 1.0
            ambient = 0.3
            diffuse = 0.8
            specular = 0.2
            roughness = 0.4
            colorscale = 'Blues'

    if st.session_state.mesh:
        fig = plot_mesh(st.session_state.mesh, show_shaded, show_wireframe, opacity, ambient, diffuse, specular, roughness, colorscale)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Load or generate a model from the sidebar to start.')

with col2:
    st.header("Mesh Statistics")
    if st.session_state.mesh:
        st.metric("Vertices", len(st.session_state.mesh.vertices))
        st.metric("Faces", len(st.session_state.mesh.faces))
        st.metric("Edges", len(st.session_state.mesh.edge_map))
        st.metric("Crease Edges", len(st.session_state.crease_edges))
        st.metric("Subdivision Level", len(st.session_state.history) - 1)
        
        if st.session_state.mesh.is_triangle_mesh():
            st.success("Triangle Mesh")
        else:
            st.info("Polygon Mesh")
    else:
        st.write("No mesh loaded.")