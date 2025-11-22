# Advanced Subdivision Surface Viewer

A comprehensive Streamlit application for loading, visualizing, and subdividing 3D meshes using various subdivision algorithms. Built with a half-edge topological data structure for efficient adjacency queries.

## ✨ Features

### Core Implementation (30 points)
- **Half-Edge Data Structure**: Complete implementation with `Vertex`, `HalfEdge`, and `Face` classes
- **OBJ File Loader**: Parses .obj files and constructs topology from face data
- **3D Visualization**: Interactive Plotly-based rendering with:
  - Shaded polygon display with customizable lighting
  - Wireframe overlay showing edge connectivity
  - Crease edge highlighting (red lines)
  - Multiple color schemes
  - Adjustable opacity and lighting

### Sample Models (10 points)
Includes diverse test models demonstrating different mesh types:
- **Cube** - 6 quad faces, valence-3 vertices
- **Tetrahedron** - 4 triangles, simple triangle mesh
- **Octahedron** - 8 triangles, valence-4 vertices
- **Pyramid** - Mixed polygons (1 quad + 4 triangles)
- **Icosahedron** - 20 triangles, valence-5 vertices

### Subdivision Algorithms

#### Catmull-Clark Subdivision (20 points)
- Works on arbitrary polygonal meshes
- Generates smooth quad-dominant surfaces
- **Crease support**: Hard edges preserved during subdivision
- Implements face points, edge points, and vertex smoothing rules

#### Loop Subdivision (20 points)
- Triangle-only meshes
- Generates smooth surfaces with improved vertex positions
- **Crease support**: Boundary and crease edge handling
- Uses 1-to-4 triangle split pattern

#### Sqrt(3) Subdivision (20 points)
- Triangle-only meshes
- Moderate refinement (√3 factor)
- Inserts face centroids and creates new connectivity

#### Doo-Sabin Subdivision (20 points)
- Works on arbitrary polygonal meshes
- Dual subdivision scheme
- Generates new faces from vertex neighborhoods

### Crease Edge Support (20 points)
- Interactive edge selection via multiselect UI
- Visual distinction (red lines in 3D view)
- Supported in Catmull-Clark and Loop algorithms
- Preserves sharp features during subdivision

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
hw4_cf/
├── app.py              # Streamlit UI with visualization controls
├── geometry.py         # Half-edge mesh data structure and algorithms
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── assets/            # Sample OBJ models
    ├── cube.obj
    ├── triangle.obj
    ├── octahedron.obj
    ├── pyramid.obj
    └── icosahedron.obj
```

## 🎯 Usage Guide

### Loading Models
1. **Upload**: Use the file uploader to load custom .obj files
2. **Samples**: Click any sample button to load predefined models

### Subdivision
1. Select an algorithm (availability depends on mesh type):
   - **Catmull-Clark & Doo-Sabin**: All meshes
   - **Loop & Sqrt(3)**: Triangle meshes only
2. Click "▶️ Subdivide" to apply one iteration
3. Click "🔄 Reset" to return to the original mesh

### Crease Edges
1. Expand "🔗 Creases & Export" in sidebar
2. Select edges from the dropdown (shows as `edge_id: v1→v2`)
3. Click "✓ Apply Creases" to mark them
4. Crease edges appear in red in wireframe view

### Visualization
- **Shaded View**: Toggle smooth shaded surfaces
- **Wireframe**: Show edge connectivity
- **Opacity**: Adjust transparency (0-1)
- **Ambient**: Control ambient lighting
- **Color Scheme**: Choose from 6 color palettes

### Export
- Click "💾 Download OBJ" to save current mesh state
- File name includes subdivision level

## 🔍 Implementation Details

### Half-Edge Topology
The core data structure maintains:
- **Vertices**: Position and outgoing half-edge reference
- **Half-Edges**: Origin vertex, twin, next, prev, face, edge ID, crease flag
- **Faces**: Reference to a boundary half-edge
- **Edge Map**: Undirected edges for efficient lookup

Topology is inferred from OBJ face data by:
1. Creating half-edges for each face edge
2. Linking next/prev within faces
3. Finding twins by reverse lookup in directed edge map
4. Assigning unique edge IDs to undirected edges

### Subdivision Implementation
Each algorithm uses the half-edge structure to:
- Navigate vertex neighborhoods (via twin/next)
- Find adjacent faces and edges
- Compute new vertex positions using local geometry
- Generate new connectivity for refined mesh

### Crease Handling
- Crease edges use midpoint rule (no smoothing)
- Vertices on creases get modified smoothing weights
- Marked via `is_crease` flag on half-edges

## 📊 Requirements Coverage

| Requirement | Points | Status | Implementation |
|------------|--------|--------|----------------|
| Half-edge data structure | 30 | ✅ | `geometry.py` - Vertex, HalfEdge, Face classes |
| OBJ loading & display | | ✅ | `load_obj_text()`, Plotly 3D rendering |
| Sample model collection | 10 | ✅ | 5 diverse models in `assets/` |
| Catmull-Clark subdivision | 20 | ✅ | `subdivide_catmull_clark()` |
| Loop subdivision | 20 | ✅ | `subdivide_loop()` |
| Sqrt(3) subdivision | 20 | ✅ | `subdivide_sqrt3()` |
| Doo-Sabin subdivision | 20 | ✅ | `subdivide_doo_sabin()` |
| Crease edge support | 20 | ✅ | Interactive selection, visual feedback |
| **Total** | **140** | | *Max grade capped at 100* |

## 🛠️ Dependencies

- `streamlit>=1.24` - Web UI framework
- `numpy>=1.25` - Numerical computations
- `plotly>=5.0` - Interactive 3D visualization

## 📝 Notes

- Topology verification available in "🔍 Topology Details" expander
- Edge list shows first 200 edges with crease indicators
- Subdivision level counter tracks iteration depth
- Color schemes: Viridis, Plasma, Turbo, Rainbow, Ice, Electric

## 🎨 UI Features

- Glassmorphism design with gradient backgrounds
- Responsive layout with sidebar controls
- Real-time mesh statistics
- Interactive 3D camera controls (drag to rotate, scroll to zoom)
- Status messages for user feedback
- Tooltip help text on controls
