"""
geometry.py
Optimized for memory with __slots__ and adds Vertex Normal computation for smooth shading.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

class Vertex:
    __slots__ = ['pos', 'halfedge', 'id']
    def __init__(self, pos: np.ndarray, id: int):
        self.pos = np.asarray(pos, dtype=float)
        self.halfedge: Optional[HalfEdge] = None
        self.id = id

class HalfEdge:
    __slots__ = ['origin', 'twin', 'next', 'prev', 'face', 'edge_id', 'is_crease']
    def __init__(self):
        self.origin: Optional[Vertex] = None
        self.twin: Optional[HalfEdge] = None
        self.next: Optional[HalfEdge] = None
        self.prev: Optional[HalfEdge] = None
        self.face: Optional[Face] = None
        self.edge_id: Optional[int] = None
        self.is_crease: bool = False

class Face:
    __slots__ = ['halfedge', 'id']
    def __init__(self, id: int):
        self.halfedge: Optional[HalfEdge] = None
        self.id = id

class Mesh:
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.halfedges: List[HalfEdge] = []
        self.faces: List[Face] = []
        self.edge_map: Dict[Tuple[int, int], int] = {}
        self.edge_halfedges: Dict[int, List[HalfEdge]] = {}

    @staticmethod
    def from_data(positions: List[Tuple[float, float, float]], faces_idx: List[List[int]]) -> "Mesh":
        m = Mesh()
        for i, p in enumerate(positions):
            m.vertices.append(Vertex(np.array(p, dtype=float), i))

        he_list: List[HalfEdge] = []
        edge_dir_map: Dict[Tuple[int, int], HalfEdge] = {}

        for fi, fverts in enumerate(faces_idx):
            face = Face(fi)
            n = len(fverts)
            hes = [HalfEdge() for _ in range(n)]
            for i, he in enumerate(hes):
                he.face = face
                vi = fverts[i]
                v = m.vertices[vi]
                he.origin = v
                if v.halfedge is None:
                    v.halfedge = he
                he_list.append(he)
            
            for i in range(n):
                hes[i].next = hes[(i + 1) % n]
                hes[i].prev = hes[(i - 1) % n]
            face.halfedge = hes[0]
            m.faces.append(face)
            
            for i in range(n):
                a = fverts[i]
                b = fverts[(i + 1) % n]
                edge_dir_map[(a, b)] = hes[i]

        edge_id = 0
        for (a, b), he in list(edge_dir_map.items()):
            if he.twin is not None: continue
            twin = edge_dir_map.get((b, a))
            he.twin = twin
            if twin: twin.twin = he
            
            key = (min(a, b), max(a, b))
            if key not in m.edge_map:
                m.edge_map[key] = edge_id
                m.edge_halfedges[edge_id] = [he]
                if twin: m.edge_halfedges[edge_id].append(twin)
                he.edge_id = edge_id
                if twin: twin.edge_id = edge_id
                edge_id += 1
            else:
                eid = m.edge_map[key]
                m.edge_halfedges[eid].append(he)
                he.edge_id = eid

        m.halfedges = he_list
        return m

    @staticmethod
    def load_obj_text(text: str) -> "Mesh":
        verts: List[Tuple[float, float, float]] = []
        faces_idx: List[List[int]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if parts[0] == 'v':
                verts.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'f':
                idxs = []
                for p in parts[1:]:
                    v = p.split('/')[0]
                    if v: idxs.append(int(v) - 1)
                faces_idx.append(idxs)
        return Mesh.from_data(verts, faces_idx)

    def to_plotly_mesh(self) -> Tuple[np.ndarray, List[List[int]]]:
        V = np.array([v.pos for v in self.vertices])
        tris: List[List[int]] = []
        for f in self.faces:
            verts = []
            cur = f.halfedge
            start = cur
            while True:
                verts.append(cur.origin.id)
                cur = cur.next
                if cur is start: break
            for i in range(1, len(verts) - 1):
                tris.append([verts[0], verts[i], verts[i + 1]])
        return V, tris

    def compute_vertex_normals(self) -> np.ndarray:
        """Computes smooth normals for each vertex by averaging adjacent face normals."""
        normals = np.zeros((len(self.vertices), 3))
        
        # Iterate over faces to compute face normals and accumulate them
        for f in self.faces:
            he = f.halfedge
            # Get all vertices in face
            vs = []
            cur = he
            start = he
            while True:
                vs.append(cur.origin.pos)
                cur = cur.next
                if cur is start: break
            
            # Newell's method or simple cross product if planar
            if len(vs) >= 3:
                # Simple cross product of first two edges (sufficient for planar faces)
                v0, v1, v2 = vs[0], vs[1], vs[2]
                fn = np.cross(v1 - v0, v2 - v0)
                norm_val = np.linalg.norm(fn)
                if norm_val > 1e-10:
                    fn /= norm_val
                
                # Add to all vertices of this face
                cur = he
                while True:
                    normals[cur.origin.id] += fn
                    cur = cur.next
                    if cur is start: break

        # Normalize the accumulated vertex normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return normals / norms

    def get_edge_list(self) -> List[Tuple[int, int, int]]:
        out = []
        for (a, b), eid in self.edge_map.items():
            out.append((eid, a, b))
        out.sort()
        return out

    def mark_crease_edges(self, crease_edge_ids: Set[int]):
        for he in self.halfedges:
            if he.edge_id in crease_edge_ids:
                he.is_crease = True
            else:
                he.is_crease = False

    def is_triangle_mesh(self) -> bool:
        for f in self.faces:
            cnt = 0
            cur = f.halfedge
            start = cur
            while True:
                cnt += 1
                cur = cur.next
                if cur is start: break
            if cnt != 3: return False
        return True

    def to_obj_text(self) -> str:
        lines: List[str] = []
        for v in self.vertices:
            lines.append(f"v {v.pos[0]} {v.pos[1]} {v.pos[2]}")
        for f in self.faces:
            verts = []
            cur = f.halfedge
            start = cur
            while True:
                verts.append(str(cur.origin.id + 1))
                cur = cur.next
                if cur is start: break
            lines.append("f " + " ".join(verts))
        return "\n".join(lines)

    # --- Generators ---
    @staticmethod
    def create_torus(major_radius=1.0, minor_radius=0.3, major_segments=32, minor_segments=16) -> "Mesh":
        positions = []
        faces = []
        for i in range(major_segments):
            theta = 2 * np.pi * i / major_segments
            for j in range(minor_segments):
                phi = 2 * np.pi * j / minor_segments
                x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
                y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
                z = minor_radius * np.sin(phi)
                positions.append((x, y, z))
        for i in range(major_segments):
            for j in range(minor_segments):
                p1 = i * minor_segments + j
                p2 = ((i + 1) % major_segments) * minor_segments + j
                p3 = ((i + 1) % major_segments) * minor_segments + ((j + 1) % minor_segments)
                p4 = i * minor_segments + ((j + 1) % minor_segments)
                faces.append([p1, p2, p3, p4])
        return Mesh.from_data(positions, faces)

    @staticmethod
    def create_icosphere(radius=1.0, subdivisions=2) -> "Mesh":
        t = (1.0 + np.sqrt(5.0)) / 2.0
        positions = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
        faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
        
        pos_arr = np.array(positions, dtype=float)
        pos_arr /= np.linalg.norm(pos_arr, axis=1)[:, np.newaxis]
        pos_arr *= radius
        positions = pos_arr.tolist()

        for _ in range(subdivisions):
            new_faces = []
            mid_points = {}
            def get_mid(p1, p2):
                key = tuple(sorted((p1, p2)))
                if key in mid_points: return mid_points[key]
                v1, v2 = np.array(positions[p1]), np.array(positions[p2])
                mid = (v1 + v2) / 2.0
                mid = mid / np.linalg.norm(mid) * radius
                positions.append(mid.tolist())
                mid_points[key] = len(positions) - 1
                return mid_points[key]
            for f in faces:
                v1, v2, v3 = f
                a = get_mid(v1, v2)
                b = get_mid(v2, v3)
                c = get_mid(v3, v1)
                new_faces.extend([[v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]])
            faces = new_faces
        return Mesh.from_data(positions, faces)

    @staticmethod
    def create_cube(size=1.0) -> "Mesh":
        s = size / 2.0
        positions = [
            (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
            (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
        ]
        faces = [
            [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
            [1, 5, 6, 2], [2, 6, 7, 3], [4, 0, 3, 7]
        ]
        return Mesh.from_data(positions, faces)

    @staticmethod
    def create_grid(width=2.0, height=2.0, w_segs=4, h_segs=4) -> "Mesh":
        positions = []
        faces = []
        dx = width / w_segs
        dy = height / h_segs
        
        for y in range(h_segs + 1):
            for x in range(w_segs + 1):
                positions.append((x * dx - width/2, y * dy - height/2, 0))
                
        for y in range(h_segs):
            for x in range(w_segs):
                p1 = y * (w_segs + 1) + x
                p2 = p1 + 1
                p3 = (y + 1) * (w_segs + 1) + x + 1
                p4 = (y + 1) * (w_segs + 1) + x
                faces.append([p1, p2, p3, p4])
        return Mesh.from_data(positions, faces)

    @staticmethod
    def create_cylinder(radius=1.0, height=2.0, segments=16) -> "Mesh":
        positions = []
        faces = []
        
        # Top and Bottom centers
        positions.append((0, 0, height/2)) # 0: Top Center
        positions.append((0, 0, -height/2)) # 1: Bottom Center
        
        # Rim vertices
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            positions.append((x, y, height/2)) # Top rim
            positions.append((x, y, -height/2)) # Bottom rim
            
        # Faces
        for i in range(segments):
            # Indices in positions array
            # Top rim: 2 + 2*i
            # Bottom rim: 2 + 2*i + 1
            
            t1 = 2 + 2*i
            b1 = 2 + 2*i + 1
            t2 = 2 + 2*((i+1)%segments)
            b2 = 2 + 2*((i+1)%segments) + 1
            
            # Side Quad
            faces.append([b1, t1, t2, b2])
            
            # Top Triangle
            faces.append([0, t2, t1])
            
            # Bottom Triangle
            faces.append([1, b1, b2])
            
        return Mesh.from_data(positions, faces)

    # --- Subdivision Algorithms (Preserved from your logic) ---
    def subdivide_catmull_clark(self, crease_edge_ids: Optional[Set[int]] = None) -> "Mesh":
        if crease_edge_ids is None: crease_edge_ids = set()
        self.mark_crease_edges(crease_edge_ids)
        
        # Pre-calculate centers
        face_points = {}
        for f in self.faces:
            verts = []
            cur = f.halfedge
            start = cur
            while True:
                verts.append(cur.origin.pos)
                cur = cur.next
                if cur is start: break
            face_points[f.id] = np.mean(np.array(verts), axis=0)

        edge_points = {}
        for (a, b), eid in self.edge_map.items():
            hes = self.edge_halfedges.get(eid, [])
            pa, pb = self.vertices[a].pos, self.vertices[b].pos
            if eid in crease_edge_ids:
                edge_points[eid] = 0.5 * (pa + pb)
            else:
                fps = [face_points[he.face.id] for he in hes if he.face]
                if not fps: edge_points[eid] = 0.5 * (pa + pb)
                else: edge_points[eid] = (pa + pb + sum(fps)) / (2 + len(fps))

        new_vertex_pos = []
        for v in self.vertices:
            he = v.halfedge
            if not he:
                new_vertex_pos.append(v.pos)
                continue
            
            # Gather neighborhood
            faces, mids, crease_cnt = [], [], 0
            cur = he
            start = he
            valence = 0
            while True:
                valence += 1
                if cur.face: faces.append(face_points[cur.face.id])
                mids.append(0.5*(cur.origin.pos + cur.next.origin.pos))
                if cur.edge_id in crease_edge_ids: crease_cnt += 1
                cur = cur.twin.next if (cur.twin and cur.twin.next) else cur.next
                if cur is start or cur is None: break
            
            if crease_cnt >= 2:
                # Feature vertex logic
                c_mids = []
                cur = he; start = he
                while True:
                    if cur.edge_id in crease_edge_ids:
                        c_mids.append(0.5*(cur.origin.pos + cur.next.origin.pos))
                    cur = cur.twin.next if (cur.twin and cur.twin.next) else cur.next
                    if cur is start or cur is None: break
                if len(c_mids) >= 2:
                    new_vertex_pos.append(v.pos * 0.75 + sum(c_mids[:2]) * 0.125)
                else: new_vertex_pos.append(v.pos)
            else:
                F = np.mean(faces, axis=0) if faces else v.pos
                R = np.mean(mids, axis=0) if mids else v.pos
                new_vertex_pos.append((F + 2*R + (valence-3)*v.pos)/valence)

        # Assemble new mesh
        new_pos = new_vertex_pos + [face_points[k] for k in sorted(face_points)]
        face_base = len(new_vertex_pos)
        edge_base = len(new_pos)
        sorted_edges = sorted(self.edge_map.values())
        new_pos += [edge_points[eid] for eid in sorted_edges]
        
        eid_map = {eid: edge_base + i for i, eid in enumerate(sorted_edges)}
        fid_map = {fid: face_base + i for i, fid in enumerate(sorted(face_points))}
        
        new_faces = []
        for f in self.faces:
            verts = []
            cur = f.halfedge
            start = cur
            while True:
                verts.append(cur.origin.id)
                cur = cur.next
                if cur is start: break
            
            k = len(verts)
            f_idx = fid_map[f.id]
            for i in range(k):
                vi = verts[i]
                vj = verts[(i+1)%k]
                vp = verts[(i-1)%k]
                eid = self.edge_map[(min(vi, vj), max(vi, vj))]
                eid_p = self.edge_map[(min(vp, vi), max(vp, vi))]
                new_faces.append([vi, eid_map[eid], f_idx, eid_map[eid_p]])
                
        return Mesh.from_data([tuple(p) for p in new_pos], new_faces)

    def subdivide_loop(self, crease_edge_ids: Optional[Set[int]] = None) -> "Mesh":
        if crease_edge_ids is None: crease_edge_ids = set()
        if not self.is_triangle_mesh(): raise ValueError("Loop requires triangle mesh")
        self.mark_crease_edges(crease_edge_ids)
        
        V_arr = np.array([v.pos for v in self.vertices])
        new_pos = list(V_arr)
        edge_idx_map = {}
        
        for (a, b), eid in self.edge_map.items():
            pa, pb = self.vertices[a].pos, self.vertices[b].pos
            hes = self.edge_halfedges.get(eid, [])
            if eid in crease_edge_ids or len(hes) < 2:
                new_pos.append(0.5 * (pa + pb))
            else:
                opp = []
                for he in hes:
                    if not he.face: continue
                    cur = he.next
                    while cur.origin.id == a or cur.origin.id == b: cur = cur.next
                    opp.append(cur.origin.pos)
                if len(opp) >= 2:
                    new_pos.append(0.375*(pa+pb) + 0.125*(opp[0]+opp[1]))
                else: new_pos.append(0.5*(pa+pb))
            edge_idx_map[eid] = len(new_pos) - 1
            
        # Update original vertices
        updated_orig = []
        for v in self.vertices:
            he = v.halfedge
            neighbors = []
            cur = he; start = he
            crease_neighbors = []
            
            while True:
                neighbors.append(cur.next.origin.pos)
                if cur.edge_id in crease_edge_ids: crease_neighbors.append(cur.next.origin.pos)
                if cur.prev.edge_id in crease_edge_ids: crease_neighbors.append(cur.prev.origin.pos) # Check incoming too approx
                
                cur = cur.twin.next if (cur.twin and cur.twin.next) else cur.next
                if cur is start or cur is None: break
                
            n = len(neighbors)
            if n == 0: updated_orig.append(v.pos); continue

            # Refined crease logic (simplified)
            is_crease_vert = False
            c_neigh = []
            cur = he; start=he
            while True:
                if cur.edge_id in crease_edge_ids: c_neigh.append(cur.next.origin.pos)
                cur = cur.twin.next if (cur.twin and cur.twin.next) else cur.next
                if cur is start or cur is None: break
            
            if len(c_neigh) >= 2:
                updated_orig.append(0.75*v.pos + 0.125*(c_neigh[0]+c_neigh[1]))
            else:
                beta = (0.625 - (0.375 + 0.25 * np.cos(2*np.pi/n))**2) / n
                updated_orig.append((1 - n*beta)*v.pos + beta*sum(neighbors))

        for i, pos in enumerate(updated_orig):
            new_pos[i] = pos
            
        new_faces = []
        for f in self.faces:
            cur = f.halfedge
            verts = [cur.origin.id, cur.next.origin.id, cur.next.next.origin.id]
            e0 = edge_idx_map[self.edge_map[tuple(sorted((verts[0], verts[1])))]]
            e1 = edge_idx_map[self.edge_map[tuple(sorted((verts[1], verts[2])))]]
            e2 = edge_idx_map[self.edge_map[tuple(sorted((verts[2], verts[0])))]]
            new_faces.extend([
                [verts[0], e0, e2], [verts[1], e1, e0], 
                [verts[2], e2, e1], [e0, e1, e2]
            ])
            
        return Mesh.from_data([tuple(p) for p in new_pos], new_faces)

    def subdivide_sqrt3(self) -> "Mesh":
        if not self.is_triangle_mesh(): raise ValueError("Sqrt3 requires triangle mesh")
        
        # 1. Calculate Face Centers
        face_centers = {}
        for f in self.faces:
            cur = f.halfedge
            vs = [cur.origin.pos, cur.next.origin.pos, cur.next.next.origin.pos]
            face_centers[f.id] = np.mean(vs, axis=0)
            
        # 2. Relax Original Vertices
        new_pos_map = {} # old_v_id -> new_pos
        new_pos_list = []
        
        # Helper to add vertex and get index
        def add_vert(p):
            new_pos_list.append(p)
            return len(new_pos_list) - 1

        for v in self.vertices:
            he = v.halfedge
            neigh = []
            cur = he; start = he
            while True:
                neigh.append(cur.next.origin.pos)
                cur = cur.twin.next if (cur.twin and cur.twin.next) else cur.next
                if cur is start or cur is None: break
            
            if not neigh: 
                new_pos_map[v.id] = add_vert(v.pos)
            else:
                n = len(neigh)
                alpha = (4 - 2*np.cos(2*np.pi/n))/9
                new_pos_map[v.id] = add_vert((1-alpha)*v.pos + alpha*np.mean(neigh, axis=0))
                
        # Add face centers to vertex list
        fc_idx_map = {}
        for fid, center in face_centers.items():
            fc_idx_map[fid] = add_vert(center)
            
        # 3. Create New Faces (Flip Edges)
        new_faces = []
        processed_edges = set()
        
        for f in self.faces:
            cur = f.halfedge; start = cur
            while True:
                u = cur.origin.id
                v = cur.next.origin.id
                edge_key = tuple(sorted((u, v)))
                
                if edge_key not in processed_edges:
                    processed_edges.add(edge_key)
                    twin = cur.twin
                    if twin and twin.face:
                        # Internal edge shared by f (cur.face) and twin.face
                        # cur is u->v on face f
                        # twin is v->u on face twin.face
                        
                        # Vertices
                        u_new = new_pos_map[u]
                        v_new = new_pos_map[v]
                        c_f = fc_idx_map[f.id]
                        c_twin = fc_idx_map[twin.face.id]
                        
                        # Two new triangles replacing the edge
                        # 1. (u, C_twin, C_f)
                        new_faces.append([u_new, c_twin, c_f])
                        # 2. (v, C_f, C_twin)
                        new_faces.append([v_new, c_f, c_twin])
                    else:
                        # Boundary edge: No flip, just split?
                        # Sqrt3 usually requires special boundary handling.
                        # Simple fallback: Keep original edge, connect to center.
                        # Triangle (u, v, C_f)
                        u_new = new_pos_map[u]
                        v_new = new_pos_map[v]
                        c_f = fc_idx_map[f.id]
                        new_faces.append([u_new, v_new, c_f])
                
                cur = cur.next
                if cur is start: break
            
        return Mesh.from_data([tuple(p) for p in new_pos_list], new_faces)

    def subdivide_doo_sabin(self) -> "Mesh":
        new_pos = []
        new_faces = []
        fv_map = {} # (face_id, old_vertex_id) -> new_vertex_index

        # 1. Create F-faces (inside old faces) and generate new vertices
        for f in self.faces:
            verts = []
            v_ids = []
            cur = f.halfedge; start = cur
            while True:
                verts.append(cur.origin.pos)
                v_ids.append(cur.origin.id)
                cur = cur.next
                if cur is start: break
            
            n = len(verts)
            f_indices = []
            
            for i in range(n):
                # Doo-Sabin point: Weighted average
                p = np.zeros(3)
                for j in range(n):
                    weight = (3 + 2 * np.cos(2 * np.pi * (i - j) / n)) / (4 * n)
                    p += verts[j] * weight
                
                new_pos.append(p)
                new_idx = len(new_pos) - 1
                f_indices.append(new_idx)
                fv_map[(f.id, v_ids[i])] = new_idx
            
            new_faces.append(f_indices)

        # 2. Create E-faces (Edges)
        processed_edges = set()
        for f in self.faces:
            cur = f.halfedge; start = cur
            while True:
                u = cur.origin.id
                v = cur.next.origin.id
                edge_key = tuple(sorted((u, v)))
                
                if edge_key not in processed_edges:
                    processed_edges.add(edge_key)
                    twin = cur.twin
                    if twin and twin.face:
                        f1 = f.id
                        f2 = twin.face.id
                        
                        # Vertices for the quad
                        p1 = fv_map[(f1, u)]
                        p2 = fv_map[(f1, v)]
                        p3 = fv_map[(f2, v)]
                        p4 = fv_map[(f2, u)]
                        
                        new_faces.append([p1, p4, p3, p2])
                
                cur = cur.next
                if cur is start: break

        # 3. Create V-faces (Vertices)
        for v in self.vertices:
            he = v.halfedge
            if not he: continue
            
            v_face_indices = []
            curr_he = he
            start_he = he
            is_boundary = False
            
            # Circulate around vertex (CCW)
            while True:
                if curr_he.face:
                    v_face_indices.append(fv_map[(curr_he.face.id, v.id)])
                
                if not curr_he.twin:
                    is_boundary = True
                    break
                curr_he = curr_he.twin.next
                if curr_he is start_he: break
            
            if not is_boundary and len(v_face_indices) > 2:
                new_faces.append(v_face_indices)

        return Mesh.from_data([tuple(p) for p in new_pos], new_faces)