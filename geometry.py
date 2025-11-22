"""
geometry.py

Provides a simple half-edge mesh representation, OBJ loader, and subdivision
implementations (Catmull-Clark and Loop). The goal is clarity and correctness
for educational purposes rather than maximum performance.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
import numpy as np


class Vertex:
    def __init__(self, pos: np.ndarray, id: int):
        self.pos = np.asarray(pos, dtype=float)
        self.halfedge: Optional[HalfEdge] = None
        self.id = id


class HalfEdge:
    def __init__(self):
        self.origin: Optional[Vertex] = None
        self.twin: Optional[HalfEdge] = None
        self.next: Optional[HalfEdge] = None
        self.prev: Optional[HalfEdge] = None
        self.face: Optional[Face] = None
        self.edge_id: Optional[int] = None
        self.is_crease: bool = False


class Face:
    def __init__(self, id: int):
        self.halfedge: Optional[HalfEdge] = None
        self.id = id


class Mesh:
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.halfedges: List[HalfEdge] = []
        self.faces: List[Face] = []
        # map undirected edge (min,max) -> edge_id
        self.edge_map: Dict[Tuple[int, int], int] = {}
        # map edge_id -> tuple of two halfedges (may be one for boundary)
        self.edge_halfedges: Dict[int, List[HalfEdge]] = {}

    @staticmethod
    def from_data(positions: List[Tuple[float, float, float]], faces_idx: List[List[int]]) -> "Mesh":
        m = Mesh()
        for i, p in enumerate(positions):
            m.vertices.append(Vertex(np.array(p, dtype=float), i))

        # create faces and halfedges
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
            # link next/prev
            for i in range(n):
                hes[i].next = hes[(i + 1) % n]
                hes[i].prev = hes[(i - 1) % n]
            face.halfedge = hes[0]
            m.faces.append(face)
            # map directed edges to halfedges for twin assignment
            for i in range(n):
                a = fverts[i]
                b = fverts[(i + 1) % n]
                edge_dir_map[(a, b)] = hes[i]

        # assign twin and edge ids
        edge_id = 0
        for (a, b), he in list(edge_dir_map.items()):
            if he.twin is not None:
                continue
            twin = edge_dir_map.get((b, a))
            he.twin = twin
            if twin:
                twin.twin = he
            key = (min(a, b), max(a, b))
            if key not in m.edge_map:
                m.edge_map[key] = edge_id
                m.edge_halfedges[edge_id] = [he]
                if twin:
                    m.edge_halfedges[edge_id].append(twin)
                he.edge_id = edge_id
                if twin:
                    twin.edge_id = edge_id
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
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                x, y, z = map(float, parts[1:4])
                verts.append((x, y, z))
            elif parts[0] == 'f':
                idxs = []
                for p in parts[1:]:
                    # support v, v//vn, v/vt/vn
                    v = p.split('/')[0]
                    if v == '':
                        continue
                    i = int(v) - 1
                    idxs.append(i)
                faces_idx.append(idxs)

        return Mesh.from_data(verts, faces_idx)

    def to_plotly_mesh(self) -> Tuple[np.ndarray, List[List[int]]]:
        # Return vertex positions and triangulated faces (list of triangles)
        V = np.array([v.pos for v in self.vertices])
        tris: List[List[int]] = []
        for f in self.faces:
            # collect vertex indices of face
            he = f.halfedge
            verts = []
            start = he
            cur = he
            while True:
                verts.append(cur.origin.id)
                cur = cur.next
                if cur is start:
                    break
            # triangulate fan
            for i in range(1, len(verts) - 1):
                tris.append([verts[0], verts[i], verts[i + 1]])
        return V, tris

    def get_edge_list(self) -> List[Tuple[int, int, int]]:
        # returns list of (edge_id, v0, v1)
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

    def subdivide_catmull_clark(self, crease_edge_ids: Optional[Set[int]] = None) -> "Mesh":
        if crease_edge_ids is None:
            crease_edge_ids = set()
        self.mark_crease_edges(crease_edge_ids)

        V = np.array([v.pos for v in self.vertices])
        nV = len(V)

        # compute face points
        face_points: Dict[int, np.ndarray] = {}
        for f in self.faces:
            positions = []
            he = f.halfedge
            start = he
            cur = he
            while True:
                positions.append(cur.origin.pos)
                cur = cur.next
                if cur is start:
                    break
            face_points[f.id] = np.mean(np.array(positions), axis=0)

        # compute edge points
        edge_points: Dict[int, np.ndarray] = {}
        for (a, b), eid in list(self.edge_map.items()):
            # find halfedges for this edge
            hes = self.edge_halfedges.get(eid, [])
            pa = self.vertices[a].pos
            pb = self.vertices[b].pos
            if eid in crease_edge_ids:
                # crease: simple midpoint
                edge_points[eid] = 0.5 * (pa + pb)
            else:
                # average of endpoints and adjacent face-points
                fps = []
                for he in hes:
                    if he and he.face is not None:
                        fps.append(face_points[he.face.id])
                if len(fps) == 0:
                    edge_points[eid] = 0.5 * (pa + pb)
                else:
                    edge_points[eid] = (pa + pb + sum(fps)) / (2 + len(fps))

        # compute new vertex positions
        new_vertex_pos: List[np.ndarray] = []
        for vi, v in enumerate(self.vertices):
            # collect adjacent face points and midpoints of edges
            faces = []
            edge_midpoints = []
            he = v.halfedge
            if he is None:
                new_vertex_pos.append(v.pos.copy())
                continue
            start = he
            cur = he
            valence = 0
            crease_count = 0
            while True:
                valence += 1
                if cur.face is not None:
                    faces.append(face_points[cur.face.id])
                eid = cur.edge_id
                a, b = min(cur.origin.id, cur.next.origin.id), max(cur.origin.id, cur.next.origin.id)
                edge_midpoints.append(0.5 * (cur.origin.pos + cur.next.origin.pos))
                if eid in crease_edge_ids:
                    crease_count += 1
                cur = cur.twin.next if (cur.twin and cur.twin.next) else (cur.next)
                if cur is start or cur is None:
                    break

            F = np.mean(np.array(faces), axis=0) if len(faces) > 0 else v.pos
            R = np.mean(np.array(edge_midpoints), axis=0) if len(edge_midpoints) > 0 else v.pos
            n = valence
            if crease_count >= 2:
                # feature vertex (on crease) — keep closer to original
                # simple rule: average of original and two crease neighbor midpoints
                # find crease neighbor midpoints
                crease_midpoints = []
                cur = he
                start = he
                while True:
                    if cur.edge_id in crease_edge_ids:
                        crease_midpoints.append(0.5 * (cur.origin.pos + cur.next.origin.pos))
                    cur = cur.twin.next if (cur.twin and cur.twin.next) else (cur.next)
                    if cur is start or cur is None:
                        break
                if len(crease_midpoints) >= 2:
                    newv = (v.pos * 0.75 + sum(crease_midpoints[:2]) * 0.25 / 2)
                else:
                    newv = v.pos.copy()
            else:
                newv = (F + 2 * R + (n - 3) * v.pos) / n
            new_vertex_pos.append(newv)

        # Build list of new positions: old vertex new positions, face points, edge points
        vertex_base = 0
        old_base = vertex_base
        new_positions: List[np.ndarray] = []
        for p in new_vertex_pos:
            new_positions.append(p)
        face_base = len(new_positions)
        for fi in sorted(face_points.keys()):
            new_positions.append(face_points[fi])
        edge_base = len(new_positions)
        # sort edge ids to have consistent ordering
        sorted_edges = sorted(list(set(self.edge_map.values())))
        eid_to_new_idx: Dict[int, int] = {}
        for i, eid in enumerate(sorted_edges):
            new_positions.append(edge_points[eid])
            eid_to_new_idx[eid] = edge_base + i

        # face id mapping to index
        fid_to_idx = {fid: face_base + i for i, fid in enumerate(sorted(face_points.keys()))}

        # create new faces: for each original face with vertices v0..vk-1, create k quads
        new_faces: List[List[int]] = []
        for f in self.faces:
            # collect vertex indices
            he = f.halfedge
            verts = []
            cur = he
            start = he
            while True:
                verts.append(cur.origin.id)
                cur = cur.next
                if cur is start:
                    break
            k = len(verts)
            fidx = fid_to_idx[f.id]
            for i in range(k):
                vi = verts[i]
                vj = verts[(i + 1) % k]
                # edge id between vi and vj
                key = (min(vi, vj), max(vi, vj))
                eid = self.edge_map[key]
                eid_prev_key = (min(verts[(i - 1) % k], vi), max(verts[(i - 1) % k], vi))
                eid_prev = self.edge_map[eid_prev_key]
                # indices in new_positions
                vi_new = old_base + vi
                ei_new = eid_to_new_idx[eid]
                ep_new = fidx
                eprev_new = eid_to_new_idx[eid_prev]
                # quad ordering: vi_new, ei_new, ep_new, eprev_new
                new_faces.append([vi_new, ei_new, ep_new, eprev_new])

        # convert new_positions to simple lists
        pos_list = [tuple(p.tolist()) for p in new_positions]
        return Mesh.from_data(pos_list, new_faces)

    def is_triangle_mesh(self) -> bool:
        for f in self.faces:
            he = f.halfedge
            cnt = 0
            cur = he
            while True:
                cnt += 1
                cur = cur.next
                if cur is he:
                    break
            if cnt != 3:
                return False
        return True

    def subdivide_loop(self, crease_edge_ids: Optional[Set[int]] = None) -> "Mesh":
        if crease_edge_ids is None:
            crease_edge_ids = set()
        if not self.is_triangle_mesh():
            raise ValueError("Loop subdivision requires a triangle-only mesh")
        self.mark_crease_edges(crease_edge_ids)

        V = np.array([v.pos for v in self.vertices])
        nV = len(V)

        # compute new edge points
        edge_point_idx: Dict[int, int] = {}
        new_positions: List[np.ndarray] = [p.copy() for p in V]
        for (a, b), eid in list(self.edge_map.items()):
            hes = self.edge_halfedges.get(eid, [])
            pa = self.vertices[a].pos
            pb = self.vertices[b].pos
            if eid in crease_edge_ids or len(hes) < 2:
                ep = 0.5 * (pa + pb)
            else:
                # two adjacent faces: get the two opposite vertices
                opp = []
                for he in hes:
                    if he and he.face is not None:
                        # find the vertex opposite to edge a-b in this triangle
                        cur = he
                        # he is (a->b) or (b->a) arbitrary; find third vertex
                        # vertices of face
                        face_verts = []
                        start = cur
                        tmp = cur
                        while True:
                            face_verts.append(tmp.origin.id)
                            tmp = tmp.next
                            if tmp is start:
                                break
                        for vid in face_verts:
                            if vid != a and vid != b:
                                opp.append(self.vertices[vid].pos)
                if len(opp) >= 2:
                    ep = (3.0/8.0)*(pa + pb) + (1.0/8.0)*(opp[0] + opp[1])
                elif len(opp) == 1:
                    ep = 0.5*(pa + pb)
                else:
                    ep = 0.5*(pa + pb)
            edge_point_idx[eid] = len(new_positions)
            new_positions.append(ep)

        # update old vertex positions
        for vi, v in enumerate(self.vertices):
            he = v.halfedge
            # collect neighbor vertices
            neigh = []
            start = he
            cur = he
            while True:
                neigh.append(cur.next.origin.pos)
                cur = cur.twin.next if (cur.twin and cur.twin.next) else cur.next
                if cur is start or cur is None:
                    break
            n = len(neigh)
            if n == 0:
                continue
            if any((he.edge_id in crease_edge_ids) for he in [he, he.prev]):
                # boundary/crease handling: simplest: average with neighbors on crease
                # find crease neighbors
                crease_neigh = []
                cur = he
                while True:
                    if cur.edge_id in crease_edge_ids:
                        crease_neigh.append(cur.next.origin.pos)
                    cur = cur.twin.next if (cur.twin and cur.twin.next) else cur.next
                    if cur is start or cur is None or len(crease_neigh) >= 2:
                        break
                if len(crease_neigh) >= 2:
                    newp = 0.75 * v.pos + 0.125 * (crease_neigh[0] + crease_neigh[1])
                else:
                    newp = v.pos
            else:
                beta = (5.0/8.0 - (3.0/8.0 + 0.25 * np.cos(2.0 * np.pi / n))**2)/n
                neigh_sum = sum(neigh)
                newp = (1 - n * beta) * v.pos + beta * neigh_sum
            new_positions[vi] = newp

        # create new faces: each original triangle (a,b,c) -> 4 triangles
        new_faces: List[List[int]] = []
        for f in self.faces:
            he = f.halfedge
            verts = []
            cur = he
            start = he
            while True:
                verts.append(cur.origin.id)
                cur = cur.next
                if cur is start:
                    break
            a, b, c = verts
            key_ab = (min(a, b), max(a, b))
            key_bc = (min(b, c), max(b, c))
            key_ca = (min(c, a), max(c, a))
            eab = edge_point_idx[self.edge_map[key_ab]]
            ebc = edge_point_idx[self.edge_map[key_bc]]
            eca = edge_point_idx[self.edge_map[key_ca]]
            new_faces.append([a, eab, eca])
            new_faces.append([b, ebc, eab])
            new_faces.append([c, eca, ebc])
            new_faces.append([eab, ebc, eca])

        pos_list = [tuple(p.tolist()) for p in new_positions]
        return Mesh.from_data(pos_list, new_faces)

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
        positions = [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)
        ]
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        
        # Normalize vertices to be on a sphere
        positions = np.array(positions, dtype=float)
        positions /= np.linalg.norm(positions, axis=1)[:, np.newaxis]
        positions *= radius
        
        # Subdivide faces
        for _ in range(subdivisions):
            new_faces = []
            mid_points = {}
            
            for face in faces:
                v1, v2, v3 = face
                
                def get_mid_point(p1, p2):
                    nonlocal positions
                    key = tuple(sorted((p1, p2)))
                    if key in mid_points:
                        return mid_points[key]
                    
                    mid = (positions[p1] + positions[p2]) / 2.0
                    mid /= np.linalg.norm(mid)
                    mid *= radius
                    
                    positions = np.vstack([positions, mid])
                    mid_points[key] = len(positions) - 1
                    return mid_points[key]

                m1 = get_mid_point(v1, v2)
                m2 = get_mid_point(v2, v3)
                m3 = get_mid_point(v3, v1)
                
                new_faces.extend([[v1, m1, m3], [v2, m2, m1], [v3, m3, m2], [m1, m2, m3]])
            faces = new_faces
            
        return Mesh.from_data(positions.tolist(), faces)

    def to_obj_text(self) -> str:
        lines: List[str] = []
        for v in self.vertices:
            x, y, z = v.pos
            lines.append(f"v {x} {y} {z}")
        for f in self.faces:
            he = f.halfedge
            verts = []
            start = he
            cur = he
            while True:
                verts.append(cur.origin.id + 1)
                cur = cur.next
                if cur is start:
                    break
            lines.append("f " + " ".join(str(vi) for vi in verts))
        return "\n".join(lines) + "\n"

    def subdivide_sqrt3(self) -> "Mesh":
        """
        A simplified Sqrt(3)-like subdivision for triangle meshes.
        This implementation inserts a face centroid for each triangle,
        creates barycentric triangles, and applies a simple smoothing
        to original vertices.
        """
        if not self.is_triangle_mesh():
            raise ValueError("Sqrt(3) subdivision requires a triangle-only mesh")

        V = np.array([v.pos for v in self.vertices])
        face_centers: Dict[int, np.ndarray] = {}
        for f in self.faces:
            he = f.halfedge
            verts = []
            cur = he
            start = he
            while True:
                verts.append(cur.origin.pos)
                cur = cur.next
                if cur is start:
                    break
            face_centers[f.id] = np.mean(np.array(verts), axis=0)

        # smoothing original vertices: average of neighbors
        new_orig = []
        for v in self.vertices:
            he = v.halfedge
            neigh = []
            start = he
            cur = he
            while True:
                neigh.append(cur.next.origin.pos)
                cur = cur.twin.next if (cur.twin and cur.twin.next) else cur.next
                if cur is start or cur is None:
                    break
            if len(neigh) == 0:
                new_orig.append(v.pos.copy())
            else:
                neigh_mean = np.mean(np.array(neigh), axis=0)
                alpha = 0.33
                new_orig.append((1 - alpha) * v.pos + alpha * neigh_mean)

        # build new positions: smoothed old vertices, then face centers
        new_positions: List[np.ndarray] = [p.copy() for p in new_orig]
        fid_to_idx = {}
        for fid in sorted(face_centers.keys()):
            fid_to_idx[fid] = len(new_positions)
            new_positions.append(face_centers[fid])

        new_faces: List[List[int]] = []
        for f in self.faces:
            he = f.halfedge
            verts = []
            cur = he
            start = he
            while True:
                verts.append(cur.origin.id)
                cur = cur.next
                if cur is start:
                    break
            cid = fid_to_idx[f.id]
            a, b, c = verts
            new_faces.append([cid, a, b])
            new_faces.append([cid, b, c])
            new_faces.append([cid, c, a])

        pos_list = [tuple(p.tolist()) for p in new_positions]
        return Mesh.from_data(pos_list, new_faces)

    def subdivide_doo_sabin(self) -> "Mesh":
        new_positions = []
        new_faces = []
        
        # A map from each original face to the indices of the new vertices that will form the new face
        face_to_new_face_indices = {}

        # Create new vertices for each original face
        for face in self.faces:
            original_vertices = []
            he = face.halfedge
            while True:
                original_vertices.append(he.origin)
                he = he.next
                if he == face.halfedge:
                    break
            
            new_face_indices = []
            for i in range(len(original_vertices)):
                new_pos = np.zeros(3)
                for j in range(len(original_vertices)):
                    weight = (1/len(original_vertices)) * (1 + np.cos(2 * np.pi * (i - j) / len(original_vertices))) if i == j else (1/len(original_vertices)) * (1 + np.cos(2 * np.pi * (i - j) / len(original_vertices)))
                    new_pos += original_vertices[j].pos * weight
                
                new_positions.append(new_pos)
                new_face_indices.append(len(new_positions) - 1)
            
            new_faces.append(new_face_indices)
            face_to_new_face_indices[face.id] = new_face_indices

        # Create faces for each original vertex
        for vertex in self.vertices:
            he = vertex.halfedge
            if not he:
                continue

            # Collect all faces around the vertex
            faces_around = []
            temp_he = he
            while True:
                faces_around.append(temp_he.face)
                if not temp_he.twin:
                    break
                temp_he = temp_he.twin.next
                if temp_he == he:
                    break
            
            new_vertex_face_indices = []
            for face in faces_around:
                # Find the index of the new vertex in the new face that corresponds to the original vertex
                original_vertices = []
                temp_he_2 = face.halfedge
                while True:
                    original_vertices.append(temp_he_2.origin)
                    temp_he_2 = temp_he_2.next
                    if temp_he_2 == face.halfedge:
                        break
                
                for i in range(len(original_vertices)):
                    if original_vertices[i] == vertex:
                        new_vertex_face_indices.append(face_to_new_face_indices[face.id][i])
                        break
            
            if len(new_vertex_face_indices) > 2:
                new_faces.append(new_vertex_face_indices)

        # Create faces for each original edge
        seen_edges = set()
        for he in self.halfedges:
            if he.twin and he.edge_id is not None and he.edge_id not in seen_edges:
                seen_edges.add(he.edge_id)
                face1 = he.face
                face2 = he.twin.face
                
                original_vertices1 = []
                temp_he_1 = face1.halfedge
                while True:
                    original_vertices1.append(temp_he_1.origin)
                    temp_he_1 = temp_he_1.next
                    if temp_he_1 == face1.halfedge:
                        break
                
                original_vertices2 = []
                temp_he_2 = face2.halfedge
                while True:
                    original_vertices2.append(temp_he_2.origin)
                    temp_he_2 = temp_he_2.next
                    if temp_he_2 == face2.halfedge:
                        break

                v1 = he.origin
                v2 = he.twin.origin
                
                idx1 = -1
                for i in range(len(original_vertices1)):
                    if original_vertices1[i] == v1:
                        idx1 = face_to_new_face_indices[face1.id][i]
                        break
                
                idx2 = -1
                for i in range(len(original_vertices1)):
                    if original_vertices1[i] == v2:
                        idx2 = face_to_new_face_indices[face1.id][i]
                        break

                idx3 = -1
                for i in range(len(original_vertices2)):
                    if original_vertices2[i] == v2:
                        idx3 = face_to_new_face_indices[face2.id][i]
                        break

                idx4 = -1
                for i in range(len(original_vertices2)):
                    if original_vertices2[i] == v1:
                        idx4 = face_to_new_face_indices[face2.id][i]
                        break
                
                if all(i != -1 for i in [idx1, idx2, idx3, idx4]):
                    new_faces.append([idx1, idx2, idx3, idx4])

        pos_list = [tuple(p) for p in new_positions]
        return Mesh.from_data(pos_list, new_faces)
