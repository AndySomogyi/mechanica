/*
 * MeshIO.cpp
 *
 *  Created on: Apr 5, 2018
 *      Author: andy
 */

#include <MxEdge.h>
#include "MeshIO.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>



#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

#include <unordered_map>



// AssImp
// Scene : Mesh*
// Mesh { vertices*, normals*, texCoord*, faces*}

/**
 *
 * * enumerate all the vertices, identify cell relationships
 * for each mesh
 *     for each vertex
 *         look in cache for vertex, if so, add mesh ptr to it. if not, insert
 *         vertex into dictionary, we share vertices.
 *
 * if every vertex has at most four incident cells, we're OK, can build a mesh.
 * create a real mesh vertex in the mesh, relate it to the cached vertices.
 *
 * for each mesh
 *     mesh faces should have the correct winding from assimp, normal facing out.
 *     create a new cell
 *
 *     for each face in mesh
 *        find an existing triangle in mesh, if found, make sure the empty partial
 *        triangle is correctly aligned for the current mesh.
 *        if not found, create a new triangle in mesh.
 *
 *     after all faces processed, go over all partial triangles in cell, connect
 *     neighbor pointers.
 *
 */





struct AiVecHasher
{
    std::size_t operator()(const aiVector3D &vec) const
    {
        using std::size_t;
        using std::hash;
        using std::string;

        // assume most vertices are around the same distance, shift x and
        // y to make hash func more unique.

        return ((hash<ai_real>()(100. * vec[0])
                ^ (hash<ai_real>()(10. * vec[1])
                        ^ (hash<ai_real>()(vec[2])))));
    }
};

struct ImpVertex;

struct ImpEdge {

    ImpEdge(ImpVertex *v1, ImpVertex *v2) {
        verts[0] = v1;
        verts[1] = v2;
    }

    const bool matches(const ImpVertex *v0, const ImpVertex *v1) const {
        return (verts[0] == v0 && verts[1] == v1) || (verts[0] == v1 && verts[1] == v0);
    }

    ImpVertex *verts[2];


    // the associated skeletal edge in the mechanica mesh,
    // created after we've verified the imported mesh.
    EdgePtr edge = nullptr;
};

typedef std::vector<ImpEdge> EdgeVector;



struct ImpVertex {
    ImpVertex(int id, const aiVector3D& vec) : id{id}, pos{vec} {};
    
    int id;
    
    // the mesh (cells) that this vertex belongs to
    std::vector<const aiMesh*> meshes;
    
    // the id of this vertex in the corresponding mesh above.
    std::vector<int> ids;

    bool containsMesh(const aiMesh* msh) {
        return std::find(meshes.begin(), meshes.end(), msh) != meshes.end();
    }

    
    bool addMesh(const aiMesh *mesh, int id) {
        
        // vertex is already attached to 4 meshes, can't add another.
        if(meshes.size() >= 4) {
            // error, mesh has vertex with more than 4 cells
            std::cout << "error, vertex has more than 4 cells" << std::endl;
            std::cout << toString();
            return false;
        }
        meshes.push_back(mesh);
        ids.push_back(id);
        return true;
    }

    // the Mx vertex that we create in our mx mesh.
    MxVertex *vert = nullptr;

    aiVector3D pos;

    // vertex can have up to 4 edges
    std::vector<const ImpEdge*> edges;

    bool hasEdge(const ImpEdge *edge) const {
        return std::find(edges.begin(), edges.end(), edge) != edges.end();
    }

    bool addEdge(const ImpEdge *edge) {
        if(edges.size() < 4) {
            edges.push_back(edge);
            return true;
        }
        return false;
    }
    
    std::string toString() const {
        std::stringstream ss;
        
        ss << "vertex:{ id: " << id << ", pos:{" << pos[0] << ", " << pos[1] << ", " << pos[2] << "}, meshes:{";
        for(int i = 0; i < meshes.size(); ++i) {
            ss << "{\"" << meshes[i]->mName.C_Str() << "\", ";
            ss << ids[i] << "}, ";
        }
        ss << "}";
        return ss.str();
    }
};

struct VectorMap {

    VectorMap(double fudge) : fudgeDistance{fudge} {};

    std::vector<ImpVertex> vertices;

    const double fudgeDistance;

    ImpVertex *findVertex(const aiVector3D &pos) {
        for(ImpVertex &v : vertices) {
            aiVector3D dist = v.pos - pos;
            if(dist.Length() <= fudgeDistance) {
                return &v;
            }
        }
        return nullptr;
    }

    ImpVertex *createVertex(const aiVector3D &pos, int id) {
        assert(findVertex(pos) == nullptr);
        vertices.push_back(ImpVertex(id, pos));
        return &vertices.back();
    }

    typedef std::vector<ImpVertex>::iterator iterator;
    typedef std::vector<ImpVertex>::const_iterator const_iterator;

    iterator begin() {
        return vertices.begin();
    }

    iterator end() {
        return vertices.end();
    }

    const_iterator end() const {
        return vertices.end();
    }

    const_iterator find(const aiVector3D &pos) const {

        for(const_iterator v = vertices.begin(); v != vertices.end(); ++v) {
            aiVector3D dist = v->pos - pos;
            if(dist.Length() <= fudgeDistance) {
                return v;
            }
        }
        return vertices.end();
    }


};






/**
 * Looks up the global vertices from a given aiFace.
 *
 * Note, the isFace has indices into the vertices of its containing aiMesh, so we
 * have to map those local vertices to the global ones.
 */
static std::vector<VertexPtr> verticesForFace(const VectorMap &vecMap, const aiMesh* cell,
        const aiFace *face);

static bool ImpTriangulateFace(const aiFace &face, const aiVector3D* verts, aiFace **resultFaces, unsigned *nResult);



/**
 * A single triangulated face
 *
 * AssImp, at least from blender reads in a face as a set of vertices. We tell AssImp
 * not to triangulate it, because we triangulate ourselves.
 *
 * The ImpFace gets created after the mesh is created, and the polygon
 * ptr gets initialized when the ImpFace is constructed.
 */
struct ImpFace {

    ImpFace(MxMesh *mesh, const VectorMap &vecMap,
            const aiFace *face, const aiMesh *aim) {

        vertices = verticesForFace(vecMap, aim, face);

        polygon = mesh->createPolygon(MxPolygon_Type, vertices);

        assert(polygon != nullptr);
    }


    bool equals(const std::vector<VertexPtr> &verts) const {

        if(verts.size() != vertices.size()) {
            return false;
        }

        std::vector<VertexPtr> v = verts;

        for(int i = 0; i < verts.size(); ++i) {
            if(std::equal(v.begin(), v.end(), vertices.begin())) {
                return true;
            }
            std::rotate(v.begin(), v.begin() + 1, v.end());
        }

        // rotate back to original position
        //std::rotate(v.begin(), v.begin() + 1, v.end());
        assert(std::equal(v.begin(), v.end(), verts.begin()));

        // reverse the vertex order
        std::reverse(std::begin(v), std::end(v));

        // check all rotations again in reverse order
        for(int i = 0; i < verts.size(); ++i) {
            if(std::equal(v.begin(), v.end(), vertices.begin())) {
                return true;
            }
            std::rotate(v.begin(), v.begin() + 1, v.end());
        }

        return false;

    }

    /**
     * ids of the global vertices in the MxMesh.
     */
    std::vector<VertexPtr> vertices;

    /**
     * the associated mechanica polygon for this face.
     */
    PolygonPtr polygon = nullptr;
};


static void addImpFaceToCell(ImpFace *face, CellPtr cell);


/**
 * A set of faces
 */
struct ImpFaceSet {

    ImpFace *findTriangulatedFace(const VectorMap &vecMap,
            const aiMesh *mesh, const aiFace *face) {
        std::vector<VertexPtr> vertices = verticesForFace(vecMap, mesh, face);

        for(ImpFace &face : faces) {
            if(face.equals(vertices)) {
                return &face;
            }
        }
        return nullptr;
    }


    std::vector<ImpFace> faces;
};




static ImpEdge *findImpEdge(EdgeVector &edges, const ImpVertex *v1, const ImpVertex *v2) {
    for (ImpEdge &e : edges) {
        if(e.matches(v1, v2)) {
            return &e;
        }
    }
    return nullptr;
}

/**
 * Creates a new ImpEdge in the edges vector. Does not check to see if one already
 * exists.
 */
static ImpEdge *createImpEdge(EdgeVector &edges,  ImpVertex *v0, ImpVertex *v1) {
    edges.emplace_back(v0, v1);
    return &edges.back();
}



static void addUnclaimedPartialTrianglesToRoot(MxMesh *mesh)
{
    for(PolygonPtr tri : mesh->polygons) {
        assert(tri->cells[0]);
        if(!tri->cells[1]) {
            VERIFY(connectPolygonCell(tri, mesh->rootCell()));
        }
    }
    //assert(mesh->rootCell()->isValid());
}


/**
 * Searches a vector map for a corresponding vertex. If found, returns the vertex,
 * otherwise creates a new vertex and returns it.
 *
 * Checks to make sure a vertex does not already belong to 4 cells.
 */
static ImpVertex *findVertex(VectorMap &vecMap, const aiVector3D vec) {
    return vecMap.findVertex(vec);
}


MxMesh* MxMesh_FromFile(const char* fname, float density, MeshCellTypeHandler cellTypeHandler)
{
    Assimp::Importer imp;

    VectorMap vecMap(0.0001);
    EdgeVector edges;
    ImpFaceSet triFaces;

    uint flags = aiProcess_JoinIdenticalVertices |
            //aiProcess_Triangulate |
            aiProcess_RemoveComponent |
            aiProcess_RemoveRedundantMaterials |
            aiProcess_FindDegenerates;

    imp.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
            aiComponent_NORMALS |
            aiComponent_TANGENTS_AND_BITANGENTS |
            aiComponent_COLORS |
            aiComponent_TEXCOORDS |
            aiComponent_BONEWEIGHTS |
            aiComponent_CAMERAS |
            aiComponent_LIGHTS |
            aiComponent_MATERIALS |
            aiComponent_TEXTURES |
            aiComponent_ANIMATIONS
            );

    // Reads the given file and returns its contents if successful.
    const aiScene *scene = imp.ReadFile (fname, flags);

    if(!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << imp.GetErrorString() << std::endl;
        return nullptr;
    }
    
    // iterate over all meshes and vertices, and add the vertices to the map
    {
        // global vertex id.
        int vId = 0;
        for(int i = 0; i < scene->mNumMeshes; ++i) {
            aiMesh *mesh = scene->mMeshes[i];

            std::cout << "processing vertices from " << mesh->mName.C_Str() << std::endl;

            for(int j = 0; j < mesh->mNumVertices; ++j) {
                const aiVector3D &vec = mesh->mVertices[j];
                ImpVertex *v = vecMap.findVertex(vec);
                if(v == nullptr) {
                    v = vecMap.createVertex(vec, vId++);
                }
                assert(v);
                if(!v->addMesh(mesh, j)) {
                    return nullptr;
                }
            }
        }

        std::cout << "processed " << vId << " vertices" << std::endl;
        for(auto vert : vecMap) {
            std::cout << vert.toString() << std::endl;
        }

        int sharedVertices = 0;

        for(auto vert : vecMap) {
            if(vert.meshes.size() > 1) {
                sharedVertices++;
                std::cout << "shared vertex: " << vert.toString() << std::endl;
            }
        }

        std::cout << "found " << sharedVertices << " shared vertices" << std::endl;
    }
    // done adding all the vertices from the file
    
    // process all the skeletal edges
    for(int i = 0; i < scene->mNumMeshes; ++i) {
        aiMesh *mesh = scene->mMeshes[i];

        for(int j = 0; j < mesh->mNumFaces; ++j) {
            aiFace *face = &mesh->mFaces[j];
            
            std::cout << "processing edges from mesh " << mesh->mName.C_Str() << ", face " << j << std::endl;

            for(int k = 0; k < face->mNumIndices; ++k) {

                const int vid0 = face->mIndices[k];
                const int vid1 = face->mIndices[(k+1) % face->mNumIndices];

                // the vertex and the next vertex in the polygon face. These are connected
                // by a skeletal edge by definition.
                aiVector3D av1 = mesh->mVertices[vid0];
                aiVector3D av2 = mesh->mVertices[vid1];

                ImpVertex *v1 = findVertex(vecMap, av1);
                if(!v1) {return nullptr;}
                ImpVertex *v2 = findVertex(vecMap, av2);
                if(!v2) {return nullptr;}
                
                

                if (!findImpEdge(edges, v1, v2)) {
                    //std::cout << "no existing edge for face " << j << ", vertices " << v1->id << ", " << v2->id << std::endl;

                    if(v1->edges.size() >= 4) {
                        std::cout << "error, mesh:" << mesh->mName.C_Str() << ", face: " << j << ", vertex: "
                                  << v1->toString() << " already has " << v1->edges.size()
                                  << " edges, can't add more." << std::endl
                                  << "tried to add edge to " << v2->toString() << std::endl
                        << "existing edges: {";
                        for(const ImpEdge* e : v1->edges) {
                            std::cout << "edge: {" << e->verts[0]->toString() << ", "
                                      << e->verts[1]->toString() << "}, ";
                        }
                        std::cout << "}" << std::endl;
   
                        //return nullptr;
                        continue;
                    }

                    if(v2->edges.size() >= 4) {
                        std::cout << "error, mesh:" << mesh->mName.C_Str() << ", face: " << j << ", vertex: "
                                  << v2->toString() << " already has " << v2->edges.size()
                                  << " edges, can't add more." << std::endl
                                  << "tried to add edge to " << v1->toString() << std::endl;
                        //return nullptr;
                        continue;
                    }
                    
                    std::cout << "creating edge: {" << v1->id << ", " << v2->id  << "}" << std::endl;

                    ImpEdge *edge = createImpEdge(edges, v1, v2);
                    v1->addEdge(edge);
                    v2->addEdge(edge);

                    //std::cout << "mesh " <<  mesh->mName.C_Str() << ", vertex " <<
                    //        std::to_string(j) << " edge count: " << std::to_string(v1->edges.size()) << std::endl;

                    //std::cout << "mesh " <<  mesh->mName.C_Str() << ", vertex " <<
                    //        std::to_string((j+1) % mesh->mNumVertices)
                    //<< " edge count: " << std::to_string(v2->edges.size()) << std::endl;

                }
                else {
                    std::cout << "found edge for face " << j << ", vertices " << vid0 << ", " << vid1 << std::endl;
                }
            }
        }
    }

    //{
    //    int j = 0;
    //
    //    for(auto vert : vecMap) {
    //        std::cout << "vertex: " << std::to_string(j++) << ", edge count: " << vert.edges.size() << std::endl;
    //    }
    //}

    // now we've iterated over all vertices in the mesh, and checked to make sure that
    // no vertex is incident to more than 4 cells, safe to make new mesh now.

    // iterate over all the stored vertices in the dictionary, and make new vertices
    // in the mesh for these.

    MxMesh *mesh = new MxMesh();
    
    // create mesh vertices for ever vertex that we read from the file
    for(ImpVertex &vert : vecMap) {
        assert(vert.vert == nullptr);
        vert.vert = mesh->createVertex({{vert.pos.x, vert.pos.y, vert.pos.z}});
        std::cout << "created new vertex: " << vert.vert << std::endl;
    }

    // first add all the vertices that are in the skeletal edge list,
    // these are skeletal vertices.
    for(ImpEdge &edge : edges) {
        edge.edge = mesh->createEdge(MxEdge_Type, edge.verts[0]->vert, edge.verts[1]->vert);
        assert(edge.edge != nullptr);
    }

    assert(scene->mRootNode);

    // A scene is organized into a hierarchical set of 'objects', starting
    // with the root object. Each object can have multiple meshes, for different
    // materials. We need to grab all the triangles from each mesh, and create a
    // cell out of them.
    // the AI mesh has a set of 'faces', Each face is a set of indices.
    // We tell AssImp not to triangulate, so each face is a set of vertices
    // that define the polygonal face between a pair of cells.
    for(int i = 0; i < scene->mRootNode->mNumChildren; ++i) {
        
        aiNode *obj = scene->mRootNode->mChildren[i];
        
        std::cout << "creating new cell \"" << obj->mName.C_Str() << "\"" << std::endl;

        CellPtr cell = mesh->createCell(cellTypeHandler(obj->mName.C_Str(), i), obj->mName.C_Str());
        
        for(int m = 0; m < obj->mNumMeshes; ++m) {
            
            aiMesh *aim = scene->mMeshes[obj->mMeshes[m]];
            
            std::cout << "processing triangles from mesh " << aim->mName.C_Str() << "\"" << std::endl;

            // these are polygonal faces, each edge in a polygonal face defines a skeletal edge.
            for(int j = 0; j < aim->mNumFaces; ++j) {
                aiFace *face = &aim->mFaces[j];

                ImpFace *triFace = triFaces.findTriangulatedFace(vecMap, aim, face);

                if(!triFace) {
                    triFaces.faces.push_back(ImpFace(mesh, vecMap, face, aim));
                    assert(triFaces.findTriangulatedFace(vecMap, aim, face));
                    triFace = &triFaces.faces.back();
                }
                else {
                    std::cout << "found shared face with " << triFace->vertices.size() << " vertices" << std::endl;
                }

                addImpFaceToCell(triFace, cell);
            }
        }
        
        // done with all of the meshes for this cell, the cell should
        // have a complete set of triangles now.
        
        for(PPolygonPtr pt : cell->surface) {
            float area = Magnum::Math::triangleArea(pt->polygon->vertices[0]->position,
                                                     pt->polygon->vertices[1]->position,
                                                     pt->polygon->vertices[2]->position);
            pt->mass = area * density;
        }
        


        //assert(mesh->valid(cell));
        //assert(cell->updateDerivedAttributes() == S_OK);
        //assert(cell->isValid());
    }
    
    addUnclaimedPartialTrianglesToRoot(mesh);

    // at this point, all the vertices, skeletal edges and triangles have been added
    // to the mesh, so now go over the triangles, check which ones are supposed to be
    // connected to the skeletal edges by their vertex relationships, and connect
    // the pointers.
    //for(PolygonPtr tri : mesh->triangles) {
    //    for(SkeletalEdgePtr edge : mesh->edges) {
    //        if(incidentEdgeTriangleVertices(edge, tri) && !connectedEdgeTrianglePointers(edge, tri)) {
    //            VERIFY(connectEdgeTriangle(edge, tri));
    //        }
    //    }
    //}
     

    // now we have connected all the skeletal edges, but triangles with a manifold
    // connection to neighboring triangles are still not connected, connect
    // these now.
    //for(PolygonPtr tri : mesh->triangles) {
    //    for(int i = 0; i < 3; ++i) {
    //        if(tri->neighbors[i] == nullptr) {
    //            assert(tri->partialTriangles[0].neighbors[i] && tri->partialTriangles[1].neighbors[i]);
    //            VERIFY(connectTriangleTriangle(tri, tri->partialTriangles[0].neighbors[i]->polygon));
    //        }
    //    }
    //}
    
    VERIFY(mesh->positionsChanged());

#ifndef NDEBUG
    for (CCellPtr cell : mesh->cells) {
        cell->dump();
    }
#endif
    return mesh;
}


/*
 *
 * Copy and pasted face triangulation routine from AssImp.
 *
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2018, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

/** @file  TriangulateProcess.cpp
 *  @brief Implementation of the post processing step to split up
 *    all faces with more than three indices into triangles.
 *
 *
 *  The triangulation algorithm will handle concave or convex polygons.
 *  Self-intersecting or non-planar polygons are not rejected, but
 *  they're probably not triangulated correctly.
 *
 * DEBUG SWITCHES - do not enable any of them in release builds:
 *
 * AI_BUILD_TRIANGULATE_COLOR_FACE_WINDING
 *   - generates vertex colors to represent the face winding order.
 *     the first vertex of a polygon becomes red, the last blue.
 * AI_BUILD_TRIANGULATE_DEBUG_POLYS
 *   - dump all polygons and their triangulation sequences to
 *     a file
 */



#include <memory>

//#define AI_BUILD_TRIANGULATE_COLOR_FACE_WINDING
//#define AI_BUILD_TRIANGULATE_DEBUG_POLYS

#define POLY_GRID_Y 40
#define POLY_GRID_X 70
#define POLY_GRID_XPAD 20
#define POLY_OUTPUT_FILE "assimp_polygons_debug.txt"


// -------------------------------------------------------------------------------
/** Compute the signed area of a triangle.
 *  The function accepts an unconstrained template parameter for use with
 *  both aiVector3D and aiVector2D, but generally ignores the third coordinate.*/
template <typename T>
inline double GetArea2D(const T& v1, const T& v2, const T& v3)
{
    return 0.5 * (v1.x * ((double)v3.y - v2.y) + v2.x * ((double)v1.y - v3.y) + v3.x * ((double)v2.y - v1.y));
}

// -------------------------------------------------------------------------------
/** Test if a given point p2 is on the left side of the line formed by p0-p1.
 *  The function accepts an unconstrained template parameter for use with
 *  both aiVector3D and aiVector2D, but generally ignores the third coordinate.*/
template <typename T>
inline bool OnLeftSideOfLine2D(const T& p0, const T& p1,const T& p2)
{
    return GetArea2D(p0,p2,p1) > 0;
}


// -------------------------------------------------------------------------------
/** Compute the normal of an arbitrary polygon in R3.
 *
 *  The code is based on Newell's formula, that is a polygons normal is the ratio
 *  of its area when projected onto the three coordinate axes.
 *
 *  @param out Receives the output normal
 *  @param num Number of input vertices
 *  @param x X data source. x[ofs_x*n] is the n'th element.
 *  @param y Y data source. y[ofs_y*n] is the y'th element
 *  @param z Z data source. z[ofs_z*n] is the z'th element
 *
 *  @note The data arrays must have storage for at least num+2 elements. Using
 *  this method is much faster than the 'other' NewellNormal()
 */
template <int ofs_x, int ofs_y, int ofs_z, typename TReal>
inline void NewellNormal (aiVector3t<TReal>& out, int num, TReal* x, TReal* y, TReal* z)
{
    // Duplicate the first two vertices at the end
    x[(num+0)*ofs_x] = x[0];
    x[(num+1)*ofs_x] = x[ofs_x];

    y[(num+0)*ofs_y] = y[0];
    y[(num+1)*ofs_y] = y[ofs_y];

    z[(num+0)*ofs_z] = z[0];
    z[(num+1)*ofs_z] = z[ofs_z];

    TReal sum_xy = 0.0, sum_yz = 0.0, sum_zx = 0.0;

    TReal *xptr = x +ofs_x, *xlow = x, *xhigh = x + ofs_x*2;
    TReal *yptr = y +ofs_y, *ylow = y, *yhigh = y + ofs_y*2;
    TReal *zptr = z +ofs_z, *zlow = z, *zhigh = z + ofs_z*2;

    for (int tmp=0; tmp < num; tmp++) {
        sum_xy += (*xptr) * ( (*yhigh) - (*ylow) );
        sum_yz += (*yptr) * ( (*zhigh) - (*zlow) );
        sum_zx += (*zptr) * ( (*xhigh) - (*xlow) );

        xptr  += ofs_x;
        xlow  += ofs_x;
        xhigh += ofs_x;

        yptr  += ofs_y;
        ylow  += ofs_y;
        yhigh += ofs_y;

        zptr  += ofs_z;
        zlow  += ofs_z;
        zhigh += ofs_z;
    }
    out = aiVector3t<TReal>(sum_yz,sum_zx,sum_xy);
}


// -------------------------------------------------------------------------------
/** Test if a given point is inside a given triangle in R2.
 * The function accepts an unconstrained template parameter for use with
 *  both aiVector3D and aiVector2D, but generally ignores the third coordinate.*/
template <typename T>
inline bool PointInTriangle2D(const T& p0, const T& p1,const T& p2, const T& pp)
{
    // Point in triangle test using baryzentric coordinates
    const aiVector2D v0 = p1 - p0;
    const aiVector2D v1 = p2 - p0;
    const aiVector2D v2 = pp - p0;

    double dot00 = v0 * v0;
    double dot01 = v0 * v1;
    double dot02 = v0 * v2;
    double dot11 = v1 * v1;
    double dot12 = v1 * v2;

    const double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    dot11 = (dot11 * dot02 - dot01 * dot12) * invDenom;
    dot00 = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (dot11 > 0) && (dot00 > 0) && (dot11 + dot00 < 1);
}



/**
 * Copied directly from AssImp.
 *
 * Accepts a single input face that defines a polygon, and generates a list of triangular faces.
 */
bool ImpTriangulateFace(const aiFace &face, const aiVector3D* verts, aiFace **resultFaces, unsigned *nResult)
{
    *nResult = 0;
    // Now we have aiMesh::mPrimitiveTypes, so this is only here for test cases
    //if (!pMesh->mPrimitiveTypes)    {
    //    bool bNeed = false;
    //
    //    for( unsigned int a = 0; a < pMesh->mNumFaces; a++) {
    //        const aiFace& face = pMesh->mFaces[a];
    //
    //        if( face.mNumIndices != 3)  {
    //            bNeed = true;
    //        }
    //    }
    //    if (!bNeed)
    //        return false;
    //}
    //else if (!(pMesh->mPrimitiveTypes & aiPrimitiveType_POLYGON)) {
    //    return false;
    //}

    // Find out how many output faces we'll get
    unsigned int numOut = 0, max_out = 0;
    bool get_normals = true;
    //for( unsigned int a = 0; a < pMesh->mNumFaces; a++) {
    //    aiFace& face = pMesh->mFaces[a];
        if (face.mNumIndices <= 4) {
            get_normals = false;
        }
        if( face.mNumIndices <= 3) {
            numOut++;
        }
        else {
            numOut += face.mNumIndices-2;
            max_out = std::max(max_out,face.mNumIndices);
        }
    //}

    // Just another check whether aiMesh::mPrimitiveTypes is correct
    //assert(numOut != pMesh->mNumFaces);

    aiVector3D* nor_out = NULL;


    // the output mesh will contain triangles, but no polys anymore
    //pMesh->mPrimitiveTypes |= aiPrimitiveType_TRIANGLE;
    //pMesh->mPrimitiveTypes &= ~aiPrimitiveType_POLYGON;

    aiFace* out = new aiFace[numOut](), *curOut = out;
    *resultFaces = out;
    std::vector<aiVector3D> temp_verts3d(max_out+2); /* temporary storage for vertices */
    std::vector<aiVector2D> temp_verts(max_out+2);

    // Apply vertex colors to represent the face winding?
#ifdef AI_BUILD_TRIANGULATE_COLOR_FACE_WINDING
    if (!pMesh->mColors[0])
        pMesh->mColors[0] = new aiColor4D[pMesh->mNumVertices];
    else
        new(pMesh->mColors[0]) aiColor4D[pMesh->mNumVertices];

    aiColor4D* clr = pMesh->mColors[0];
#endif

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS
    FILE* fout = fopen(POLY_OUTPUT_FILE,"a");
#endif



    // use std::unique_ptr to avoid slow std::vector<bool> specialiations
    std::unique_ptr<bool[]> done(new bool[max_out]);
    //for( unsigned int a = 0; a < pMesh->mNumFaces; a++) {



        unsigned int* idx = face.mIndices;
        int num = (int)face.mNumIndices, ear = 0, tmp, prev = num-1, next = 0, max = num;

        // Apply vertex colors to represent the face winding?
#ifdef AI_BUILD_TRIANGULATE_COLOR_FACE_WINDING
        for (unsigned int i = 0; i < face.mNumIndices; ++i) {
            aiColor4D& c = clr[idx[i]];
            c.r = (i+1) / (float)max;
            c.b = 1.f - c.r;
        }
#endif

        aiFace* const last_face = curOut;

        // if it's a simple point,line or triangle: just copy it
        if( face.mNumIndices <= 3)
        {
            aiFace& nface = *curOut++;
            nface.mNumIndices = face.mNumIndices;
            nface.mIndices    = face.mIndices;

            //face.mIndices = NULL;
            *nResult = 1;
            return true;
        }
        // optimized code for quadrilaterals
        else if ( face.mNumIndices == 4) {

            // quads can have at maximum one concave vertex. Determine
            // this vertex (if it exists) and start tri-fanning from
            // it.
            unsigned int start_vertex = 0;
            for (unsigned int i = 0; i < 4; ++i) {
                const aiVector3D& v0 = verts[face.mIndices[(i+3) % 4]];
                const aiVector3D& v1 = verts[face.mIndices[(i+2) % 4]];
                const aiVector3D& v2 = verts[face.mIndices[(i+1) % 4]];

                const aiVector3D& v = verts[face.mIndices[i]];

                aiVector3D left = (v0-v);
                aiVector3D diag = (v1-v);
                aiVector3D right = (v2-v);

                left.Normalize();
                diag.Normalize();
                right.Normalize();

                const float angle = std::acos(left*diag) + std::acos(right*diag);
                if (angle > AI_MATH_PI_F) {
                    // this is the concave point
                    start_vertex = i;
                    break;
                }
            }

            const unsigned int temp[] = {face.mIndices[0], face.mIndices[1], face.mIndices[2], face.mIndices[3]};

            aiFace& nface = *curOut++;
            nface.mNumIndices = 3;
            nface.mIndices = new unsigned int[3];

            nface.mIndices[0] = temp[start_vertex];
            nface.mIndices[1] = temp[(start_vertex + 1) % 4];
            nface.mIndices[2] = temp[(start_vertex + 2) % 4];

            aiFace& sface = *curOut++;
            sface.mNumIndices = 3;
            sface.mIndices = new unsigned int[3];

            sface.mIndices[0] = temp[start_vertex];
            sface.mIndices[1] = temp[(start_vertex + 2) % 4];
            sface.mIndices[2] = temp[(start_vertex + 3) % 4];

            // prevent double deletion of the indices field
            // face.mIndices = NULL;
            *nResult = (unsigned)(curOut-out);
            return true;
        }
        else
        {
            // A polygon with more than 3 vertices can be either concave or convex.
            // Usually everything we're getting is convex and we could easily
            // triangulate by tri-fanning. However, LightWave is probably the only
            // modeling suite to make extensive use of highly concave, monster polygons ...
            // so we need to apply the full 'ear cutting' algorithm to get it right.

            // RERQUIREMENT: polygon is expected to be simple and *nearly* planar.
            // We project it onto a plane to get a 2d triangle.

            // Collect all vertices of of the polygon.
            for (tmp = 0; tmp < max; ++tmp) {
                temp_verts3d[tmp] = verts[idx[tmp]];
            }

            // Get newell normal of the polygon. Store it for future use if it's a polygon-only mesh
            aiVector3D n;
            NewellNormal<3,3,3>(n,max,&temp_verts3d.front().x,&temp_verts3d.front().y,&temp_verts3d.front().z);
            if (nor_out) {
                 for (tmp = 0; tmp < max; ++tmp)
                     nor_out[idx[tmp]] = n;
            }

            // Select largest normal coordinate to ignore for projection
            const float ax = (n.x>0 ? n.x : -n.x);
            const float ay = (n.y>0 ? n.y : -n.y);
            const float az = (n.z>0 ? n.z : -n.z);

            unsigned int ac = 0, bc = 1; /* no z coord. projection to xy */
            float inv = n.z;
            if (ax > ay) {
                if (ax > az) { /* no x coord. projection to yz */
                    ac = 1; bc = 2;
                    inv = n.x;
                }
            }
            else if (ay > az) { /* no y coord. projection to zy */
                ac = 2; bc = 0;
                inv = n.y;
            }

            // Swap projection axes to take the negated projection vector into account
            if (inv < 0.f) {
                std::swap(ac,bc);
            }

            for (tmp =0; tmp < max; ++tmp) {
                temp_verts[tmp].x = verts[idx[tmp]][ac];
                temp_verts[tmp].y = verts[idx[tmp]][bc];
                done[tmp] = false;
            }

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS
            // plot the plane onto which we mapped the polygon to a 2D ASCII pic
            aiVector2D bmin,bmax;
            ArrayBounds(&temp_verts[0],max,bmin,bmax);

            char grid[POLY_GRID_Y][POLY_GRID_X+POLY_GRID_XPAD];
            std::fill_n((char*)grid,POLY_GRID_Y*(POLY_GRID_X+POLY_GRID_XPAD),' ');

            for (int i =0; i < max; ++i) {
                const aiVector2D& v = (temp_verts[i] - bmin) / (bmax-bmin);
                const size_t x = static_cast<size_t>(v.x*(POLY_GRID_X-1)), y = static_cast<size_t>(v.y*(POLY_GRID_Y-1));
                char* loc = grid[y]+x;
                if (grid[y][x] != ' ') {
                    for(;*loc != ' '; ++loc);
                    *loc++ = '_';
                }
                *(loc+::ai_snprintf(loc, POLY_GRID_XPAD,"%i",i)) = ' ';
            }


            for(size_t y = 0; y < POLY_GRID_Y; ++y) {
                grid[y][POLY_GRID_X+POLY_GRID_XPAD-1] = '\0';
                fprintf(fout,"%s\n",grid[y]);
            }

            fprintf(fout,"\ntriangulation sequence: ");
#endif

            //
            // FIXME: currently this is the slow O(kn) variant with a worst case
            // complexity of O(n^2) (I think). Can be done in O(n).
            while (num > 3) {

                // Find the next ear of the polygon
                int num_found = 0;
                for (ear = next;;prev = ear,ear = next) {

                    // break after we looped two times without a positive match
                    for (next=ear+1;done[(next>=max?next=0:next)];++next);
                    if (next < ear) {
                        if (++num_found == 2) {
                            break;
                        }
                    }
                    const aiVector2D* pnt1 = &temp_verts[ear],
                        *pnt0 = &temp_verts[prev],
                        *pnt2 = &temp_verts[next];

                    // Must be a convex point. Assuming ccw winding, it must be on the right of the line between p-1 and p+1.
                    if (OnLeftSideOfLine2D(*pnt0,*pnt2,*pnt1)) {
                        continue;
                    }

                    // and no other point may be contained in this triangle
                    for ( tmp = 0; tmp < max; ++tmp) {

                        // We need to compare the actual values because it's possible that multiple indexes in
                        // the polygon are referring to the same position. concave_polygon.obj is a sample
                        //
                        // FIXME: Use 'epsiloned' comparisons instead? Due to numeric inaccuracies in
                        // PointInTriangle() I'm guessing that it's actually possible to construct
                        // input data that would cause us to end up with no ears. The problem is,
                        // which epsilon? If we chose a too large value, we'd get wrong results
                        const aiVector2D& vtmp = temp_verts[tmp];
                        if ( vtmp != *pnt1 && vtmp != *pnt2 && vtmp != *pnt0 && PointInTriangle2D(*pnt0,*pnt1,*pnt2,vtmp)) {
                            break;
                        }
                    }
                    if (tmp != max) {
                        continue;
                    }

                    // this vertex is an ear
                    break;
                }
                if (num_found == 2) {

                    // Due to the 'two ear theorem', every simple polygon with more than three points must
                    // have 2 'ears'. Here's definitely something wrong ... but we don't give up yet.
                    //

                    // Instead we're continuing with the standard tri-fanning algorithm which we'd
                    // use if we had only convex polygons. That's life.
                    // DefaultLogger::get()->error("Failed to triangulate polygon (no ear found). Probably not a simple polygon?");

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS
                    fprintf(fout,"critical error here, no ear found! ");
#endif
                    num = 0;
                    break;

                    curOut -= (max-num); /* undo all previous work */
                    for (tmp = 0; tmp < max-2; ++tmp) {
                        aiFace& nface = *curOut++;

                        nface.mNumIndices = 3;
                        if (!nface.mIndices)
                            nface.mIndices = new unsigned int[3];

                        nface.mIndices[0] = 0;
                        nface.mIndices[1] = tmp+1;
                        nface.mIndices[2] = tmp+2;

                    }
                    num = 0;
                    break;
                }

                aiFace& nface = *curOut++;
                nface.mNumIndices = 3;

                if (!nface.mIndices) {
                    nface.mIndices = new unsigned int[3];
                }

                // setup indices for the new triangle ...
                nface.mIndices[0] = prev;
                nface.mIndices[1] = ear;
                nface.mIndices[2] = next;

                // exclude the ear from most further processing
                done[ear] = true;
                --num;
            }
            if (num > 0) {
                // We have three indices forming the last 'ear' remaining. Collect them.
                aiFace& nface = *curOut++;
                nface.mNumIndices = 3;
                if (!nface.mIndices) {
                    nface.mIndices = new unsigned int[3];
                }

                for (tmp = 0; done[tmp]; ++tmp);
                nface.mIndices[0] = tmp;

                for (++tmp; done[tmp]; ++tmp);
                nface.mIndices[1] = tmp;

                for (++tmp; done[tmp]; ++tmp);
                nface.mIndices[2] = tmp;

            }
        }

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS

        for(aiFace* f = last_face; f != curOut; ++f) {
            unsigned int* i = f->mIndices;
            fprintf(fout," (%i %i %i)",i[0],i[1],i[2]);
        }

        fprintf(fout,"\n*********************************************************************\n");
        fflush(fout);

#endif

        for(aiFace* f = last_face; f != curOut; ) {
            unsigned int* i = f->mIndices;

            //  drop dumb 0-area triangles
            if (std::fabs(GetArea2D(temp_verts[i[0]],temp_verts[i[1]],temp_verts[i[2]])) < 1e-5f) {

                //DefaultLogger::get()->debug("Dropping triangle with area 0");
                --curOut;

                delete[] f->mIndices;
                f->mIndices = NULL;

                for(aiFace* ff = f; ff != curOut; ++ff) {
                    ff->mNumIndices = (ff+1)->mNumIndices;
                    ff->mIndices = (ff+1)->mIndices;
                    (ff+1)->mIndices = NULL;
                }
                continue;
            }

            i[0] = idx[i[0]];
            i[1] = idx[i[1]];
            i[2] = idx[i[2]];
            ++f;
        }

        //delete[] face.mIndices;
        //face.mIndices = NULL;
    //}

#ifdef AI_BUILD_TRIANGULATE_DEBUG_POLYS
    fclose(fout);
#endif

    // kill the old faces
    //delete [] pMesh->mFaces;



    // ... and store the new ones
    //pMesh->mFaces    = out;
    *nResult = (unsigned)(curOut-out); /* not necessarily equal to numOut */
    return true;
}


static std::vector<VertexPtr> verticesForFace(const VectorMap &vecMap, const aiMesh* mesh,
        const aiFace *face) {

    std::vector<VertexPtr> result(face->mNumIndices);

    for(int i = 0; i < face->mNumIndices; ++i) {
        assert(face->mIndices[i] >= 0 && face->mIndices[i] < mesh->mNumVertices);
        const aiVector3D &vert = mesh->mVertices[face->mIndices[i]];
        VectorMap::const_iterator j = vecMap.find(vert);
        assert(j != vecMap.end());
        result[i] = j->vert;
    }

    return result;
}



inline void addImpFaceToCell(ImpFace* face, CellPtr cell)
{
    assert(face->polygon);
    VERIFY(connectPolygonCell(face->polygon, cell));
}
