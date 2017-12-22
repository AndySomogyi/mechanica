/*
 * MxMeshVoronoiImporter.cpp
 *
 *  Created on: Jul 8, 2017
 *      Author: andy
 */

#include <iostream>
#include <tuple>

#include <MxMeshVoronoiImporter.h>
#include <voro++.hh>
#include <Magnum/Math/Vector3.h>
#include <MagnumPlugins/ObjImporter/ObjImporter.h>


#include <Magnum/Trade/MeshData3D.h>


using namespace std;
using namespace Magnum;
using namespace Magnum::Trade;


typedef std::optional<MeshData3D> OptMeshData3D;

using namespace Magnum;

const int Keep = 10000;

/**
 * Big problem with Voro++ is that it often generates many degenerate triangles.
 * How to deal with degenerate triangles is tricky.
 */

template<typename T>
bool pack(MxMeshVoronoiImporter& imp, T& container) {


    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<VertexPtr> newInd;

    voro::c_loop_all vl(container);
    double *cellOrigin;
    voro::voronoicell c;
    int i = 0;
    if(vl.start()) do {
        if(container.compute_cell(c,vl)) {

            //if (i != 3) continue;

            // the center of the voronoi cell, should be same as the
            // inserted particle location.
            cellOrigin = container.p[vl.ijk] + container.ps * vl.q;

            cout << "Voro++ Cell [" << i << "]" << std::endl;
            c.draw_pov_mesh(cellOrigin[0], cellOrigin[1], cellOrigin[2], stdout);
            cout << std::endl;

            // grab the indexed vertices from the cell
            c.indexed_triangular_faces(cellOrigin[0], cellOrigin[1], cellOrigin[2], vertices, indices);

            // do these in-place once bugs are worked out
            newInd.resize(indices.size());

            for (int j = 0; j < indices.size(); ++j) {
                int k = indices[j];
                newInd[j] = imp.mesh.createVertex({float(vertices[3*k]), float(vertices[3*k+1]), float(vertices[3*k+2])});
            }

            // allocate first in the cells vector, then we write directly to that memory block,
            // avoid copy
            CellPtr cell = imp.mesh.createCell();

            // add the faces to the cell, then sort out the connectivity
            for(int j = 0; j < newInd.size(); j+=3) {
                std::array<VertexPtr, 3> triInd = {{newInd[j], newInd[j+1], newInd[j+2]}};
                imp.createTriangleForCell(triInd, cell);
            }

            std::cout << "The cell...\n";
            cell->writePOV(std::cout);

            std::cout << "connecting...\n";


            // boundary had better be connected from a voronoi cell
            bool connected = cell->manifold();

            cout << "MxCell["<< i << "]" << std::endl;
            cell->writePOV(std::cout);
            cout << std::endl;


            assert(connected);
        }
    } while(vl.inc() && ++i < Keep);

    //mesh.initPos.resize(mesh.vertices.size());

    //for(int i = 0; i < mesh.vertices.size(); ++i) {
     //   mesh.initPos[i] = mesh.vertices[i].position;
    //}


    //mesh.dump(0);

    return true;

}

static OptMeshData3D  readPoints(const std::string& path) {

    ObjImporter importer{};

    if (importer.openFile(path)) {
        cout << "opened file \"" << path << "\" OK" << std::endl;
        cout << "mesh 3d count: " << importer.mesh3DCount() << std::endl;
    } else {
        cout << "failed to open " <<  path << std::endl;
        return OptMeshData3D{};
    }

    int defScene = importer.defaultScene();
    cout << "default scene: " << defScene << std::endl;
    cout << "getting mesh3d(0)" << std::endl;
    return importer.mesh3D(0);
}



bool MxMeshVoronoiImporter::testing(const std::string& path,
        const Magnum::Vector3& min, const Magnum::Vector3& max,
        const Magnum::Vector3i& n, const std::array<bool, 3> periodic)
{
    auto meshData = readPoints(path);

    cout << "pos count: " << meshData->positionArrayCount() << std::endl;



    std::vector<Vector3> positions = meshData->positions(0);

    //container::container(double ax_,double bx_,double ay_,double by_,double az_,double bz_,
        //    int nx_,int ny_,int nz_,bool xperiodic_,bool yperiodic_,bool zperiodic_,int init_mem)
    voro::container container(min[0], max[0], min[1], max[1], min[2], max[2],
        n[0], n[1], n[2], periodic[0], periodic[1], periodic[2], 50);

    {
        int i = 0;
        for (const Vector3& vec : positions) {
            container.put(i++, vec[0], vec[1], vec[2]);
        }
    }

    // Output the particle positions in povray format
    container.draw_particles_pov("test_p.pov");

    // Output the Voronoi cells in povray format
    container.draw_cells_pov("test_v.pov");


    // The voro++ void voronoicell_base::draw_pov(double x,double y,double z,FILE* fp)
    // code, looks like the domain is doubled,  weird...
    //    int i,j,k;double *ptsp=pts,*pt2;
    //    char posbuf1[128],posbuf2[128];
    //    for(i=0;i<p;i++,ptsp+=3) {
    //        sprintf(posbuf1,"%g,%g,%g",x+*ptsp*0.5,y+ptsp[1]*0.5,z+ptsp[2]*0.5);
    //        fprintf(fp,"sphere{<%s>,r}\n",posbuf1);
    //        for(j=0;j<nu[i];j++) {
    //            k=ed[i][j];
    //            if(k<i) {
    //                pt2=pts+3*k;
    //                sprintf(posbuf2,"%g,%g,%g",x+*pt2*0.5,y+0.5*pt2[1],z+0.5*pt2[2]);
    //                if(strcmp(posbuf1,posbuf2)!=0) fprintf(fp,"cylinder{<%s>,<%s>,r}\n",posbuf1,posbuf2);
    //            }
    //        }
    //    }



    {
        voro::c_loop_all vl(container);

        voro::voronoicell c; double *pp;
        if(vl.start()) do {
            if(container.compute_cell(c,vl)) {
                // voro++ stores vertex in contiguous arrays, c.pts
                const int stride = 3;

                // the center of the voronoi cell, should be same as the
                // inserted particle location.
                pp = container.p[vl.ijk] + container.ps * vl.q;

                cout << "voronoi center: " << pp[0] << ", " << pp[1] << ", " << pp[2] << std::endl;
                cout << "points {" << std::endl;
                for (int i = 0; i < c.p; ++i) {
                    cout << "    " << 0.5 * c.pts[i*stride]   + pp[0]
                                   << "," << 0.5 * c.pts[i*stride+1] + pp[1]
                                   << "," << 0.5 * c.pts[i*stride+2] + pp[2] << std::endl;
                }
                cout << "}" << std::endl;
                //c.draw_gnuplot(*pp,pp[1],pp[2],fp);
            }
        }
        while(vl.inc());
    }

    {
        voro::c_loop_all vl(container);

        voro::voronoicell c; double *pp;
        if(vl.start()) do {
            if(container.compute_cell(c,vl)) {
                // voro++ stores vertex in contiguous arrays, c.pts
                const int stride = 3;

                // the center of the voronoi cell, should be same as the
                // inserted particle location.
                pp = container.p[vl.ijk] + container.ps * vl.q;

                std::vector<double> vertices;
                std::vector<int> indices;

                c.vertices(pp[0], pp[1], pp[2], vertices);
                c.face_vertices(indices);

                cout << "voro_print_face_vertices: " << std::endl;
                voro::voro_print_face_vertices(indices, stdout);
                cout << std::endl;

                cout << "voro_print_vector: " << std::endl;
                voro::voro_print_vector(indices, stdout);
                cout << std::endl;

                cout << "voro_print_vector: " << std::endl;
                voro::voro_print_vector(vertices, stdout);
                cout << std::endl;

                c.indexed_triangular_faces(pp[0], pp[1], pp[2], vertices, indices);

                cout << "voro_print_face_vertices: " << std::endl;
                voro::voro_print_face_vertices(indices, stdout);
                cout << std::endl;

                cout << "voro_print_vector: " << std::endl;
                voro::voro_print_vector(indices, stdout);
                cout << std::endl;

                cout << "voro_print_vector: " << std::endl;
                voro::voro_print_vector(vertices, stdout);
                cout << std::endl;

                c.draw_pov_mesh(pp[0], pp[1], pp[2], stdout);
                cout << std::endl;
            }
        }
        while(vl.inc());
    }



    return true;
}

bool MxMeshVoronoiImporter::readFile(const std::string& path,
        const Magnum::Vector3& min, const Magnum::Vector3& max,
        const Magnum::Vector3i& n, const std::array<bool, 3> periodic)
{
    auto meshData = readPoints(path);

    cout << "pos count: " << meshData->positionArrayCount() << std::endl;

    std::vector<Vector3> positions = meshData->positions(0);

    //container::container(double ax_,double bx_,double ay_,double by_,double az_,double bz_,
    //    int nx_,int ny_,int nz_,bool xperiodic_,bool yperiodic_,bool zperiodic_,int init_mem)
    voro::container container(min[0], max[0], min[1], max[1], min[2], max[2],
        n[0], n[1], n[2], periodic[0], periodic[1], periodic[2], 50);


    return pack(*this, container);
}

#include <random>



bool MxMeshVoronoiImporter::random(uint numPts, const Magnum::Vector3& min,
        const Magnum::Vector3& max, const Magnum::Vector3i& n,
        const std::array<bool, 3> periodic) {


    //container::container(double ax_,double bx_,double ay_,double by_,double az_,double bz_,
    //    int nx_,int ny_,int nz_,bool xperiodic_,bool yperiodic_,bool zperiodic_,int init_mem)
    voro::container container(min[0], max[0], min[1], max[1], min[2], max[2],
        n[0], n[1], n[2], periodic[0], periodic[1], periodic[2], 50);

    std::default_random_engine eng;

    std::uniform_real_distribution<double> xRand(min[0] + 0.1, max[0] - 0.1);
    std::uniform_real_distribution<double> yRand(min[1] + 0.1, max[1] - 0.1);
    std::uniform_real_distribution<double> zRand(min[2] + 0.1, max[1] - 0.1);


    for (int i = 0; i < numPts; ++i)  {
        container.put(i++, xRand(eng), yRand(eng), zRand(eng));
    }

    return pack(*this, container);
}

bool MxMeshVoronoiImporter::monodisperse() {
    // Set up constants for the container geometry
    const double x_min=-3,x_max=3;
    const double y_min=-3,y_max=3;
    const double z_min=0,z_max=6;

    // Set up the number of blocks that the container is divided
    // into.
    const int n_x=3,n_y=3,n_z=3;


    // Create a container with the geometry given above, and make it
    // non-periodic in each of the three coordinates. Allocate space for
    // eight particles within each computational block. Import
    // the monodisperse test packing and output the Voronoi
    // tessellation in gnuplot and POV-Ray formats.
    voro::container_poly container(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
            false,false,false,8);
    container.import("/Users/andy/src/mechanica/extern/Voroxx/examples/custom/pack_six_cube_poly");



    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<int> newInd;

    return pack(*this, container);

}

using namespace voro;

// Golden ratio constants
const double Phi=0.5*(1+sqrt(5.0));
const double phi=0.5*(1-sqrt(5.0));


// Set up the number of blocks that the container is divided
// into.
const int n_x=5,n_y=5,n_z=5;


// Create a wall class that, whenever called, will replace the Voronoi cell
// with a prescribed shape, in this case a dodecahedron
class wall_initial_shape : public wall {
    public:
        wall_initial_shape() {

            // Create a dodecahedron
            v.init(-2,2,-2,2,-2,2);
            v.plane(0,Phi,1);v.plane(0,-Phi,1);v.plane(0,Phi,-1);
            v.plane(0,-Phi,-1);v.plane(1,0,Phi);v.plane(-1,0,Phi);
            v.plane(1,0,-Phi);v.plane(-1,0,-Phi);v.plane(Phi,1,0);
            v.plane(-Phi,1,0);v.plane(Phi,-1,0);v.plane(-Phi,-1,0);
        };
        bool point_inside(double x,double y,double z) {return true;}
        bool cut_cell(voronoicell &c,double x,double y,double z) {

            // Set the cell to be equal to the dodecahedron
            c=v;
            return true;
        }
        bool cut_cell(voronoicell_neighbor &c,double x,double y,double z) {

            // Set the cell to be equal to the dodecahedron
            c=v;
            return true;
        }
    private:
        voronoicell v;
};

bool MxMeshVoronoiImporter::irregular(MxMesh& mesh) {

    // Set up constants for the container geometry
    const double x_min=-6,x_max=6;
    const double y_min=-6,y_max=6;
    const double z_min=-3,z_max=9;

    // Create a container with the geometry given above. This is bigger
    // than the particle packing itself.
    container con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
            false,false,false,8);

    // Create the "initial shape" wall class and add it to the container
    wall_initial_shape(wis);
    con.add_wall(wis);

    // Import the irregular particle packing
    con.import("/Users/andy/src/mechanica/extern/Voroxx/examples/extra/pack_irregular");

    // Save the particles and Voronoi cells in POV-Ray format
    con.draw_particles_pov("irregular_p.pov");
    con.draw_cells_pov("irregular_v.pov");

    return pack(*this, con);
}


void MxMeshVoronoiImporter::createTriangleForCell(
        const std::array<VertexPtr, 3>& verts, CellPtr cell) {
    TrianglePtr tri = mesh.findTriangle(verts);
    if(tri) {
        if(::incident(tri, mesh.rootCell())) {
            assert(mesh.rootCell()->removeChild(tri) == S_OK);
        }
    }
    else {
        tri = mesh.createTriangle(nullptr, verts);
    }

    assert(tri);
    assert(tri->cells[0] == nullptr || tri->cells[1] == nullptr);

    Vector3 meshNorm = Math::normal(verts[0]->position, verts[1]->position, verts[2]->position);
    float orientation = Math::dot(meshNorm, tri->normal);

    int cellIndx = orientation > 0 ? 0 : 1;
    int rootIndx = (cellIndx+1)%2;
    cell->appendChild(&tri->partialTriangles[cellIndx]);
    mesh.rootCell()->appendChild(&tri->partialTriangles[rootIndx]);
}


