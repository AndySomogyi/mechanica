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
        const Magnum::Vector3i& n, const std::array<bool, 3> periodic,
        MxMesh& mesh)
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
        const Magnum::Vector3i& n, const std::array<bool, 3> periodic,
        MxMesh& mesh)
{
    auto meshData = readPoints(path);

    cout << "pos count: " << meshData->positionArrayCount() << std::endl;

    std::vector<Vector3> positions = meshData->positions(0);

    //container::container(double ax_,double bx_,double ay_,double by_,double az_,double bz_,
    //    int nx_,int ny_,int nz_,bool xperiodic_,bool yperiodic_,bool zperiodic_,int init_mem)
    voro::container container(min[0], max[0], min[1], max[1], min[2], max[2],
        n[0], n[1], n[2], periodic[0], periodic[1], periodic[2], 50);


    for (int i = 0; i < positions.size(); ++i)  {
        const Vector3& vec = positions[i];
        container.put(i++, vec[0], vec[1], vec[2]);
    }

    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<int> newInd;

    voro::c_loop_all vl(container);
    double *cellOrigin;
    voro::voronoicell c;
    if(vl.start()) do {
        if(container.compute_cell(c,vl)) {
            // the center of the voronoi cell, should be same as the
            // inserted particle location.
            cellOrigin = container.p[vl.ijk] + container.ps * vl.q;

            // grab the indexed vertices from the cell
            c.indexed_triangular_faces(cellOrigin[0], cellOrigin[1], cellOrigin[2], vertices, indices);

            // do these in-place once bugs are worked out
            newInd.resize(indices.size());

            for (int j = 0; j < indices.size(); ++j) {
                int k = indices[j];
                newInd[j] = mesh.appendVertex({float(vertices[3*k]), float(vertices[3*k+1]), float(vertices[3*k+2])});
            }

            // allocate first in the cells vector, then we write directly to that memory block,
            // avoid copy
            MxCell &cell = mesh.createCell();

            // add the faces to the cell, then sort out the connectivity
            for(int i = 0; i < newInd.size(); i+=3) {
                MxPartialFace pf = {{uint(newInd[i]), uint(newInd[i+1]), uint(newInd[i+2])}};
                cell.boundary.push_back(pf);
            }

            // boundary had better be connected from a voronoi cell
            bool connected = cell.connectBoundary();
            assert(connected);
        }
    } while(vl.inc());
    
    
    mesh.dump(0);

    return true;
}
