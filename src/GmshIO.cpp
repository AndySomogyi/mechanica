// The libMesh Finite Element Library.
// Copyright (C) 2002-2017 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

// C++ includes
#include <GmshIO.h>
#include <fstream>
#include <set>
#include <cstring> // std::memcpy
#include <numeric>
#include <cassert>
#include <sstream>
#include <iostream>

// Local includes

namespace Gmsh {

Element e(Line{});

Element q(Quadrangle{});

#define libmesh_error_msg(STUFF)


Mesh Mesh::read(const std::string &path) {
    std::ifstream in (path.c_str());
    return read(in);
}


// TODO, add range checking for the number.
ElementType elementType(int val) {
    return static_cast<ElementType>(val);
}

template<typename ElmType> Element createElement(int id, int physical, int elementary, const std::vector<int> &nodes) {
    if (nodes.size() != ElmType::NodeSize) {
        assert(0 && "wrong node size for element type");
    }

    ElmType elm{id, physical, elementary};
    std::copy(nodes.begin(), nodes.end(), elm.nodes);

    return Element(elm);
}


// As of version 2.2, the format for each element line is:
// elm-number elm-type number-of-tags < tag > ... node-number-list
// From the Gmsh docs:
// * the first tag is the number of the
//   physical entity to which the element belongs
// * the second is the number of the elementary geometrical
//   entity to which the element belongs
// * the third is the number of mesh partitions to which the element
//   belongs
// * The rest of the tags are the partition ids (negative
//   partition ids indicate ghost cells). A zero tag is
//   equivalent to no tag. Gmsh and most codes using the
//   MSH 2 format require at least the first two tags
//   (physical and elementary tags).
Element Element::read(double version, const std::string &line) {
    std::stringstream iss( line );
    std::vector<int> nodes;

    unsigned int
    id, type,
    physical=1, elementary=1,
    nnodes=0, ntags;

    // Note: tag has to be an int because it could be negative,
    // see above.
    int tag;

    if (version <= 1.0)
        iss >> id >> type >> physical >> elementary >> nnodes;

    else
    {
        iss >> id >> type >> ntags;

        if (ntags > 2) {
            //libmesh_do_once(libMesh::err << "Warning, ntags=" << ntags << ", but we currently only support reading 2 flags." << std::endl;);
        }
        for (unsigned int j = 0; j < ntags; j++)
        {
            iss >> tag;
            if (j == 0)
                physical = tag;
            else if (j == 1)
                elementary = tag;
        }
    }

    // grab all the nodes
    int node;
    while ( iss >> node ) { nodes.push_back( node );}

    #define GMSH_CASE_CREATE(TYPE) case ElementType::TYPE : return createElement<TYPE>(id, physical, elementary, nodes);

    switch(static_cast<ElementType>(type)) {
    GMSH_CASE_CREATE(Line);
    GMSH_CASE_CREATE(Triangle);
    GMSH_CASE_CREATE(Quadrangle);
    GMSH_CASE_CREATE(Tetrahedron);
    GMSH_CASE_CREATE(Hexahedron);
    GMSH_CASE_CREATE(Prism);
    GMSH_CASE_CREATE(Line3);
    GMSH_CASE_CREATE(Triangle6);
    GMSH_CASE_CREATE(Quadrangle9);
    GMSH_CASE_CREATE(Tetrahedron10);
    GMSH_CASE_CREATE(Hexadedron27);
    GMSH_CASE_CREATE(Prism18);
    GMSH_CASE_CREATE(Pyramid14);
    GMSH_CASE_CREATE(Point);
    GMSH_CASE_CREATE(Quadrangle8);
    GMSH_CASE_CREATE(Hexadedron20);
    GMSH_CASE_CREATE(Prism15);
    GMSH_CASE_CREATE(Pyramid13);
    GMSH_CASE_CREATE(TriangleInc9);
    GMSH_CASE_CREATE(Triangle10);
    GMSH_CASE_CREATE(TrianbleInc12);
    GMSH_CASE_CREATE(Triangle15);
    GMSH_CASE_CREATE(TriangleInc15);
    GMSH_CASE_CREATE(Triangle21);
    GMSH_CASE_CREATE(Edge4);
    GMSH_CASE_CREATE(Edge5);
    GMSH_CASE_CREATE(Edge6);
    GMSH_CASE_CREATE(Tetrahedron20);
    GMSH_CASE_CREATE(Tetrahedron35);
    GMSH_CASE_CREATE(Tetrahedron56);
    GMSH_CASE_CREATE(Hexahedron64);
    GMSH_CASE_CREATE(Hexahedron125);
    default:
        throw std::domain_error("Error, " + std::to_string(type) + " is not a valid Gmsh element type");
    }

    #undef GMSH_CASE_CREATE
}

void test() {
    Line line{0, 0, 0, {0, 1}};

    Element e{line};

    Element e2 = createElement<Line>(0, 0, 0, {0, 0});

    std::cout << e.id();
}


#if 0
// Initialize the static data member
GmshIO::ElementMaps GmshIO::_element_maps = GmshIO::build_element_maps();



// Definition of the static function which constructs the ElementMaps object.
GmshIO::ElementMaps GmshIO::build_element_maps()
{
  // Object to be filled up
  ElementMaps em;

  // POINT (import only)
  em.in.insert(std::make_pair(15, ElementDefinition(NODEELEM, 15, 0, 1)));

  // Add elements with trivial node mappings
  em.add_def(ElementDefinition(EDGE2, 1, 1, 2));
  em.add_def(ElementDefinition(EDGE3, 8, 1, 3));
  em.add_def(ElementDefinition(TRI3, 2, 2, 3));
  em.add_def(ElementDefinition(TRI6, 9, 2, 6));
  em.add_def(ElementDefinition(QUAD4, 3, 2, 4));
  em.add_def(ElementDefinition(QUAD8, 16, 2, 8));
  em.add_def(ElementDefinition(QUAD9, 10, 2, 9));
  em.add_def(ElementDefinition(HEX8, 5, 3, 8));
  em.add_def(ElementDefinition(TET4, 4, 3, 4));
  em.add_def(ElementDefinition(PRISM6, 6, 3, 6));
  em.add_def(ElementDefinition(PYRAMID5, 7, 3, 5));

  // Add elements with non-trivial node mappings

  // HEX20
  {
    ElementDefinition eledef(HEX20, 17, 3, 20);
    const unsigned int nodes[] = {0,1,2,3,4,5,6,7,8,11,12,9,13,10,14,15,16,19,17,18};
    std::vector<unsigned int>(nodes, nodes+eledef.nnodes).swap(eledef.nodes); // swap trick
    em.add_def(eledef);
  }

  // HEX27
  {
    ElementDefinition eledef(HEX27, 12, 3, 27);
    const unsigned int nodes[] = {0,1,2,3,4,5,6,7,8,11,12,9,13,10,14,
                                  15,16,19,17,18,20,21,24,22,23,25,26};
    std::vector<unsigned int>(nodes, nodes+eledef.nnodes).swap(eledef.nodes); // swap trick
    em.add_def(eledef);
  }

  // TET10
  {
    ElementDefinition eledef(TET10, 11, 3, 10);
    const unsigned int nodes[] = {0,1,2,3,4,5,6,7,9,8};
    std::vector<unsigned int>(nodes, nodes+eledef.nnodes).swap(eledef.nodes); // swap trick
    em.add_def(eledef);
  }

  // PRISM15
  {
    ElementDefinition eledef(PRISM15, 18, 3, 15);
    const unsigned int nodes[] = {0,1,2,3,4,5,6,8,9,7,10,11,12,14,13};
    std::vector<unsigned int>(nodes, nodes+eledef.nnodes).swap(eledef.nodes); // swap trick
    em.add_def(eledef);
  }

  // PRISM18
  {
    ElementDefinition eledef(PRISM18, 13, 3, 18);
    const unsigned int nodes[] = {0,1,2,3,4,5,6,8,9,7,10,11,12,14,13,15,17,16};
    std::vector<unsigned int>(nodes, nodes+eledef.nnodes).swap(eledef.nodes); // swap trick
    em.add_def(eledef);
  }

  return em;
}



GmshIO::GmshIO (const MeshBase & mesh) :
  MeshOutput<MeshBase>(mesh),
  _binary(false),
  _write_lower_dimensional_elements(true)
{
}



GmshIO::GmshIO (MeshBase & mesh) :
  MeshInput<MeshBase>  (mesh),
  MeshOutput<MeshBase> (mesh),
  _binary (false),
  _write_lower_dimensional_elements(true)
{
}



bool & GmshIO::binary ()
{
  return _binary;
}

#endif


#if 0
bool & GmshIO::write_lower_dimensional_elements ()
{
  return _write_lower_dimensional_elements;
}
#endif

#if 0
void GmshIO::read (const std::string & name)
{
  std::ifstream in (name.c_str());
  this->read_mesh (in);
}
#endif



Mesh Mesh::read(std::istream & in)
{
    Mesh mesh;

    assert(in.good());

    // some variables
    int format=0, size=0;
    double version = 1.0;


    // For reading the file line by line
    std::string s;

    while (!in.eof())
    {
        // Try to read something.  This may set EOF!
        std::getline(in, s);

        // Process s...

        if (s.find("$MeshFormat") == static_cast<std::string::size_type>(0))
        {
            in >> version >> format >> size;
            if ((version != 2.0) && (version != 2.1) && (version != 2.2))
            {
                // Some notes on gmsh mesh versions:
                //
                // Mesh version 2.0 goes back as far as I know.  It's not explicitly
                // mentioned here: http://www.geuz.org/gmsh/doc/VERSIONS.txt
                //
                // As of gmsh-2.4.0:
                // bumped mesh version format to 2.1 (small change in the $PhysicalNames
                // section, where the group dimension is now required);
                // [Since we don't even parse the PhysicalNames section at the time
                //  of this writing, I don't think this change affects us.]
                //
                // Mesh version 2.2 tested by Manav Bhatia; no other
                // libMesh code changes were required for support
                libmesh_error_msg("Error: Unknown msh file version " << version);
            }

            if (format)
                libmesh_error_msg("Error: Unknown data format for mesh in Gmsh reader.");
        }

        // Read and process the "PhysicalNames" section.
        else if (s.find("$PhysicalNames") == static_cast<std::string::size_type>(0))
        {
            // The lines in the PhysicalNames section should look like the following:
            // 2 1 "frac" lower_dimensional_block
            // 2 3 "top"
            // 2 4 "bottom"
            // 3 2 "volume"

            // Read in the number of physical groups to expect in the file.
            unsigned int num_physical_groups = 0;
            in >> num_physical_groups;

            // Read rest of line including newline character.
            std::getline(in, s);

            for (unsigned int i=0; i<num_physical_groups; ++i)
            {
                // Read an entire line of the PhysicalNames section.
                std::getline(in, s);

                // Use an istringstream to extract the physical
                // dimension, physical id, and physical name from
                // this line.
                std::istringstream s_stream(s);
                unsigned phys_dim;
                int phys_id;
                std::string phys_name;
                s_stream >> phys_dim >> phys_id >> phys_name;

                // Not sure if this is true for all Gmsh files, but
                // my test file has quotes around the phys_name
                // string.  So let's erase any quotes now...
                phys_name.erase(std::remove(phys_name.begin(), phys_name.end(), '"'), phys_name.end());

                // Record this ID for later assignment of subdomain/sideset names.
                mesh.physicals[phys_id] = Physical{phys_dim, phys_name};

                // If 's' also contains the libmesh-specific string
                // "lower_dimensional_block", add this block ID to
                // the list of blocks which are not boundary
                // conditions.

                //                  if (s.find("lower_dimensional_block") != std::string::npos)
                //                    {
                //                      lower_dimensional_blocks.insert(cast_int<subdomain_id_type>(phys_id));
                //
                //                      // The user has explicitly told us that this
                //                      // block is a subdomain, so set that association
                //                      // in the Mesh.
                //                      mesh.subdomain_name(cast_int<subdomain_id_type>(phys_id)) = phys_name;
                //                    }
            }
        }

        // read the node block
        else if (s.find("$NOD") == static_cast<std::string::size_type>(0) ||
                s.find("$NOE") == static_cast<std::string::size_type>(0) ||
                s.find("$Nodes") == static_cast<std::string::size_type>(0))
        {
            unsigned int num_nodes = 0;
            in >> num_nodes;
            mesh.nodes.reserve(num_nodes);

            // read in the nodal coordinates and form points.
            double x, y, z;
            unsigned int id;

            // add the nodal coordinates to the mesh
            for (int i=0; i<num_nodes; ++i)
            {
                in >> id >> x >> y >> z;
                mesh.nodes[id] = Node{id, {x, y, z}};
            }

            // read the $ENDNOD delimiter
            std::getline(in, s);
        }


        // Read the element block
        else if (s.find("$ELM") == static_cast<std::string::size_type>(0) ||
                s.find("$Elements") == static_cast<std::string::size_type>(0))
        {
            // For reading the number of elements and the node ids from the stream
            unsigned int num_elem = 0;

            // read how many elements are there, and reserve space in the mesh
            in >> num_elem;
            mesh.elements.reserve(num_elem);

            // read the trailing EOL
            std::getline(in, s);
            if (s.length() != 0) {
                throw std::domain_error("Error, found junk after \"$Elements\"");
            }

            // read the elements
            for (unsigned int iel=0; iel<num_elem; ++iel)
            {
                // read the line for the element
                std::getline(in, s);

                Element element = Element::read(version, s);

                mesh.elements.push_back(element);
            } // element loop

            // read the $ENDELM delimiter
            std::getline(in, s);

            // TODO verify the ENDELM delimiter
        }
    } // while !eof()
    return mesh;
}



#if 0
void GmshIO::write (const std::string & name)
{
  if (MeshOutput<MeshBase>::mesh().processor_id() == 0)
    {
      // Open the output file stream
      std::ofstream out_stream (name.c_str());

      // Make sure it opened correctly
      if (!out_stream.good())
        libmesh_file_error(name.c_str());

      this->write_mesh (out_stream);
    }
}
#endif




#if 0
void GmshIO::write_nodal_data (const std::string & fname,
                               const std::vector<Number> & soln,
                               const std::vector<std::string> & names)
{
  LOG_SCOPE("write_nodal_data()", "GmshIO");

  if (MeshOutput<MeshBase>::mesh().processor_id() == 0)
    this->write_post  (fname, &soln, &names);
}
#endif


#if 0
void GmshIO::write_mesh (std::ostream & out_stream)
{
  // Be sure that the stream is valid.
  libmesh_assert (out_stream.good());

  // Get a const reference to the mesh
  const MeshBase & mesh = MeshOutput<MeshBase>::mesh();

  // If requested, write out lower-dimensional elements for
  // element-side-based boundary conditions.
  unsigned int n_boundary_faces = 0;
  if (this->write_lower_dimensional_elements())
    n_boundary_faces = mesh.get_boundary_info().n_boundary_conds();

  // Note: we are using version 2.0 of the gmsh output format.

  // Write the file header.
  out_stream << "$MeshFormat\n";
  out_stream << "2.0 0 " << sizeof(Real) << '\n';
  out_stream << "$EndMeshFormat\n";

  // write the nodes in (n x y z) format
  out_stream << "$Nodes\n";
  out_stream << mesh.n_nodes() << '\n';

  for (unsigned int v=0; v<mesh.n_nodes(); v++)
    out_stream << mesh.node_ref(v).id()+1 << " "
               << mesh.node_ref(v)(0) << " "
               << mesh.node_ref(v)(1) << " "
               << mesh.node_ref(v)(2) << '\n';
  out_stream << "$EndNodes\n";

  {
    // write the connectivity
    out_stream << "$Elements\n";
    out_stream << mesh.n_active_elem() + n_boundary_faces << '\n';

    MeshBase::const_element_iterator       it  = mesh.active_elements_begin();
    const MeshBase::const_element_iterator end = mesh.active_elements_end();

    // loop over the elements
    for ( ; it != end; ++it)
      {
        const Elem * elem = *it;

        // Make sure we have a valid entry for
        // the current element type.
        libmesh_assert (_element_maps.out.count(elem->type()));

        // consult the export element table
        std::map<ElemType, ElementDefinition>::iterator def_it =
          _element_maps.out.find(elem->type());

        // Assert that we found it
        if (def_it == _element_maps.out.end())
          libmesh_error_msg("Element type " << elem->type() << " not found in _element_maps.");

        // Get a reference to the ElementDefinition object
        const ElementDefinition & eletype = def_it->second;

        // The element mapper better not require any more nodes
        // than are present in the current element!
        libmesh_assert_less_equal (eletype.nodes.size(), elem->n_nodes());

        // elements ids are 1 based in Gmsh
        out_stream << elem->id()+1 << " ";
        // element type
        out_stream << eletype.gmsh_type;

        // write the number of tags (3) and their values:
        // 1 (physical entity)
        // 2 (geometric entity)
        // 3 (partition entity)
        out_stream << " 3 "
                   << static_cast<unsigned int>(elem->subdomain_id())
                   << " 0 "
                   << elem->processor_id()+1
                   << " ";

        // if there is a node translation table, use it
        if (eletype.nodes.size() > 0)
          for (unsigned int i=0; i < elem->n_nodes(); i++)
            out_stream << elem->node_id(eletype.nodes[i])+1 << " "; // gmsh is 1-based
        // otherwise keep the same node order
        else
          for (unsigned int i=0; i < elem->n_nodes(); i++)
            out_stream << elem->node_id(i)+1 << " ";                  // gmsh is 1-based
        out_stream << "\n";
      } // element loop
  }

  {
    // A counter for writing surface elements to the Gmsh file
    // sequentially.  We start numbering them with a number strictly
    // larger than the largest element ID in the mesh.  Note: the
    // MeshBase docs say "greater than or equal to" the maximum
    // element id in the mesh, so technically we might need a +1 here,
    // but all of the implementations return an ID strictly greater
    // than the largest element ID in the Mesh.
    unsigned int e_id = mesh.max_elem_id();

    // loop over the elements, writing out boundary faces
    MeshBase::const_element_iterator       it  = mesh.active_elements_begin();
    const MeshBase::const_element_iterator end = mesh.active_elements_end();

    if (n_boundary_faces)
      {
        // Construct the list of boundary sides
        std::vector<dof_id_type> element_id_list;
        std::vector<unsigned short int> side_list;
        std::vector<boundary_id_type> bc_id_list;

        mesh.get_boundary_info().build_side_list(element_id_list, side_list, bc_id_list);

        // Loop over these lists, writing data to the file.
        for (std::size_t idx=0; idx<element_id_list.size(); ++idx)
          {
            const Elem & elem = mesh.elem_ref(element_id_list[idx]);

            UniquePtr<const Elem> side = elem.build_side_ptr(side_list[idx]);

            // Map from libmesh elem type to gmsh elem type.
            std::map<ElemType, ElementDefinition>::iterator def_it =
              _element_maps.out.find(side->type());

            // If we didn't find it, that's an error
            if (def_it == _element_maps.out.end())
              libmesh_error_msg("Element type " << side->type() << " not found in _element_maps.");

            // consult the export element table
            const GmshIO::ElementDefinition & eletype = def_it->second;

            // The element mapper better not require any more nodes
            // than are present in the current element!
            libmesh_assert_less_equal (eletype.nodes.size(), side->n_nodes());

            // elements ids are 1-based in Gmsh
            out_stream << e_id+1 << " ";

            // element type
            out_stream << eletype.gmsh_type;

            // write the number of tags:
            // 1 (physical entity)
            // 2 (geometric entity)
            // 3 (partition entity)
            out_stream << " 3 "
                       << bc_id_list[idx]
                       << " 0 "
                       << elem.processor_id()+1
                       << " ";

            // if there is a node translation table, use it
            if (eletype.nodes.size() > 0)
              for (unsigned int i=0; i < side->n_nodes(); i++)
                out_stream << side->node_id(eletype.nodes[i])+1 << " "; // gmsh is 1-based

            // otherwise keep the same node order
            else
              for (unsigned int i=0; i < side->n_nodes(); i++)
                out_stream << side->node_id(i)+1 << " ";                // gmsh is 1-based

            // Go to the next line
            out_stream << "\n";

            // increment this index too...
            ++e_id;
          }
      }

    out_stream << "$EndElements\n";
  }
}
#endif




#if 0
void GmshIO::write_post (const std::string & fname,
                         const std::vector<Number> * v,
                         const std::vector<std::string> * solution_names)
{

  // Should only do this on processor 0!
  libmesh_assert_equal_to (MeshOutput<MeshBase>::mesh().processor_id(), 0);

  // Create an output stream
  std::ofstream out_stream(fname.c_str());

  // Make sure it opened correctly
  if (!out_stream.good())
    libmesh_file_error(fname.c_str());

  // create a character buffer
  char buf[80];

  // Get a constant reference to the mesh.
  const MeshBase & mesh = MeshOutput<MeshBase>::mesh();

  //  write the data
  if ((solution_names != libmesh_nullptr) && (v != libmesh_nullptr))
    {
      const unsigned int n_vars =
        cast_int<unsigned int>(solution_names->size());

      if (!(v->size() == mesh.n_nodes()*n_vars))
        libMesh::err << "ERROR: v->size()=" << v->size()
                     << ", mesh.n_nodes()=" << mesh.n_nodes()
                     << ", n_vars=" << n_vars
                     << ", mesh.n_nodes()*n_vars=" << mesh.n_nodes()*n_vars
                     << "\n";

      libmesh_assert_equal_to (v->size(), mesh.n_nodes()*n_vars);

      // write the header
      out_stream << "$PostFormat\n";
      if (this->binary())
        out_stream << "1.2 1 " << sizeof(double) << "\n";
      else
        out_stream << "1.2 0 " << sizeof(double) << "\n";
      out_stream << "$EndPostFormat\n";

      // Loop over the elements to see how much of each type there are
      unsigned int n_points=0, n_lines=0, n_triangles=0, n_quadrangles=0,
        n_tetrahedra=0, n_hexahedra=0, n_prisms=0, n_pyramids=0;
      unsigned int n_scalar=0, n_vector=0, n_tensor=0;
      unsigned int nb_text2d=0, nb_text2d_chars=0, nb_text3d=0, nb_text3d_chars=0;

      {
        MeshBase::const_element_iterator       it  = mesh.active_elements_begin();
        const MeshBase::const_element_iterator end = mesh.active_elements_end();


        for ( ; it != end; ++it)
          {
            const ElemType elemtype = (*it)->type();

            switch (elemtype)
              {
              case EDGE2:
              case EDGE3:
              case EDGE4:
                {
                  n_lines += 1;
                  break;
                }
              case TRI3:
              case TRI6:
                {
                  n_triangles += 1;
                  break;
                }
              case QUAD4:
              case QUAD8:
              case QUAD9:
                {
                  n_quadrangles += 1;
                  break;
                }
              case TET4:
              case TET10:
                {
                  n_tetrahedra += 1;
                  break;
                }
              case HEX8:
              case HEX20:
              case HEX27:
                {
                  n_hexahedra += 1;
                  break;
                }
              case PRISM6:
              case PRISM15:
              case PRISM18:
                {
                  n_prisms += 1;
                  break;
                }
              case PYRAMID5:
                {
                  n_pyramids += 1;
                  break;
                }
              default:
                libmesh_error_msg("ERROR: Nonexistent element type " << (*it)->type());
              }
          }
      }

      // create a view for each variable
      for (unsigned int ivar=0; ivar < n_vars; ivar++)
        {
          std::string varname = (*solution_names)[ivar];

          // at the moment, we just write out scalar quantities
          // later this should be made configurable through
          // options to the writer class
          n_scalar = 1;

          // write the variable as a view, and the number of time steps
          out_stream << "$View\n" << varname << " " << 1 << "\n";

          // write how many of each geometry type are written
          out_stream << n_points * n_scalar << " "
                     << n_points * n_vector << " "
                     << n_points * n_tensor << " "
                     << n_lines * n_scalar << " "
                     << n_lines * n_vector << " "
                     << n_lines * n_tensor << " "
                     << n_triangles * n_scalar << " "
                     << n_triangles * n_vector << " "
                     << n_triangles * n_tensor << " "
                     << n_quadrangles * n_scalar << " "
                     << n_quadrangles * n_vector << " "
                     << n_quadrangles * n_tensor << " "
                     << n_tetrahedra * n_scalar << " "
                     << n_tetrahedra * n_vector << " "
                     << n_tetrahedra * n_tensor << " "
                     << n_hexahedra * n_scalar << " "
                     << n_hexahedra * n_vector << " "
                     << n_hexahedra * n_tensor << " "
                     << n_prisms * n_scalar << " "
                     << n_prisms * n_vector << " "
                     << n_prisms * n_tensor << " "
                     << n_pyramids * n_scalar << " "
                     << n_pyramids * n_vector << " "
                     << n_pyramids * n_tensor << " "
                     << nb_text2d << " "
                     << nb_text2d_chars << " "
                     << nb_text3d << " "
                     << nb_text3d_chars << "\n";

          // if binary, write a marker to identify the endianness of the file
          if (this->binary())
            {
              const int one = 1;
              std::memcpy(buf, &one, sizeof(int));
              out_stream.write(buf, sizeof(int));
            }

          // the time steps (there is just 1 at the moment)
          if (this->binary())
            {
              double one = 1;
              std::memcpy(buf, &one, sizeof(double));
              out_stream.write(buf, sizeof(double));
            }
          else
            out_stream << "1\n";

          // Loop over the elements and write out the data
          MeshBase::const_element_iterator       it  = mesh.active_elements_begin();
          const MeshBase::const_element_iterator end = mesh.active_elements_end();

          for ( ; it != end; ++it)
            {
              const Elem * elem = *it;

              // this is quite crappy, but I did not invent that file format!
              for (unsigned int d=0; d<3; d++)  // loop over the dimensions
                {
                  for (unsigned int n=0; n < elem->n_vertices(); n++)   // loop over vertices
                    {
                      const Point & vertex = elem->point(n);
                      if (this->binary())
                        {
                          double tmp = vertex(d);
                          std::memcpy(buf, &tmp, sizeof(double));
                          out_stream.write(reinterpret_cast<char *>(buf), sizeof(double));
                        }
                      else
                        out_stream << vertex(d) << " ";
                    }
                  if (!this->binary())
                    out_stream << "\n";
                }

              // now finally write out the data
              for (unsigned int i=0; i < elem->n_vertices(); i++)   // loop over vertices
                if (this->binary())
                  {
#ifdef LIBMESH_USE_COMPLEX_NUMBERS
                    libMesh::out << "WARNING: Gmsh::write_post does not fully support "
                                 << "complex numbers. Will only write the real part of "
                                 << "variable " << varname << std::endl;
#endif
                    double tmp = libmesh_real((*v)[elem->node_id(i)*n_vars + ivar]);
                    std::memcpy(buf, &tmp, sizeof(double));
                    out_stream.write(reinterpret_cast<char *>(buf), sizeof(double));
                  }
                else
                  {
#ifdef LIBMESH_USE_COMPLEX_NUMBERS
                    libMesh::out << "WARNING: Gmsh::write_post does not fully support "
                                 << "complex numbers. Will only write the real part of "
                                 << "variable " << varname << std::endl;
#endif
                    out_stream << libmesh_real((*v)[elem->node_id(i)*n_vars + ivar]) << "\n";
                  }
            }
          if (this->binary())
            out_stream << "\n";
          out_stream << "$EndView\n";

        } // end variable loop (writing the views)
    }
}
#endif

}

