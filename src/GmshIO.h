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



#ifndef LIBMESH_GMSH_IO_H
#define LIBMESH_GMSH_IO_H

// Local includes
#include "mechanica_private.h"

// C++ includes
#include <cstddef>
#include <vector>
#include <array>
#include <unordered_map>
#include <string>


namespace Gmsh {

enum class ElementType : std::int8_t {
    Invalid =         -1,  // invalid type
    Line =             1,  // 2-node line.
    Triangle =         2,  // 3-node triangle.
    Quadrangle =       3,  // 4-node quadrangle.
    Tetrahedron =      4,  // 4-node tetrahedron.
    Hexahedron =       5,  // 8-node hexahedron.
    Prism =            6,  // 6-node prism.
    Line3 =            8,  // 3-node second order line
                           // (2 nodes associated with the vertices and 1 with the edge).
    Triangle6 =        9,  // 6-node second order triangle
                           // (3 nodes associated with the vertices and 3 with the edges).
    Quadrangle9 =      10, // 9-node second order quadrangle
                           // (4 nodes associated with the vertices, 4 with the edges and 1 with the face).
    Tetrahedron10 =    11, // 10-node second order tetrahedron
                           // (4 nodes associated with the vertices and 6 with the edges).
    Hexadedron27 =     12, // 27-node second order hexahedron
                           // (8 nodes associated with the vertices, 12 with the edges,
                           // 6 with the faces and 1 with the volume).
    Prism18 =          13, // 18-node second order prism (6 nodes associated with the vertices,
                           // 9 with the edges and 3 with the quadrangular faces).
    Pyramid14 =        14, // 14-node second order pyramid (5 nodes associated with the vertices,
                           // 8 with the edges and 1 with the quadrangular face).
    Point =            15, // 1-node point.
    Quadrangle8 =      16, // 8-node second order quadrangle (4 nodes associated with the
                           // vertices and 4 with the edges).
    Hexadedron20 =     17, // 20-node second order hexahedron (8 nodes associated with
                           // the vertices and 12 with the edges).
    Prism15 =          18, // 15-node second order prism (6 nodes associated with the
                           // vertices and 9 with the edges).
    Pyramid13 =        19, // 13-node second order pyramid (5 nodes associated with the
                           // vertices and 8 with the edges).
    TriangleInc9 =     20, // 9-node third order incomplete triangle (3 nodes associated
                           // with the vertices, 6 with the edges)
    Triangle10 =       21, // 10-node third order triangle (3 nodes associated with the
                           // vertices, 6 with the edges, 1 with the face)
    TrianbleInc12 =    22, //12-node fourth order incomplete triangle (3 nodes associated
                           // with the vertices, 9 with the edges)
    Triangle15 =       23, // 15-node fourth order triangle (3 nodes associated with the
                           // vertices, 9 with the edges, 3 with the face)
    TriangleInc15 =    24, // 15-node fifth order incomplete triangle (3 nodes associated
                           // with the vertices, 12 with the edges)
    Triangle21 =       25, // 21-node fifth order complete triangle (3 nodes associated
                           // with the vertices, 12 with the edges, 6 with the face)
    Edge4 =            26, // 4-node third order edge (2 nodes associated with the
                           // vertices, 2 internal to the edge)
    Edge5 =            27, // 5-node fourth order edge (2 nodes associated with the
                           // vertices, 3 internal to the edge)
    Edge6 =            28, // 6-node fifth order edge (2 nodes associated with the
                           // vertices, 4 internal to the edge)
    Tetrahedron20 =    29, // 20-node third order tetrahedron (4 nodes associated with the
                           // vertices, 12 with the edges, 4 with the faces)
    Tetrahedron35 =    30, // 35-node fourth order tetrahedron (4 nodes associated with the
                           // vertices, 18 with the edges, 12 with the faces, 1 in the volume)
    Tetrahedron56 =    31, // 56-node fifth order tetrahedron (4 nodes associated with the
                           // vertices, 24 with the edges, 24 with the faces, 4 in the volume)
    Hexahedron64 =     92, // 64-node third order hexahedron (8 nodes associated with the
                           // vertices, 24 with the edges, 24 with the faces, 8 in the volume)
    Hexahedron125 =    93, // 125-node fourth order hexahedron (8 nodes associated with the
                           // vertices, 36 with the edges, 54 with the faces, 27 in the volume)

};

#define GMSH_DEFINE_ELEMENT(NAME, DIMENSION, NODESIZE)  \
struct NAME {                                      \
    static const unsigned Dimension = DIMENSION;   \
    static const unsigned NodeSize = NODESIZE;     \
    int id;                                        \
    int physical;                                  \
    int elementary;                                \
    int nodes[NodeSize];                           \
};

GMSH_DEFINE_ELEMENT(Line, 2, 2);            // 2-node line.
GMSH_DEFINE_ELEMENT(Triangle, 2, 3);        // 3-node triangle.
GMSH_DEFINE_ELEMENT(Quadrangle, 2, 4);      // 4-node quadrangle.
GMSH_DEFINE_ELEMENT(Tetrahedron, 3, 4);     // 4-node tetrahedron.
GMSH_DEFINE_ELEMENT(Hexahedron, 3, 8);      // 8-node hexahedron.
GMSH_DEFINE_ELEMENT(Prism, 3, 6);           // 6-node prism.
GMSH_DEFINE_ELEMENT(Line3, 2, 3);           // 3-node second order line
                                            // (2 nodes associated with the vertices and 1 with the edge).
GMSH_DEFINE_ELEMENT(Triangle6, 2, 6);       // 6-node second order triangle
                                            // (3 nodes associated with the vertices and 3 with the edges).
GMSH_DEFINE_ELEMENT(Quadrangle9, 2, 9);     // 9-node second order quadrangle
                                            // (4 nodes associated with the vertices, 4 with the edges and 1 with the face).
GMSH_DEFINE_ELEMENT(Tetrahedron10, 3, 10);  // 10-node second order tetrahedron
                                            // (4 nodes associated with the vertices and 6 with the edges).
GMSH_DEFINE_ELEMENT(Hexadedron27, 3, 27);   // 27-node second order hexahedron
                                            // (8 nodes associated with the vertices, 12 with the edges,
                                            // 6 with the faces and 1 with the volume).
GMSH_DEFINE_ELEMENT(Prism18, 3, 18);        // 18-node second order prism (6 nodes associated with the vertices,
                                            // 9 with the edges and 3 with the quadrangular faces).
GMSH_DEFINE_ELEMENT(Pyramid14, 3, 14);      // 14-node second order pyramid (5 nodes associated with the vertices,
                                            // 8 with the edges and 1 with the quadrangular face).
GMSH_DEFINE_ELEMENT(Point, 0, 1);           // 1-node point.
GMSH_DEFINE_ELEMENT(Quadrangle8, 2, 8);     // 8-node second order quadrangle (4 nodes associated with the
                                            // vertices and 4 with the edges).
GMSH_DEFINE_ELEMENT(Hexadedron20, 3, 20);   // 20-node second order hexahedron (8 nodes associated with
                                            // the vertices and 12 with the edges).
GMSH_DEFINE_ELEMENT(Prism15, 3, 15);        // 15-node second order prism (6 nodes associated with the
                                            // vertices and 9 with the edges).
GMSH_DEFINE_ELEMENT(Pyramid13, 3, 13);      // 13-node second order pyramid (5 nodes associated with the
                                            // vertices and 8 with the edges).
GMSH_DEFINE_ELEMENT(TriangleInc9, 2, 9);    // 9-node third order incomplete triangle (3 nodes associated
                                            // with the vertices, 6 with the edges)
GMSH_DEFINE_ELEMENT(Triangle10, 2, 10);     // 10-node third order triangle (3 nodes associated with the
                                            // vertices, 6 with the edges, 1 with the face)
GMSH_DEFINE_ELEMENT(TrianbleInc12, 2, 12);  // 12-node fourth order incomplete triangle (3 nodes associated
                                            // with the vertices, 9 with the edges)
GMSH_DEFINE_ELEMENT(Triangle15, 2, 15);     // 15-node fourth order triangle (3 nodes associated with the
                                            // vertices, 9 with the edges, 3 with the face)
GMSH_DEFINE_ELEMENT(TriangleInc15, 2, 15);  // 15-node fifth order incomplete triangle (3 nodes associated
                                            // with the vertices, 12 with the edges)
GMSH_DEFINE_ELEMENT(Triangle21, 2, 21);     // 21-node fifth order complete triangle (3 nodes associated
                                            // with the vertices, 12 with the edges, 6 with the face)
GMSH_DEFINE_ELEMENT(Edge4, 1, 4);           // 4-node third order edge (2 nodes associated with the
                                            // vertices, 2 internal to the edge)
GMSH_DEFINE_ELEMENT(Edge5, 1, 5);           // 5-node fourth order edge (2 nodes associated with the
                                            // vertices, 3 internal to the edge)
GMSH_DEFINE_ELEMENT(Edge6, 1, 6);           // 6-node fifth order edge (2 nodes associated with the
                                            // vertices, 4 internal to the edge)
GMSH_DEFINE_ELEMENT(Tetrahedron20, 3, 20);  // 20-node third order tetrahedron (4 nodes associated with the
                                            // vertices, 12 with the edges, 4 with the faces)
GMSH_DEFINE_ELEMENT(Tetrahedron35, 3, 35);  // 35-node fourth order tetrahedron (4 nodes associated with the
                                            // vertices, 18 with the edges, 12 with the faces, 1 in the volume)
GMSH_DEFINE_ELEMENT(Tetrahedron56, 3, 56);  // 56-node fifth order tetrahedron (4 nodes associated with the
                                            // vertices, 24 with the edges, 24 with the faces, 4 in the volume)
GMSH_DEFINE_ELEMENT(Hexahedron64, 3, 64);   // 64-node third order hexahedron (8 nodes associated with the
                                            // vertices, 24 with the edges, 24 with the faces, 8 in the volume)
GMSH_DEFINE_ELEMENT(Hexahedron125, 3, 125); // 125-node fourth order hexahedron (8 nodes associated with the
                                            // vertices, 36 with the edges, 54 with the faces, 27 in the volume)
#undef GMSH_DEFINE_ELEMENT

struct Physical {
    unsigned dim;
    std::string name;
};

#define GMSH_ELEMENT_ITEM(TYPE) \
    explicit Element(const TYPE& val) : type{ElementType::TYPE} { init (val_ ## TYPE, val); }

#define GMSH_ELEMENT_GET(TYPE) template<> inline const TYPE& Element::get<TYPE>() { return val_ ## TYPE; };

#define GMSH_UNION(TYPE) TYPE val_ ## TYPE

#define GMSH_ELEMENT_COPY_CTOR(TYPE) \
case ElementType::TYPE: val_ ## TYPE = other.val_ ## TYPE; break;

struct Element {
    const ElementType type;

    template <typename T> const T& get();

    Element(const Element &other) : type{other.type} {
        copy(other);
    };

    Element& operator=(const Element& other)
    {
        // Nothing to do in case of self-assignment
        if (&other != this)
        {
            new (this) Element(other);
        }
        return *this;
    }

    Element() : type{ElementType::Invalid} {};

    ~Element() {};


    /**
     * Read an element from a string. The string should be a complete
     * line with an EOL marker. If this string is not a well-formed line,
     * will throw an exception
     */
    static Element read(double version, const std::string &line);

    GMSH_ELEMENT_ITEM(Line);
    GMSH_ELEMENT_ITEM(Triangle);
    GMSH_ELEMENT_ITEM(Quadrangle);
    GMSH_ELEMENT_ITEM(Tetrahedron);
    GMSH_ELEMENT_ITEM(Hexahedron);
    GMSH_ELEMENT_ITEM(Prism);
    GMSH_ELEMENT_ITEM(Line3);
    GMSH_ELEMENT_ITEM(Triangle6);
    GMSH_ELEMENT_ITEM(Quadrangle9);
    GMSH_ELEMENT_ITEM(Tetrahedron10);
    GMSH_ELEMENT_ITEM(Hexadedron27);
    GMSH_ELEMENT_ITEM(Prism18);
    GMSH_ELEMENT_ITEM(Pyramid14);
    GMSH_ELEMENT_ITEM(Point);
    GMSH_ELEMENT_ITEM(Quadrangle8);
    GMSH_ELEMENT_ITEM(Hexadedron20);
    GMSH_ELEMENT_ITEM(Prism15);
    GMSH_ELEMENT_ITEM(Pyramid13);
    GMSH_ELEMENT_ITEM(TriangleInc9);
    GMSH_ELEMENT_ITEM(Triangle10);
    GMSH_ELEMENT_ITEM(TrianbleInc12);
    GMSH_ELEMENT_ITEM(Triangle15);
    GMSH_ELEMENT_ITEM(TriangleInc15);
    GMSH_ELEMENT_ITEM(Triangle21);
    GMSH_ELEMENT_ITEM(Edge4);
    GMSH_ELEMENT_ITEM(Edge5);
    GMSH_ELEMENT_ITEM(Edge6);
    GMSH_ELEMENT_ITEM(Tetrahedron20);
    GMSH_ELEMENT_ITEM(Tetrahedron35);
    GMSH_ELEMENT_ITEM(Tetrahedron56);
    GMSH_ELEMENT_ITEM(Hexahedron64);
    GMSH_ELEMENT_ITEM(Hexahedron125);


    // access the common items for each union type.
    // this works because all union types share the same binary layout
    // as the Point, except all the others have larger nodes array.
    int id() const {
        return val_Point.id;
    }

    int elementary() const {
            return val_Point.elementary;
    }

    int physical() const {
            return val_Point.physical;
    }


private:
    union {
    GMSH_UNION(Line);
    GMSH_UNION(Triangle);
    GMSH_UNION(Quadrangle);
    GMSH_UNION(Tetrahedron);
    GMSH_UNION(Hexahedron);
    GMSH_UNION(Prism);
    GMSH_UNION(Line3);
    GMSH_UNION(Triangle6);
    GMSH_UNION(Quadrangle9);
    GMSH_UNION(Tetrahedron10);
    GMSH_UNION(Hexadedron27);
    GMSH_UNION(Prism18);
    GMSH_UNION(Pyramid14);
    GMSH_UNION(Point);
    GMSH_UNION(Quadrangle8);
    GMSH_UNION(Hexadedron20);
    GMSH_UNION(Prism15);
    GMSH_UNION(Pyramid13);
    GMSH_UNION(TriangleInc9);
    GMSH_UNION(Triangle10);
    GMSH_UNION(TrianbleInc12);
    GMSH_UNION(Triangle15);
    GMSH_UNION(TriangleInc15);
    GMSH_UNION(Triangle21);
    GMSH_UNION(Edge4);
    GMSH_UNION(Edge5);
    GMSH_UNION(Edge6);
    GMSH_UNION(Tetrahedron20);
    GMSH_UNION(Tetrahedron35);
    GMSH_UNION(Tetrahedron56);
    GMSH_UNION(Hexahedron64);
    GMSH_UNION(Hexahedron125);

    };

    template <typename T>
    void init(T& member, const T& val)
    {
      new (&member) T(val);
    }


    void copy(const Element& other) {
        switch(type) {
        GMSH_ELEMENT_COPY_CTOR(Line);
        GMSH_ELEMENT_COPY_CTOR(Triangle);
        GMSH_ELEMENT_COPY_CTOR(Quadrangle);
        GMSH_ELEMENT_COPY_CTOR(Tetrahedron);
        GMSH_ELEMENT_COPY_CTOR(Hexahedron);
        GMSH_ELEMENT_COPY_CTOR(Prism);
        GMSH_ELEMENT_COPY_CTOR(Line3);
        GMSH_ELEMENT_COPY_CTOR(Triangle6);
        GMSH_ELEMENT_COPY_CTOR(Quadrangle9);
        GMSH_ELEMENT_COPY_CTOR(Tetrahedron10);
        GMSH_ELEMENT_COPY_CTOR(Hexadedron27);
        GMSH_ELEMENT_COPY_CTOR(Prism18);
        GMSH_ELEMENT_COPY_CTOR(Pyramid14);
        GMSH_ELEMENT_COPY_CTOR(Point);
        GMSH_ELEMENT_COPY_CTOR(Quadrangle8);
        GMSH_ELEMENT_COPY_CTOR(Hexadedron20);
        GMSH_ELEMENT_COPY_CTOR(Prism15);
        GMSH_ELEMENT_COPY_CTOR(Pyramid13);
        GMSH_ELEMENT_COPY_CTOR(TriangleInc9);
        GMSH_ELEMENT_COPY_CTOR(Triangle10);
        GMSH_ELEMENT_COPY_CTOR(TrianbleInc12);
        GMSH_ELEMENT_COPY_CTOR(Triangle15);
        GMSH_ELEMENT_COPY_CTOR(TriangleInc15);
        GMSH_ELEMENT_COPY_CTOR(Triangle21);
        GMSH_ELEMENT_COPY_CTOR(Edge4);
        GMSH_ELEMENT_COPY_CTOR(Edge5);
        GMSH_ELEMENT_COPY_CTOR(Edge6);
        GMSH_ELEMENT_COPY_CTOR(Tetrahedron20);
        GMSH_ELEMENT_COPY_CTOR(Tetrahedron35);
        GMSH_ELEMENT_COPY_CTOR(Tetrahedron56);
        GMSH_ELEMENT_COPY_CTOR(Hexahedron64);
        GMSH_ELEMENT_COPY_CTOR(Hexahedron125);
        default: assert(0);
        }
    }
};

GMSH_ELEMENT_GET(Line);
GMSH_ELEMENT_GET(Triangle);
GMSH_ELEMENT_GET(Quadrangle);
GMSH_ELEMENT_GET(Tetrahedron);
GMSH_ELEMENT_GET(Hexahedron);
GMSH_ELEMENT_GET(Prism);
GMSH_ELEMENT_GET(Line3);
GMSH_ELEMENT_GET(Triangle6);
GMSH_ELEMENT_GET(Quadrangle9);
GMSH_ELEMENT_GET(Tetrahedron10);
GMSH_ELEMENT_GET(Hexadedron27);
GMSH_ELEMENT_GET(Prism18);
GMSH_ELEMENT_GET(Pyramid14);
GMSH_ELEMENT_GET(Point);
GMSH_ELEMENT_GET(Quadrangle8);
GMSH_ELEMENT_GET(Hexadedron20);
GMSH_ELEMENT_GET(Prism15);
GMSH_ELEMENT_GET(Pyramid13);
GMSH_ELEMENT_GET(TriangleInc9);
GMSH_ELEMENT_GET(Triangle10);
GMSH_ELEMENT_GET(TrianbleInc12);
GMSH_ELEMENT_GET(Triangle15);
GMSH_ELEMENT_GET(TriangleInc15);
GMSH_ELEMENT_GET(Triangle21);
GMSH_ELEMENT_GET(Edge4);
GMSH_ELEMENT_GET(Edge5);
GMSH_ELEMENT_GET(Edge6);
GMSH_ELEMENT_GET(Tetrahedron20);
GMSH_ELEMENT_GET(Tetrahedron35);
GMSH_ELEMENT_GET(Tetrahedron56);
GMSH_ELEMENT_GET(Hexahedron64);
GMSH_ELEMENT_GET(Hexahedron125);

#undef GMSH_ELEMENT_ITEM
#undef GMSH_ELEMENT_GET
#undef GMSH_UNION

struct Node {
    unsigned int id;
    double pos[3];
};

struct Mesh {
    std::unordered_map<unsigned int, Node> nodes;
    std::vector<Element> elements;
    std::unordered_map<int, Physical> physicals;

    /**
     * Reads in a mesh in the Gmsh *.msh format from the ASCII file
     * given by name.
     *
     * The user is responsible for calling Mesh::prepare_for_use()
     * after reading the mesh and before using it.
     */
    static Mesh read(std::istream &in);


    static Mesh read(const std::string &path);

    /**
     * This method implements writing a mesh to a specified file
     * in the Gmsh *.msh format.
     */
    static void write(const Mesh& mesh, std::ofstream &out);



    /**
     * This method implements writing a mesh with nodal data to a
     * specified file where the nodal data and variable names are provided.
     */
    //virtual void write_nodal_data (const std::string &,
      //                             const std::vector<Number> &,
        //                           const std::vector<std::string> &) libmesh_override;

    /**
     * Flag indicating whether or not to write a binary file.  While binary
     * files may end up being smaller than equivalent ASCII files, they will
     * almost certainly take longer to write.  The reason for this is that
     * the ostream::write() function which is used to write "binary" data to
     * streams, only takes a pointer to char as its first argument.  This means
     * if you want to write anything other than a buffer of chars, you first
     * have to use a strange memcpy hack to get the data into the desired format.
     * See the templated to_binary_stream() function below.
     */
    bool binary () {
        return static_cast<ElementType>(1) == ElementType::Edge4;
    }

    /**
     * Flag to write binary data.
     */
    bool _binary;


};



#if 0

/**
 * This class implements writing meshes in the Gmsh format.
 * For a full description of the Gmsh format and to obtain the
 * GMSH software see
 * <a href="http://http://www.geuz.org/gmsh/">the Gmsh home page</a>
 *
 * Based on the libMesh Gmsh IO class written by John W. Peterson and
 * Martin Luthi
 */
class GmshIO
{
public:



  /**
   * Constructor.  Takes a non-const Mesh reference which it
   * will fill up with elements via the read() command.
   */
  explicit GmshIO ();

  /**
   * Reads in a mesh in the Gmsh *.msh format from the ASCII file
   * given by name.
   *
   * The user is responsible for calling Mesh::prepare_for_use()
   * after reading the mesh and before using it.
   */
  virtual void read (const std::string & name);

  /**
   * This method implements writing a mesh to a specified file
   * in the Gmsh *.msh format.
   */
  virtual void write (const std::string & name) ;


  /**
   * This method implements writing a mesh with nodal data to a
   * specified file where the nodal data and variable names are provided.
   */
  virtual void write_nodal_data (const std::string &,
                                 const std::vector<Number> &,
                                 const std::vector<std::string> &) libmesh_override;

  /**
   * Flag indicating whether or not to write a binary file.  While binary
   * files may end up being smaller than equivalent ASCII files, they will
   * almost certainly take longer to write.  The reason for this is that
   * the ostream::write() function which is used to write "binary" data to
   * streams, only takes a pointer to char as its first argument.  This means
   * if you want to write anything other than a buffer of chars, you first
   * have to use a strange memcpy hack to get the data into the desired format.
   * See the templated to_binary_stream() function below.
   */
  bool & binary ();

  /**
   * Access to the flag which controls whether boundary elements are
   * written to the Mesh file.
   */
  bool & write_lower_dimensional_elements ();

private:
  /**
   * Implementation of the read() function.  This function
   * is called by the public interface function and implements
   * reading the file.
   */
  void read_mesh (std::istream & in);

  /**
   * This method implements writing a mesh to a
   * specified file.  This will write an ASCII *.msh file.
   */
  void write_mesh (std::ostream & out);

  /**
   * This method implements writing a mesh with nodal data to a specified file
   * where the nodal data and variable names are optionally provided.  This
   * will write an ASCII or binary *.pos file, depending on the binary flag.
   */
  void write_post (const std::string &,
                   const std::vector<Number> * = libmesh_nullptr,
                   const std::vector<std::string> * = libmesh_nullptr);

  /**
   * Flag to write binary data.
   */
  bool _binary;

  /**
   * If true, lower-dimensional elements based on the boundary
   * conditions get written to the output file.
   */
  bool _write_lower_dimensional_elements;

  /**
   * Defines mapping from libMesh element types to Gmsh element types or vice-versa.
   */
  struct ElementDefinition
  {
    ElementDefinition(ElemType type_in,
                      unsigned gmsh_type_in,
                      unsigned dim_in,
                      unsigned nnodes_in) :
      type(type_in),
      gmsh_type(gmsh_type_in),
      dim(dim_in),
      nnodes(nnodes_in)
    {}

    ElemType type;
    unsigned int gmsh_type;
    unsigned int dim;
    unsigned int nnodes;
    std::vector<unsigned int> nodes;
  };

  /**
   * struct which holds a map from Gmsh to libMesh element numberings
   * and vice-versa.
   */
  struct ElementMaps
  {
    // Helper function to add a (key, value) pair to both maps
    void add_def(const ElementDefinition & eledef)
    {
      out.insert(std::make_pair(eledef.type, eledef));
      in.insert(std::make_pair(eledef.gmsh_type, eledef));
    }

    std::map<ElemType, ElementDefinition> out;
    std::map<unsigned int, ElementDefinition> in;
  };

  /**
   * A static ElementMaps object that is built statically and used by
   * all instances of this class.
   */
  static ElementMaps _element_maps;

  /**
   * A static function used to construct the _element_maps struct,
   * statically.
   */
  static ElementMaps build_element_maps();
};

#endif

}




#endif // LIBMESH_GMSH_IO_H
