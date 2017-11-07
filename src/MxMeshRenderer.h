/*
 * MxMeshRenderer.h
 *
 *  Created on: Jul 7, 2017
 *      Author: andy
 */

#ifndef SRC_MXMESHRENDERER_H_
#define SRC_MXMESHRENDERER_H_

#include <MxMesh.h>

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Buffer.h>
#include <Magnum/Mesh.h>


struct MagnumCellRenderer : MxCellRenderer {


    CellPtr cell;
    Magnum::Mesh mesh;
    Magnum::Buffer indexBuffer;
    Magnum::Buffer vertexBuffer;

    MagnumCellRenderer(CellPtr cell) : cell{cell} {invalidate();};

    void draw(AbstractShaderProgram& shader);

    virtual HRESULT invalidate();

    virtual ~MagnumCellRenderer();
};



/**
 * purpose: map the geometry of an MxMesh object to the screen, preform the
 * rendering of a MxMesh.
 *
 * Key ideas is that the MxMesh represents just the geometry and other various
 * `statefull` attributes, (such as say chemical concentrations, and other scalar
 * and in the future, potentially vector values). This class then reads those values
 * and figures out how to visually present them.
 *
 * This class is responsible for generating and maintaining a set of vertex buffer
 * object, and mesh, as in Magnum::Mesh objects, getting updates from the MxMesh
 * source, updating the VBOs as needed, and formating and presenting the visual
 * display of them.
 *
 * Design considerations:
 * This class will need to create multiple different kinds of shaders, we'll need
 * some shaders for wireframe, some for face attributes, etc... Eventually, we'll
 * need to create new shaders to render vector fields, and other kinds of data.
 *
 * Need to think about what kinds of data changes quickly, and what changes more slowly,
 * and be able to flexibly update data on the graphics board at different rates.
 *
 * Graphics hardware is really set up for vertex, rather than face attributes,
 * but we store a lot of state at the face, rather than the vertex level, i.e.
 * face concentrations, and other attributes. In order to get things moving quickly,
 * we're going to send redundant position information to the graphics processor,
 * but we'll have different color info attached to each vertex. To be more flexible,
 * we will use separate arrays for each vertex attribute, instead of usign packed,
 * interleaved arrays. Future versions will explore packing with interleaved arrays.
 * separate ararys enable us to simple add a new array each time we come up with a
 * different kind of attribute.
 *
 * Is it better to have separate index buffers for each MxCell? Yes, probably,
 * because we will have very many cells, and each cell will have relativly few
 * faces per cell, I'm guessing on the order of 30 up to around a thousand. Each
 * cell would then incur a separate draw call, and would create many more index
 * buffers on the graphics processor. The advantage of separate index buffers is
 * that it's simpler to implement (probably). Is it really that much simpler to
 * to create separate meshes and index buffers? There's not much of a differce
 * from the MxCell side, in either case, the MxCell needs methods where it can
 * write a set of indices, vertices, etc. to a given pointer. In either case,
 * all we have to do is control what the pointer points to. For separate index
 * buffers, it's just the start of the IB array, for the same, we just bump the
 * pointer for each MxCell that we process.
 *
 * Separate mesh for each cell:
 * keep track of a vector of meshes
 *
 *
 * The renderer takes care of calculating a color representation for each scalar
 * field vertex attribute that the underlying MxMesh has. The mesh thus does not
 * keep colors, only float scalars.
 *
 *
 *
 */
class MxMeshRenderer {
    //using Magnum::Matrix4;
public:

    /**
     * @brief Flag
     *
     * @see @ref Flags, @ref MxMeshRenderer()
     */
    enum class Flag: Magnum::UnsignedByte {

        Wireframe = 1 << 0,

        /**
         * Don't use geometry shader for wireframe visualization. If
         * enabled, you might need to provide also @ref VertexIndex
         * attribute in the mesh.
         */
        NoGeometryShader = 1 << 1
    };

    /** @brief Flags */
    typedef Corrade::Containers::EnumSet<Flag> Flags;


    /**
     * @brief Constructor
     * @param flags     Flags
     */
    explicit MxMeshRenderer(Flags flags = {});


    ~MxMeshRenderer();

    MxMeshRenderer& setMesh(MxMesh *mesh);

    MxMeshRenderer& setModelMatrix(const Magnum::Matrix4& mat);

    MxMeshRenderer& setViewMatrix(const Magnum::Matrix4& mat);

    MxMeshRenderer& setProjectionMatrix(const Magnum::Matrix4& mat);


    /**
     * @brief Set viewport size
     * @return Reference to self (for method chaining)
     *
     * Has effect only if @ref Flag::Wireframe is enabled and geometry
     * shaders are used.
     */
    MxMeshRenderer& setViewportSize(const Magnum::Vector2& size);

    /**
     * @brief Set base object color
     * @return Reference to self (for method chaining)
     *
     * Initial value is fully opaque white.
     */
    MxMeshRenderer& setColor(const Magnum::Color4& color);

    /**
     * @brief Set wireframe color
     * @return Reference to self (for method chaining)
     *
     * Initial value is fully opaque black. Has effect only if
     * @ref Flag::Wireframe is enabled.
     */
    MxMeshRenderer& setWireframeColor(const Magnum::Color4& color);

    /**
     * @brief Set wireframe width
     * @return Reference to self (for method chaining)
     *
     * Initial value is `1.0f`. Has effect only if @ref Flag::Wireframe is
     * enabled.
     */
    MxMeshRenderer& setWireframeWidth(float width);

    /**
     * @brief Set line smoothness
     * @return Reference to self (for method chaining)
     *
     * Initial value is `2.0f`. Has effect only if @ref Flag::Wireframe is
     * enabled.
     */
    MxMeshRenderer& setSmoothness(float smoothness);

    /**
     * Draw the mesh(s) with opengl in it's current state
     */
    void draw();


private:

    MxMesh *mesh;
    class MxMeshShaderProgram *shader;



    /**
     * Only set a single combined matrix in the shader, this way,
     * the shader only performs a single matrix multiply of the vertices, update the
     * shader matrix whenever any of these change.
     *
     * multiplication order is the reverse of the pipeline.
     * Therefore you do totalmat = proj * view * model.
     */
    Magnum::Matrix4 modelMat, viewMat, projMat = Magnum::Math::IdentityInit;



};

#endif /* SRC_MXMESHRENDERER_H_ */
