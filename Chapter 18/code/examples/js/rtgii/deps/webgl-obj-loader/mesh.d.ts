import { Layout } from "./layout";
import { Material, MaterialLibrary } from "./material";
export interface MeshOptions {
    enableWTextureCoord?: boolean;
    calcTangentsAndBitangents?: boolean;
    materials?: {
        [key: string]: Material;
    };
}
export interface MaterialNameToIndex {
    [k: string]: number;
}
export interface IndexToMaterial {
    [k: number]: Material;
}
export interface ArrayBufferWithItemSize extends ArrayBuffer {
    numItems?: number;
}
export interface Uint16ArrayWithItemSize extends Uint16Array {
    numItems?: number;
}
/**
 * The main Mesh class. The constructor will parse through the OBJ file data
 * and collect the vertex, vertex normal, texture, and face information. This
 * information can then be used later on when creating your VBOs. See
 * OBJ.initMeshBuffers for an example of how to use the newly created Mesh
 */
export default class Mesh {
    vertices: number[];
    vertexNormals: number[];
    textures: number[];
    indices: number[];
    name: string;
    vertexMaterialIndices: number[];
    indicesPerMaterial: number[][];
    materialNames: string[];
    materialIndices: MaterialNameToIndex;
    materialsByIndex: IndexToMaterial;
    tangents: number[];
    bitangents: number[];
    textureStride: number;
    /**
     * Create a Mesh
     * @param {String} objectData - a string representation of an OBJ file with
     *     newlines preserved.
     * @param {Object} options - a JS object containing valid options. See class
     *     documentation for options.
     * @param {bool} options.enableWTextureCoord - Texture coordinates can have
     *     an optional "w" coordinate after the u and v coordinates. This extra
     *     value can be used in order to perform fancy transformations on the
     *     textures themselves. Default is to truncate to only the u an v
     *     coordinates. Passing true will provide a default value of 0 in the
     *     event that any or all texture coordinates don't provide a w value.
     *     Always use the textureStride attribute in order to determine the
     *     stride length of the texture coordinates when rendering the element
     *     array.
     * @param {bool} options.calcTangentsAndBitangents - Calculate the tangents
     *     and bitangents when loading of the OBJ is completed. This adds two new
     *     attributes to the Mesh instance: `tangents` and `bitangents`.
     */
    constructor(objectData: string, options?: MeshOptions);
    /**
     * Calculates the tangents and bitangents of the mesh that forms an orthogonal basis together with the
     * normal in the direction of the texture coordinates. These are useful for setting up the TBN matrix
     * when distorting the normals through normal maps.
     * Method derived from: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
     *
     * This method requires the normals and texture coordinates to be parsed and set up correctly.
     * Adds the tangents and bitangents as members of the class instance.
     */
    calculateTangentsAndBitangents(): void;
    /**
     * @param layout - A {@link Layout} object that describes the
     * desired memory layout of the generated buffer
     * @return The packed array in the ... TODO
     */
    makeBufferData(layout: Layout): ArrayBufferWithItemSize;
    makeIndexBufferData(): Uint16ArrayWithItemSize;
    makeIndexBufferDataForMaterials(...materialIndices: Array<number>): Uint16ArrayWithItemSize;
    addMaterialLibrary(mtl: MaterialLibrary): void;
}
