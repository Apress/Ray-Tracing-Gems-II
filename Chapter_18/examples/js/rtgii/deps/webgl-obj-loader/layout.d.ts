export declare enum TYPES {
    "BYTE" = "BYTE",
    "UNSIGNED_BYTE" = "UNSIGNED_BYTE",
    "SHORT" = "SHORT",
    "UNSIGNED_SHORT" = "UNSIGNED_SHORT",
    "FLOAT" = "FLOAT"
}
export interface AttributeInfo {
    attribute: Attribute;
    size: Attribute["size"];
    type: Attribute["type"];
    normalized: Attribute["normalized"];
    offset: number;
    stride: number;
}
/**
 * An exception for when two or more of the same attributes are found in the
 * same layout.
 * @private
 */
export declare class DuplicateAttributeException extends Error {
    /**
     * Create a DuplicateAttributeException
     * @param {Attribute} attribute - The attribute that was found more than
     *        once in the {@link Layout}
     */
    constructor(attribute: Attribute);
}
/**
 * Represents how a vertex attribute should be packed into an buffer.
 * @private
 */
export declare class Attribute {
    key: string;
    size: number;
    type: TYPES;
    normalized: boolean;
    sizeOfType: number;
    sizeInBytes: number;
    /**
     * Create an attribute. Do not call this directly, use the predefined
     * constants.
     * @param {string} key - The name of this attribute as if it were a key in
     *        an Object. Use the camel case version of the upper snake case
     *        const name.
     * @param {number} size - The number of components per vertex attribute.
     *        Must be 1, 2, 3, or 4.
     * @param {string} type - The data type of each component for this
     *        attribute. Possible values:<br/>
     *        "BYTE": signed 8-bit integer, with values in [-128, 127]<br/>
     *        "SHORT": signed 16-bit integer, with values in
     *            [-32768, 32767]<br/>
     *        "UNSIGNED_BYTE": unsigned 8-bit integer, with values in
     *            [0, 255]<br/>
     *        "UNSIGNED_SHORT": unsigned 16-bit integer, with values in
     *            [0, 65535]<br/>
     *        "FLOAT": 32-bit floating point number
     * @param {boolean} normalized - Whether integer data values should be
     *        normalized when being casted to a float.<br/>
     *        If true, signed integers are normalized to [-1, 1].<br/>
     *        If true, unsigned integers are normalized to [0, 1].<br/>
     *        For type "FLOAT", this parameter has no effect.
     */
    constructor(key: string, size: number, type: TYPES, normalized?: boolean);
}
/**
 * A class to represent the memory layout for a vertex attribute array. Used by
 * {@link Mesh}'s TBD(...) method to generate a packed array from mesh data.
 * <p>
 * Layout can sort of be thought of as a C-style struct declaration.
 * {@link Mesh}'s TBD(...) method will use the {@link Layout} instance to
 * pack an array in the given attribute order.
 * <p>
 * Layout also is very helpful when calling a WebGL context's
 * <code>vertexAttribPointer</code> method. If you've created a buffer using
 * a Layout instance, then the same Layout instance can be used to determine
 * the size, type, normalized, stride, and offset parameters for
 * <code>vertexAttribPointer</code>.
 * <p>
 * For example:
 * <pre><code>
 *
 * const index = glctx.getAttribLocation(shaderProgram, "pos");
 * glctx.vertexAttribPointer(
 *   layout.position.size,
 *   glctx[layout.position.type],
 *   layout.position.normalized,
 *   layout.position.stride,
 *   layout.position.offset);
 * </code></pre>
 * @see {@link Mesh}
 */
export declare class Layout {
    /**
     * Attribute layout to pack a vertex's x, y, & z as floats
     *
     * @see {@link Layout}
     */
    static POSITION: Attribute;
    /**
     * Attribute layout to pack a vertex's normal's x, y, & z as floats
     *
     * @see {@link Layout}
     */
    static NORMAL: Attribute;
    /**
     * Attribute layout to pack a vertex's normal's x, y, & z as floats.
     * <p>
     * This value will be computed on-the-fly based on the texture coordinates.
     * If no texture coordinates are available, the generated value will default to
     * 0, 0, 0.
     *
     * @see {@link Layout}
     */
    static TANGENT: Attribute;
    /**
     * Attribute layout to pack a vertex's normal's bitangent x, y, & z as floats.
     * <p>
     * This value will be computed on-the-fly based on the texture coordinates.
     * If no texture coordinates are available, the generated value will default to
     * 0, 0, 0.
     * @see {@link Layout}
     */
    static BITANGENT: Attribute;
    /**
     * Attribute layout to pack a vertex's texture coordinates' u & v as floats
     *
     * @see {@link Layout}
     */
    static UV: Attribute;
    /**
     * Attribute layout to pack an unsigned short to be interpreted as a the index
     * into a {@link Mesh}'s materials list.
     * <p>
     * The intention of this value is to send all of the {@link Mesh}'s materials
     * into multiple shader uniforms and then reference the current one by this
     * vertex attribute.
     * <p>
     * example glsl code:
     *
     * <pre><code>
     *  // this is bound using MATERIAL_INDEX
     *  attribute int materialIndex;
     *
     *  struct Material {
     *    vec3 diffuse;
     *    vec3 specular;
     *    vec3 specularExponent;
     *  };
     *
     *  uniform Material materials[MAX_MATERIALS];
     *
     *  // ...
     *
     *  vec3 diffuse = materials[materialIndex];
     *
     * </code></pre>
     * TODO: More description & test to make sure subscripting by attributes even
     * works for webgl
     *
     * @see {@link Layout}
     */
    static MATERIAL_INDEX: Attribute;
    static MATERIAL_ENABLED: Attribute;
    static AMBIENT: Attribute;
    static DIFFUSE: Attribute;
    static SPECULAR: Attribute;
    static SPECULAR_EXPONENT: Attribute;
    static EMISSIVE: Attribute;
    static TRANSMISSION_FILTER: Attribute;
    static DISSOLVE: Attribute;
    static ILLUMINATION: Attribute;
    static REFRACTION_INDEX: Attribute;
    static SHARPNESS: Attribute;
    static MAP_DIFFUSE: Attribute;
    static MAP_AMBIENT: Attribute;
    static MAP_SPECULAR: Attribute;
    static MAP_SPECULAR_EXPONENT: Attribute;
    static MAP_DISSOLVE: Attribute;
    static ANTI_ALIASING: Attribute;
    static MAP_BUMP: Attribute;
    static MAP_DISPLACEMENT: Attribute;
    static MAP_DECAL: Attribute;
    static MAP_EMISSIVE: Attribute;
    stride: number;
    attributes: Attribute[];
    attributeMap: {
        [idx: string]: AttributeInfo;
    };
    /**
     * Create a Layout object. This constructor will throw if any duplicate
     * attributes are given.
     * @param {Array} ...attributes - An ordered list of attributes that
     *        describe the desired memory layout for each vertex attribute.
     *        <p>
     *
     * @see {@link Mesh}
     */
    constructor(...attributes: Attribute[]);
}
