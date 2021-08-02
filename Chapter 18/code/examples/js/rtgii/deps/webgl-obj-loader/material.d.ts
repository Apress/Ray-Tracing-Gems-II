export declare type Vec3 = [number, number, number];
export interface UVW {
    u: number;
    v: number;
    w: number;
}
export interface TextureMapData {
    colorCorrection: boolean;
    horizontalBlending: boolean;
    verticalBlending: boolean;
    boostMipMapSharpness: number;
    modifyTextureMap: {
        brightness: number;
        contrast: number;
    };
    offset: UVW;
    scale: UVW;
    turbulence: UVW;
    clamp: boolean;
    textureResolution: number | null;
    bumpMultiplier: number;
    imfChan: string | null;
    filename: string;
    reflectionType?: string;
    texture?: HTMLImageElement;
}
/**
 * The Material class.
 */
export declare class Material {
    name: string;
    /**
     * Constructor
     * @param {String} name the unique name of the material
     */
    ambient: Vec3;
    diffuse: Vec3;
    specular: Vec3;
    emissive: Vec3;
    transmissionFilter: Vec3;
    dissolve: number;
    specularExponent: number;
    transparency: number;
    illumination: number;
    refractionIndex: number;
    sharpness: number;
    mapDiffuse: TextureMapData;
    mapAmbient: TextureMapData;
    mapSpecular: TextureMapData;
    mapSpecularExponent: TextureMapData;
    mapDissolve: TextureMapData;
    antiAliasing: boolean;
    mapBump: TextureMapData;
    mapDisplacement: TextureMapData;
    mapDecal: TextureMapData;
    mapEmissive: TextureMapData;
    mapReflections: TextureMapData[];
    constructor(name: string);
}
/**
 * https://en.wikipedia.org/wiki/Wavefront_.obj_file
 * http://paulbourke.net/dataformats/mtl/
 */
export declare class MaterialLibrary {
    data: string;
    /**
     * Constructs the Material Parser
     * @param mtlData the MTL file contents
     */
    currentMaterial: Material;
    materials: {
        [k: string]: Material;
    };
    constructor(data: string);
    /**
     * Creates a new Material object and adds to the registry.
     * @param tokens the tokens associated with the directive
     */
    parse_newmtl(tokens: string[]): void;
    /**
     * See the documenation for parse_Ka below for a better understanding.
     *
     * Given a list of possible color tokens, returns an array of R, G, and B
     * color values.
     *
     * @param tokens the tokens associated with the directive
     * @return {*} a 3 element array containing the R, G, and B values
     * of the color.
     */
    parseColor(tokens: string[]): Vec3;
    /**
     * Parse the ambient reflectivity
     *
     * A Ka directive can take one of three forms:
     *   - Ka r g b
     *   - Ka spectral file.rfl
     *   - Ka xyz x y z
     * These three forms are mutually exclusive in that only one
     * declaration can exist per material. It is considered a syntax
     * error otherwise.
     *
     * The "Ka" form specifies the ambient reflectivity using RGB values.
     * The "g" and "b" values are optional. If only the "r" value is
     * specified, then the "g" and "b" values are assigned the value of
     * "r". Values are normally in the range 0.0 to 1.0. Values outside
     * of this range increase or decrease the reflectivity accordingly.
     *
     * The "Ka spectral" form specifies the ambient reflectivity using a
     * spectral curve. "file.rfl" is the name of the ".rfl" file containing
     * the curve data. "factor" is an optional argument which is a multiplier
     * for the values in the .rfl file and defaults to 1.0 if not specified.
     *
     * The "Ka xyz" form specifies the ambient reflectivity using CIEXYZ values.
     * "x y z" are the values of the CIEXYZ color space. The "y" and "z" arguments
     * are optional and take on the value of the "x" component if only "x" is
     * specified. The "x y z" values are normally in the range of 0.0 to 1.0 and
     * increase or decrease ambient reflectivity accordingly outside of that
     * range.
     *
     * @param tokens the tokens associated with the directive
     */
    parse_Ka(tokens: string[]): void;
    /**
     * Diffuse Reflectivity
     *
     * Similar to the Ka directive. Simply replace "Ka" with "Kd" and the rules
     * are the same
     *
     * @param tokens the tokens associated with the directive
     */
    parse_Kd(tokens: string[]): void;
    /**
     * Spectral Reflectivity
     *
     * Similar to the Ka directive. Simply replace "Ks" with "Kd" and the rules
     * are the same
     *
     * @param tokens the tokens associated with the directive
     */
    parse_Ks(tokens: string[]): void;
    /**
     * Emissive
     *
     * The amount and color of light emitted by the object.
     *
     * @param tokens the tokens associated with the directive
     */
    parse_Ke(tokens: string[]): void;
    /**
     * Transmission Filter
     *
     * Any light passing through the object is filtered by the transmission
     * filter, which only allows specific colors to pass through. For example, Tf
     * 0 1 0 allows all of the green to pass through and filters out all of the
     * red and blue.
     *
     * Similar to the Ka directive. Simply replace "Ks" with "Tf" and the rules
     * are the same
     *
     * @param tokens the tokens associated with the directive
     */
    parse_Tf(tokens: string[]): void;
    /**
     * Specifies the dissolve for the current material.
     *
     * Statement: d [-halo] `factor`
     *
     * Example: "d 0.5"
     *
     * The factor is the amount this material dissolves into the background. A
     * factor of 1.0 is fully opaque. This is the default when a new material is
     * created. A factor of 0.0 is fully dissolved (completely transparent).
     *
     * Unlike a real transparent material, the dissolve does not depend upon
     * material thickness nor does it have any spectral character. Dissolve works
     * on all illumination models.
     *
     * The dissolve statement allows for an optional "-halo" flag which indicates
     * that a dissolve is dependent on the surface orientation relative to the
     * viewer. For example, a sphere with the following dissolve, "d -halo 0.0",
     * will be fully dissolved at its center and will appear gradually more opaque
     * toward its edge.
     *
     * "factor" is the minimum amount of dissolve applied to the material. The
     * amount of dissolve will vary between 1.0 (fully opaque) and the specified
     * "factor". The formula is:
     *
     *    dissolve = 1.0 - (N*v)(1.0-factor)
     *
     * @param tokens the tokens associated with the directive
     */
    parse_d(tokens: string[]): void;
    /**
     * The "illum" statement specifies the illumination model to use in the
     * material. Illumination models are mathematical equations that represent
     * various material lighting and shading effects.
     *
     * The illumination number can be a number from 0 to 10. The following are
     * the list of illumination enumerations and their summaries:
     * 0. Color on and Ambient off
     * 1. Color on and Ambient on
     * 2. Highlight on
     * 3. Reflection on and Ray trace on
     * 4. Transparency: Glass on, Reflection: Ray trace on
     * 5. Reflection: Fresnel on and Ray trace on
     * 6. Transparency: Refraction on, Reflection: Fresnel off and Ray trace on
     * 7. Transparency: Refraction on, Reflection: Fresnel on and Ray trace on
     * 8. Reflection on and Ray trace off
     * 9. Transparency: Glass on, Reflection: Ray trace off
     * 10. Casts shadows onto invisible surfaces
     *
     * Example: "illum 2" to specify the "Highlight on" model
     *
     * @param tokens the tokens associated with the directive
     */
    parse_illum(tokens: string[]): void;
    /**
     * Optical Density (AKA Index of Refraction)
     *
     * Statement: Ni `index`
     *
     * Example: Ni 1.0
     *
     * Specifies the optical density for the surface. `index` is the value
     * for the optical density. The values can range from 0.001 to 10.  A value of
     * 1.0 means that light does not bend as it passes through an object.
     * Increasing the optical_density increases the amount of bending. Glass has
     * an index of refraction of about 1.5. Values of less than 1.0 produce
     * bizarre results and are not recommended
     *
     * @param tokens the tokens associated with the directive
     */
    parse_Ni(tokens: string[]): void;
    /**
     * Specifies the specular exponent for the current material. This defines the
     * focus of the specular highlight.
     *
     * Statement: Ns `exponent`
     *
     * Example: "Ns 250"
     *
     * `exponent` is the value for the specular exponent. A high exponent results
     * in a tight, concentrated highlight. Ns Values normally range from 0 to
     * 1000.
     *
     * @param tokens the tokens associated with the directive
     */
    parse_Ns(tokens: string[]): void;
    /**
     * Specifies the sharpness of the reflections from the local reflection map.
     *
     * Statement: sharpness `value`
     *
     * Example: "sharpness 100"
     *
     * If a material does not have a local reflection map defined in its material
     * defintions, sharpness will apply to the global reflection map defined in
     * PreView.
     *
     * `value` can be a number from 0 to 1000. The default is 60. A high value
     * results in a clear reflection of objects in the reflection map.
     *
     * Tip: sharpness values greater than 100 introduce aliasing effects in
     * flat surfaces that are viewed at a sharp angle.
     *
     * @param tokens the tokens associated with the directive
     */
    parse_sharpness(tokens: string[]): void;
    /**
     * Parses the -cc flag and updates the options object with the values.
     *
     * @param values the values passed to the -cc flag
     * @param options the Object of all image options
     */
    parse_cc(values: string[], options: TextureMapData): void;
    /**
     * Parses the -blendu flag and updates the options object with the values.
     *
     * @param values the values passed to the -blendu flag
     * @param options the Object of all image options
     */
    parse_blendu(values: string[], options: TextureMapData): void;
    /**
     * Parses the -blendv flag and updates the options object with the values.
     *
     * @param values the values passed to the -blendv flag
     * @param options the Object of all image options
     */
    parse_blendv(values: string[], options: TextureMapData): void;
    /**
     * Parses the -boost flag and updates the options object with the values.
     *
     * @param values the values passed to the -boost flag
     * @param options the Object of all image options
     */
    parse_boost(values: string[], options: TextureMapData): void;
    /**
     * Parses the -mm flag and updates the options object with the values.
     *
     * @param values the values passed to the -mm flag
     * @param options the Object of all image options
     */
    parse_mm(values: string[], options: TextureMapData): void;
    /**
     * Parses and sets the -o, -s, and -t  u, v, and w values
     *
     * @param values the values passed to the -o, -s, -t flag
     * @param {Object} option the Object of either the -o, -s, -t option
     * @param {Integer} defaultValue the Object of all image options
     */
    parse_ost(values: string[], option: UVW, defaultValue: number): void;
    /**
     * Parses the -o flag and updates the options object with the values.
     *
     * @param values the values passed to the -o flag
     * @param options the Object of all image options
     */
    parse_o(values: string[], options: TextureMapData): void;
    /**
     * Parses the -s flag and updates the options object with the values.
     *
     * @param values the values passed to the -s flag
     * @param options the Object of all image options
     */
    parse_s(values: string[], options: TextureMapData): void;
    /**
     * Parses the -t flag and updates the options object with the values.
     *
     * @param values the values passed to the -t flag
     * @param options the Object of all image options
     */
    parse_t(values: string[], options: TextureMapData): void;
    /**
     * Parses the -texres flag and updates the options object with the values.
     *
     * @param values the values passed to the -texres flag
     * @param options the Object of all image options
     */
    parse_texres(values: string[], options: TextureMapData): void;
    /**
     * Parses the -clamp flag and updates the options object with the values.
     *
     * @param values the values passed to the -clamp flag
     * @param options the Object of all image options
     */
    parse_clamp(values: string[], options: TextureMapData): void;
    /**
     * Parses the -bm flag and updates the options object with the values.
     *
     * @param values the values passed to the -bm flag
     * @param options the Object of all image options
     */
    parse_bm(values: string[], options: TextureMapData): void;
    /**
     * Parses the -imfchan flag and updates the options object with the values.
     *
     * @param values the values passed to the -imfchan flag
     * @param options the Object of all image options
     */
    parse_imfchan(values: string[], options: TextureMapData): void;
    /**
     * This only exists for relection maps and denotes the type of reflection.
     *
     * @param values the values passed to the -type flag
     * @param options the Object of all image options
     */
    parse_type(values: string[], options: TextureMapData): void;
    /**
     * Parses the texture's options and returns an options object with the info
     *
     * @param tokens all of the option tokens to pass to the texture
     * @return {Object} a complete object of objects to apply to the texture
     */
    parseOptions(tokens: string[]): TextureMapData;
    /**
     * Parses the given texture map line.
     *
     * @param tokens all of the tokens representing the texture
     * @return a complete object of objects to apply to the texture
     */
    parseMap(tokens: string[]): TextureMapData;
    /**
     * Parses the ambient map.
     *
     * @param tokens list of tokens for the map_Ka direcive
     */
    parse_map_Ka(tokens: string[]): void;
    /**
     * Parses the diffuse map.
     *
     * @param tokens list of tokens for the map_Kd direcive
     */
    parse_map_Kd(tokens: string[]): void;
    /**
     * Parses the specular map.
     *
     * @param tokens list of tokens for the map_Ks direcive
     */
    parse_map_Ks(tokens: string[]): void;
    /**
     * Parses the emissive map.
     *
     * @param tokens list of tokens for the map_Ke direcive
     */
    parse_map_Ke(tokens: string[]): void;
    /**
     * Parses the specular exponent map.
     *
     * @param tokens list of tokens for the map_Ns direcive
     */
    parse_map_Ns(tokens: string[]): void;
    /**
     * Parses the dissolve map.
     *
     * @param tokens list of tokens for the map_d direcive
     */
    parse_map_d(tokens: string[]): void;
    /**
     * Parses the anti-aliasing option.
     *
     * @param tokens list of tokens for the map_aat direcive
     */
    parse_map_aat(tokens: string[]): void;
    /**
     * Parses the bump map.
     *
     * @param tokens list of tokens for the map_bump direcive
     */
    parse_map_bump(tokens: string[]): void;
    /**
     * Parses the bump map.
     *
     * @param tokens list of tokens for the bump direcive
     */
    parse_bump(tokens: string[]): void;
    /**
     * Parses the disp map.
     *
     * @param tokens list of tokens for the disp direcive
     */
    parse_disp(tokens: string[]): void;
    /**
     * Parses the decal map.
     *
     * @param tokens list of tokens for the map_decal direcive
     */
    parse_decal(tokens: string[]): void;
    /**
     * Parses the refl map.
     *
     * @param tokens list of tokens for the refl direcive
     */
    parse_refl(tokens: string[]): void;
    /**
     * Parses the MTL file.
     *
     * Iterates line by line parsing each MTL directive.
     *
     * This function expects the first token in the line
     * to be a valid MTL directive. That token is then used
     * to try and run a method on this class. parse_[directive]
     * E.g., the `newmtl` directive would try to call the method
     * parse_newmtl. Each parsing function takes in the remaining
     * list of tokens and updates the currentMaterial class with
     * the attributes provided.
     */
    parse(): void;
}
