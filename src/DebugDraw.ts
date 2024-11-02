
import { mat4, ReadonlyMat4, ReadonlyVec3, vec2, vec3, vec4 } from "gl-matrix";

interface DebugDrawOptions {
    flags: DebugDrawFlags;
};

export const enum DebugDrawFlags {
    WorldSpace = 0,
    ViewSpace = 1 << 0,
    ScreenSpace = 1 << 1,

    DepthTint = 1 << 2,

    Default = DepthTint,
};

const SpaceMask = DebugDrawFlags.WorldSpace | DebugDrawFlags.ViewSpace | DebugDrawFlags.ScreenSpace;

interface GfxColor {
    r: number;
    g: number;
    b: number;
    a: number;
}

function align(n: number, multiple: number): number {
    const mask = (multiple - 1);
    return (n + mask) & ~mask;
}

function assert(b: boolean, message: string = ""): asserts b {
    if (!b) {
        console.error(new Error().stack);
        throw new Error(`Assert fail: ${message}`);
    }
}

// https://jcgt.org/published/0006/01/01/
function branchlessONB(dstA: vec3, dstB: vec3, n: ReadonlyVec3): void {
    const sign = n[2] >= 0.0 ? 1.0 : -1.0;
    const a = -1.0 / (sign + n[2]);
    const b = n[0] * n[1] * a;
    vec3.set(dstA, 1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
    vec3.set(dstB, b, sign + n[1] * n[1] * a, -n[1]);
}

export const Red = { r: 1, g: 0, b: 0, a: 1 };
export const Green = { r: 0, g: 1, b: 0, a: 1 };
export const Blue = { r: 0, g: 0, b: 1, a: 1 };
export const White = { r: 1, g: 1, b: 1, a: 1 };

const TAU = Math.PI * 2;

const Vec3UnitX: ReadonlyVec3 = vec3.fromValues(1, 0, 0);
const Vec3UnitY: ReadonlyVec3 = vec3.fromValues(0, 1, 0);
const Vec3UnitZ: ReadonlyVec3 = vec3.fromValues(0, 0, 1);

function getMatrixAxisX(dst: vec3, m: ReadonlyMat4): void {
    vec3.set(dst, m[0], m[1], m[2]);
}

function getMatrixAxisY(dst: vec3, m: ReadonlyMat4): void {
    vec3.set(dst, m[4], m[5], m[6]);
}

function getMatrixAxisZ(dst: vec3, m: ReadonlyMat4): void {
    vec3.set(dst, m[8], m[9], m[10]);
}

enum BehaviorType {
    Lines,
    LinesDepthWrite,
    LinesDepthTint,
    Solid,
    SolidDepthWrite,
    SolidDepthTint,
    Font,
    Count,
};

function fillVec3p(d: Float32Array, offs: number, v: ReadonlyVec3): number {
    d[offs + 0] = v[0];
    d[offs + 1] = v[1];
    d[offs + 2] = v[2];
    return 3;
}

function fillColor(d: Float32Array, offs: number, c: Readonly<GfxColor>, a: number = c.a): number {
    d[offs + 0] = c.r;
    d[offs + 1] = c.g;
    d[offs + 2] = c.b;
    d[offs + 3] = a;
    return 4;
}

class BufferPage {
    public vertexData: Float32Array;
    public indexData: Uint16Array;
    public vertexBuffer: GPUBuffer;
    public indexBuffer: GPUBuffer;

    public vertexDataOffs = 0;
    public vertexStride = 3+4;
    public indexDataOffs = 0;

    public lifetime = 3;

    constructor(device: GPUDevice, public readonly pso: GPURenderPipeline, public readonly behaviorType: BehaviorType, vertexCount: number, indexCount: number) {
        this.vertexData = new Float32Array(vertexCount * this.vertexStride);
        this.vertexBuffer = device.createBuffer({
            size: this.vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            label: `DebugDraw BufferPage Vertex`,
        });

        this.indexData = new Uint16Array(align(indexCount, 2));
        this.indexBuffer = device.createBuffer({
            size: this.indexData.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            label: `DebugDraw BufferPage Index`,
        });
    }

    public getCurrentVertexID() { return (this.vertexDataOffs / this.vertexStride) >>> 0; }
    private remainVertex() { return this.vertexData.length - this.vertexDataOffs; }
    private remainIndex() { return this.indexData.length - this.indexDataOffs; }
    public getDrawCount() { return this.indexDataOffs; }

    public canAllocVertices(vertexCount: number, indexCount: number): boolean {
        return (vertexCount * this.vertexStride) <= this.remainVertex() && indexCount <= this.remainIndex();
    }

    public vertexPCF(v: ReadonlyVec3, c: GfxColor, options: DebugDrawOptions): void {
        this.vertexDataOffs += fillVec3p(this.vertexData, this.vertexDataOffs, v);
        const flags = options.flags;
        // encode flags in alpha
        const alpha = (1.0 - c.a) + flags;
        this.vertexDataOffs += fillColor(this.vertexData, this.vertexDataOffs, c, alpha);
    }

    public index(n: number): void {
        this.indexData[this.indexDataOffs++] = n;
    }

    public uploadData(device: GPUDevice): boolean {
        if (this.vertexDataOffs === 0) {
            if (--this.lifetime === 0) {
                this.destroy();
                return false;
            } else {
                return true;
            }
        }

        const queue = device.queue;
        queue.writeBuffer(this.vertexBuffer, 0, new Uint8Array(this.vertexData.buffer), 0, this.vertexDataOffs * 4);
        queue.writeBuffer(this.indexBuffer, 0, new Uint8Array(this.indexData.buffer), 0, this.indexDataOffs * 2);
        return true;
    }

    public endFrame(): boolean {
        this.vertexDataOffs = 0;
        this.indexDataOffs = 0;
        this.lifetime = 3;
        return true;
    }

    public destroy(): void {
        this.vertexBuffer.destroy();
        if (this.indexBuffer !== null)
            this.indexBuffer.destroy();
    }
}

class FontTexture {
    private characters: string = '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~';
    private font = `24px Consolas`;
    public gpuTexture: GPUTexture;
    public cellWidth: number;
    public cellHeight: number;
    public advanceX: number;
    public advanceY: number;
    public strokeWidth = 3;
    public padding: number = 1;
    public numCellsPerRow: number;
    public fontParams = new Float32Array(4);

    constructor(device: GPUDevice) {
        this.rasterize(device);
    }

    public getCharacterIndex(c: string): number {
        return this.characters.indexOf(c);
    }

    public getCellX(index: number): number {
        return (index % this.numCellsPerRow);
    }

    public getCellY(index: number): number {
        return (index / this.numCellsPerRow) | 0;
    }

    private rasterize(device: GPUDevice): void {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;
        const ctx = canvas.getContext('2d')!;
        ctx.font = this.font;

        ctx.textAlign = `left`;
        ctx.textBaseline = `top`;

        ctx.fillStyle = `black`;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        let cellWidth = 0;
        let cellHeight = 0;
        for (const c of this.characters) {
            const measure = ctx.measureText(c);
            const w = Math.ceil(measure.actualBoundingBoxRight + measure.actualBoundingBoxLeft);
            const h = Math.ceil(measure.actualBoundingBoxDescent + measure.actualBoundingBoxAscent);
            cellWidth = Math.max(cellWidth, w);
            cellHeight = Math.max(cellHeight, h);
        }

        this.advanceX = cellWidth;
        this.advanceY = cellHeight;

        // Padding
        const extra = this.padding + this.strokeWidth * 0.5;
        cellWidth += extra * 2;
        cellHeight += extra * 2;

        ctx.strokeStyle = `rgba(255, 255, 255, 0.5)`;
        ctx.lineWidth = this.strokeWidth;
        ctx.fillStyle = `rgba(255, 255, 255, 1.0)`;
        const numCellsPerRow = (canvas.width / cellWidth) | 0;
        let cellY = 0;
        let cellX = 0;
        for (const char of this.characters) {
            ctx.strokeText(char, cellX * cellWidth + extra, cellY * cellHeight + extra);
            ctx.fillText(char, cellX * cellWidth + extra, cellY * cellHeight + extra);
            cellX++;
    
            if (cellX === numCellsPerRow) {
                cellX = 0;
                cellY++;
            }
        }

        this.gpuTexture = device.createTexture({
            size: canvas,
            format: 'r8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            label: `DebugFont`,
        });

        this.cellWidth = cellWidth;
        this.cellHeight = cellHeight;
        this.numCellsPerRow = numCellsPerRow;
        this.fontParams[0] = cellWidth / canvas.width;
        this.fontParams[1] = cellHeight / canvas.height;
        this.fontParams[2] = this.padding / canvas.width;
        this.fontParams[3] = this.padding / canvas.width;

        device.queue.copyExternalImageToTexture({ source: canvas }, { texture: this.gpuTexture }, canvas);
    }

    public destroy(): void {
        this.gpuTexture.destroy();
    }
}

export class DebugDraw {
    private pages: BufferPage[] = [];
    private defaultPageVertexCount = 1024;
    private defaultPageIndexCount = 1024;
    private shaderModule: GPUShaderModule;
    private currentPage: BufferPage | null = null; // for the batch system
    private pso: GPURenderPipeline[] = [];
    private bindGroupLayout: GPUBindGroupLayout;
    private pipelineLayout: GPUPipelineLayout;
    private uniformBuffer: GPUBuffer;
    private dummyDepthBuffer: GPUTexture;
    private debugDrawGPU: DebugDrawGPU;
    private lineThickness = 3;
    private fontTexture: FontTexture;
    private fontSampler: GPUSampler;
    private screenWidth = 1;
    private screenHeight = 1;
    private screenPrintPos = vec2.create();

    public static scratchVec3 = [vec3.create(), vec3.create(), vec3.create(), vec3.create()];

    constructor(private device: GPUDevice, private colorTextureFormat: GPUTextureFormat) {
        this.shaderModule = this.createShaderModule();
        this.uniformBuffer = device.createBuffer({ usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, size: 64+64+16+16, label: `DebugDraw Uniforms` });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { } },
                { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
            ],
            label: 'DebugDraw',
        });
        this.pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout],
            label: 'DebugDraw',
        });

        for (let i = 0; i < BehaviorType.Count; i++)
            this.pso[i] = this.createPipeline(i);

        this.dummyDepthBuffer = device.createTexture({ format: 'depth24plus', size: [1, 1], usage: GPUTextureUsage.TEXTURE_BINDING });

        this.fontTexture = new FontTexture(this.device);
        this.debugDrawGPU = new DebugDrawGPU(this, this.device);

        this.fontSampler = device.createSampler({
            addressModeU: `clamp-to-edge`,
            addressModeV: `clamp-to-edge`,
            minFilter: `linear`,
            magFilter: `linear`,
            label: `DebugDraw Font`,
        })
    }

    public getGPUBuffer(): GPUBuffer {
        return this.debugDrawGPU.gpuBuffer;
    }

    private createShaderModule(): GPUShaderModule {
        const code = `
struct ViewData {
    clip_from_view: mat4x4f,
    view_from_world: mat4x4f,
    misc: vec4f, // screen_size_inv
    font_params: vec4f, // cell_width, cell_height
};

@id(0) override behavior_type: u32 = ${BehaviorType.LinesDepthWrite};

@group(0) @binding(0) var<uniform> view_data: ViewData;
@group(0) @binding(1) var font_texture: texture_2d<f32>;
@group(0) @binding(2) var font_sampler: sampler;
@group(0) @binding(3) var depth_buffer: texture_depth_2d;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) @interpolate(flat, either) flags: u32,
};

@vertex
fn main_vs(@location(0) position: vec3f, @location(1) color_: vec4f, @builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Flags are packed in the integer component of the alpha.
    var color = color_;
    var flags = u32(color.a);

    var space = flags & ${SpaceMask};
    if (space == ${DebugDrawFlags.WorldSpace}) {
        out.position = view_data.clip_from_view * view_data.view_from_world * vec4f(position, 1.0f);
    } else if (space == ${DebugDrawFlags.ViewSpace}) {
        out.position = view_data.clip_from_view * vec4f(position, 1.0f);
    } else if (space == ${DebugDrawFlags.ScreenSpace}) {
        out.position = vec4f(position.xy, 0.1f, 1.0f);
    }

    if (behavior_type == ${BehaviorType.Font}) {
        // Encoding for font characters is a bit silly. The color takes the range 0.0f - 1.0f,
        //   index 0 has the range [0.0f, 1.0f]
        //   index 1 has the range [2.0f, 3.0f]
        var base = vec2i(color.rg);
        var font_character_index = base / 2;

        color.r = color.r - f32(font_character_index.x * 2);
        color.g = color.g - f32(font_character_index.y * 2);

        var tl_uv = vec2f(font_character_index) * view_data.font_params.xy + view_data.font_params.zw;
        var br_uv = vec2f(font_character_index + vec2i(1, 1)) * view_data.font_params.xy - view_data.font_params.zw;

        var quad_vertex = vertex_index & 3;
        out.uv.x = select(tl_uv.x, br_uv.x, (quad_vertex == 1 || quad_vertex == 2));
        out.uv.y = select(tl_uv.y, br_uv.y, (quad_vertex == 2 || quad_vertex == 3));
    }

    if (behavior_type == ${BehaviorType.Lines} || behavior_type == ${BehaviorType.LinesDepthTint} || behavior_type == ${BehaviorType.LinesDepthWrite}) {
        // Hacky thick line support.
        if (instance_index >= 1) {
            var line_idx = instance_index - 1;
            var offs: vec2f;
            offs.x = select(1.0f, -1.0f, (line_idx & 1u) != 0u);
            offs.y = select(1.0f, -1.0f, (line_idx & 2u) != 0u);
            offs *= f32((line_idx / 4) + 1);

            var inv_screen_size = view_data.misc.xy;
            out.position += vec4f(offs * inv_screen_size, 0.0f, 0.0f) * out.position.w;
        }
    }

    out.flags = flags;
    var alpha = 1.0f - fract(color.a);
    out.color = vec4f(color.rgb * alpha, alpha);
    return out;
}

@fragment
fn main_ps(vertex: VertexOutput) -> @location(0) vec4f {
    var color = vertex.color;

    if (behavior_type == ${BehaviorType.Font}) {
        var coverage = textureSample(font_texture, font_sampler, vertex.uv).r;
        // Map range 0.0f - 0.5f to outline color (solid black), and 0.5f to 1.0f range to color.
        var transparent = vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        var outline_color = vec4f(0.0f, 0.0f, 0.0f, 1.0f);
        if (coverage > 0.5f) {
            color = mix(outline_color, color, ((coverage - 0.5f) * 2.0f));
        } else {
            color = mix(transparent, outline_color, coverage * 2.0f);
        }
    }

    if (behavior_type == ${BehaviorType.LinesDepthTint} || behavior_type == ${BehaviorType.SolidDepthTint}) {
        // Do manual depth testing so we can do a depth tint.
        var depth = textureLoad(depth_buffer, vec2u(vertex.position.xy), 0);

        if (depth > vertex.position.z) {
            color *= 0.15f;
        }
    }

    return color;
}
`;
        return this.device.createShaderModule({ code, label: `DebugDraw` });
    }

    private createPipeline(behaviorType: BehaviorType): GPURenderPipeline {
        // Create the PSO based on our state.
        const isLines = behaviorType === BehaviorType.LinesDepthWrite || behaviorType === BehaviorType.LinesDepthTint;
        const isDepthWrite = behaviorType === BehaviorType.LinesDepthWrite || behaviorType === BehaviorType.SolidDepthWrite;
        const isDepthTint = behaviorType === BehaviorType.LinesDepthTint || behaviorType === BehaviorType.SolidDepthTint;

        const desc: GPURenderPipelineDescriptor = {
            layout: this.pipelineLayout,
            primitive: {
                topology: isLines ? 'line-list' : 'triangle-list',
            },
            vertex: {
                module: this.shaderModule,
                constants: {
                    0: behaviorType,
                },
                entryPoint: 'main_vs',
                buffers: [{
                    attributes: [
                        { shaderLocation: 0, format: 'float32x3', offset: 0 },
                        { shaderLocation: 1, format: 'float32x4', offset: 3 * 4 },
                    ],
                    arrayStride: 7 * 4,
                }],
            },
            fragment: {
                module: this.shaderModule,
                constants: {
                    0: behaviorType,
                },
                entryPoint: 'main_ps',
                targets: [{ format: this.colorTextureFormat, blend: {
                    color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha', },
                    alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha', },
                } }],
            },
            depthStencil: isDepthTint ? undefined : {
                format: 'depth24plus',
                depthCompare: 'greater',
                depthWriteEnabled: isDepthWrite,
            },
            label: `DebugDraw ${BehaviorType[behaviorType]}`,
        };

        return this.device.createRenderPipeline(desc);
    }

    private getBehaviorType(isLines: boolean, isOpaque: boolean, options: DebugDrawOptions) {
        const isDepthTint = options.flags & DebugDrawFlags.DepthTint;
        if (isLines)
            return isDepthTint ? BehaviorType.LinesDepthTint : isOpaque ? BehaviorType.LinesDepthWrite : BehaviorType.Lines;
        else
            return isDepthTint ? BehaviorType.SolidDepthTint : isOpaque ? BehaviorType.SolidDepthWrite : BehaviorType.Lines;
    }

    private beginBatch(behaviorType: BehaviorType, vertexCount: number, indexCount: number): void {
        this.currentPage = this.findPage(behaviorType, vertexCount, indexCount);
    }

    public beginBatchLine(numSegments: number, color: GfxColor, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const behaviorType = this.getBehaviorType(true, color.a >= 1.0, options);
        this.beginBatch(behaviorType, numSegments * 2, numSegments * 2);
    }

    public endBatch(): void {
        assert(this.currentPage !== null);
        this.currentPage = null;
    }

    private findPage(behaviorType: BehaviorType, vertexCount: number, indexCount: number): BufferPage {
        if (this.currentPage !== null) {
            assert(this.currentPage.behaviorType === behaviorType);
            assert(this.currentPage.canAllocVertices(vertexCount, indexCount));
            return this.currentPage;
        }

        for (let i = 0; i < this.pages.length; i++) {
            const page = this.pages[i];
            if (page.behaviorType === behaviorType && page.canAllocVertices(vertexCount, indexCount))
                return page;
        }

        vertexCount = align(vertexCount, this.defaultPageVertexCount);
        indexCount = align(indexCount, this.defaultPageIndexCount);
        const page = new BufferPage(this.device, this.pso[behaviorType], behaviorType, vertexCount, indexCount);
        this.pages.push(page);
        return page;
    }

    public drawLine(p0: ReadonlyVec3, p1: ReadonlyVec3, color0: GfxColor, color1 = color0, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(this.getBehaviorType(true, color0.a >= 1.0 && color1.a >= 1.0, options), 2, 2);

        const baseVertex = page.getCurrentVertexID();
        page.vertexPCF(p0, color0, options);
        page.vertexPCF(p1, color1, options);

        for (let i = 0; i < 2; i++)
            page.index(baseVertex + i);
    }

    public drawVector(p0: ReadonlyVec3, dir: ReadonlyVec3, mag: number, color0: GfxColor, color1 = color0, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        vec3.scaleAndAdd(DebugDraw.scratchVec3[0], p0, dir, mag);
        this.drawLine(p0, DebugDraw.scratchVec3[0], color0, color1, options);
    }

    public drawBasis(m: ReadonlyMat4, mag = 100, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(this.getBehaviorType(true, true, options), 6, 6);

        const baseVertex = page.getCurrentVertexID();
        mat4.getTranslation(DebugDraw.scratchVec3[0], m);

        // X
        getMatrixAxisX(DebugDraw.scratchVec3[1], m);
        vec3.scaleAndAdd(DebugDraw.scratchVec3[1], DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], mag);
        page.vertexPCF(DebugDraw.scratchVec3[0], Red, options);
        page.vertexPCF(DebugDraw.scratchVec3[1], Red, options);

        // Y
        getMatrixAxisY(DebugDraw.scratchVec3[1], m);
        vec3.scaleAndAdd(DebugDraw.scratchVec3[1], DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], mag);
        page.vertexPCF(DebugDraw.scratchVec3[0], Green, options);
        page.vertexPCF(DebugDraw.scratchVec3[1], Green, options);

        // Z
        getMatrixAxisZ(DebugDraw.scratchVec3[1], m);
        vec3.scaleAndAdd(DebugDraw.scratchVec3[1], DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], mag);
        page.vertexPCF(DebugDraw.scratchVec3[0], Blue, options);
        page.vertexPCF(DebugDraw.scratchVec3[1], Blue, options);

        for (let i = 0; i < 6; i++)
            page.index(baseVertex + i);
    }

    public drawLocator(center: ReadonlyVec3, mag: number, color: GfxColor, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(this.getBehaviorType(true, color.a >= 1.0, options), 6, 6);

        const baseVertex = page.getCurrentVertexID();
        for (let i = 0; i < 3; i++) {
            vec3.copy(DebugDraw.scratchVec3[0], center);
            vec3.copy(DebugDraw.scratchVec3[1], center);
            DebugDraw.scratchVec3[0][i] -= mag;
            DebugDraw.scratchVec3[1][i] += mag;

            page.vertexPCF(DebugDraw.scratchVec3[0], color, options);
            page.vertexPCF(DebugDraw.scratchVec3[1], color, options);
        }

        for (let i = 0; i < 6; i++)
            page.index(baseVertex + i);
    }

    public drawDiscLineN(center: ReadonlyVec3, n: ReadonlyVec3, r: number, color: GfxColor, sides = 32, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        branchlessONB(DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], n);
        this.drawDiscSolidRU(center, DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], r, color, sides, options);
    }

    public drawDiscLineRU(center: ReadonlyVec3, right: ReadonlyVec3, up: ReadonlyVec3, r: number, color: GfxColor, sides = 32, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(this.getBehaviorType(true, color.a >= 1.0, options), sides, sides * 2);

        const baseVertex = page.getCurrentVertexID();
        const s = DebugDraw.scratchVec3[2];
        for (let i = 0; i < sides; i++) {
            const theta = i / sides * TAU;
            const sin = Math.sin(theta) * r, cos = Math.cos(theta) * r;
            s[0] = center[0] + right[0] * cos + up[0] * sin;
            s[1] = center[1] + right[1] * cos + up[1] * sin;
            s[2] = center[2] + right[2] * cos + up[2] * sin;
            page.vertexPCF(s, color, options);

            page.index(baseVertex + i);
            page.index(baseVertex + ((i === sides - 1) ? 0 : i + 1));
        }
    }

    public drawSphereLine(center: ReadonlyVec3, r: number, color: GfxColor, sides = 32, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }) {
        this.drawDiscLineRU(center, Vec3UnitX, Vec3UnitY, r, color, sides, options);
        this.drawDiscLineRU(center, Vec3UnitX, Vec3UnitZ, r, color, sides, options);
        this.drawDiscLineRU(center, Vec3UnitY, Vec3UnitZ, r, color, sides, options);
    }

    public drawDiscSolidN(center: ReadonlyVec3, n: ReadonlyVec3, r: number, color: GfxColor, sides = 32, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        branchlessONB(DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], n);
        this.drawDiscSolidRU(center, DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], r, color, sides, options);
    }

    public drawDiscSolidRU(center: ReadonlyVec3, right: ReadonlyVec3, up: ReadonlyVec3, r: number, color: GfxColor, sides = 32, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(this.getBehaviorType(false, color.a >= 1.0, options), sides + 1, sides * 3);

        const baseVertex = page.getCurrentVertexID();
        page.vertexPCF(center, color, options);
        const s = DebugDraw.scratchVec3[2];
        for (let i = 0; i < sides - 1; i++) {
            const theta = i / sides * TAU;
            const sin = Math.sin(theta) * r, cos = Math.cos(theta) * r;
            s[0] = center[0] + right[0] * cos + up[0] * sin;
            s[1] = center[1] + right[1] * cos + up[1] * sin;
            s[2] = center[2] + right[2] * cos + up[2] * sin;
            page.vertexPCF(s, color, options);
        }

        // construct trifans by hand
        for (let i = 0; i < sides - 2; i++) {
            page.index(baseVertex);
            page.index(baseVertex + 1 + i);
            page.index(baseVertex + 2 + i);
        }

        page.index(baseVertex);
        page.index(baseVertex + sides - 1);
        page.index(baseVertex + 1);
    }

    private rectCorner(dst: vec3[], center: ReadonlyVec3, right: ReadonlyVec3, up: ReadonlyVec3, rightMag: number, upMag: number): void {
        // TL, TR, BL, BR
        for (let i = 0; i < 4; i++) {
            const signX = i & 1 ? -1 : 1;
            const signY = i & 2 ? -1 : 1;
            const s = dst[i];
            vec3.scaleAndAdd(s, center, right, signX * rightMag);
            vec3.scaleAndAdd(s, s, up, signY * upMag);
        }
    }

    public drawTriSolidP(p0: ReadonlyVec3, p1: ReadonlyVec3, p2: ReadonlyVec3, color: GfxColor, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(this.getBehaviorType(false,color.a >= 1.0,  options), 3, 3);

        const baseVertex = page.getCurrentVertexID();
        page.vertexPCF(p0, color, options);
        page.vertexPCF(p1, color, options);
        page.vertexPCF(p2, color, options);

        for (let i = 0; i < 3; i++)
            page.index(baseVertex + i);
    }

    public drawRectLineP(p0: ReadonlyVec3, p1: ReadonlyVec3, p2: ReadonlyVec3, p3: ReadonlyVec3, color: GfxColor, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(this.getBehaviorType(true, color.a >= 1.0, options), 4, 8);

        const baseVertex = page.getCurrentVertexID();
        page.vertexPCF(p0, color, options);
        page.vertexPCF(p1, color, options);
        page.vertexPCF(p3, color, options);
        page.vertexPCF(p2, color, options);

        for (let i = 0; i < 4; i++) {
            page.index(baseVertex + i);
            page.index(baseVertex + ((i + 1) & 3));
        }
    }

    public drawRectLineRU(center: ReadonlyVec3, right: ReadonlyVec3, up: ReadonlyVec3, rightMag: number, upMag: number, color: GfxColor, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        this.rectCorner(DebugDraw.scratchVec3, center, right, up, rightMag, upMag);
        this.drawRectLineP(DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], DebugDraw.scratchVec3[2], DebugDraw.scratchVec3[3], color, options);
    }

    public drawRectSolidP(p0: ReadonlyVec3, p1: ReadonlyVec3, p2: ReadonlyVec3, p3: ReadonlyVec3, color: GfxColor, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(this.getBehaviorType(false, color.a >= 1.0, options), 4, 6);

        const baseVertex = page.getCurrentVertexID();
        page.vertexPCF(p0, color, options);
        page.vertexPCF(p1, color, options);
        page.vertexPCF(p3, color, options);
        page.vertexPCF(p2, color, options);

        page.index(baseVertex + 0);
        page.index(baseVertex + 1);
        page.index(baseVertex + 2);
        page.index(baseVertex + 0);
        page.index(baseVertex + 2);
        page.index(baseVertex + 3);
    }

    public drawRectSolidRU(center: ReadonlyVec3, right: ReadonlyVec3, up: ReadonlyVec3, rightMag: number, upMag: number, color: GfxColor, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        this.rectCorner(DebugDraw.scratchVec3, center, right, up, rightMag, upMag);
        this.drawRectSolidP(DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], DebugDraw.scratchVec3[2], DebugDraw.scratchVec3[3], color, options);
    }

    // TODO(jstpierre): Clean this up and make it more generic.
    public screenPrintText(s: string, size: number = 1, color: GfxColor = White): void {
        const numQuads = s.length;
        const page = this.findPage(BehaviorType.Font, numQuads * 4, numQuads * 6);

        const options = { flags: DebugDrawFlags.ScreenSpace };

        let baseVertex = page.getCurrentVertexID();

        const advanceX = size * this.fontTexture.advanceX;
        const advanceY = size * this.fontTexture.advanceY;

        const sizeX = advanceX / this.screenWidth * 2;
        const sizeY = -advanceY / this.screenHeight * 2;

        const x = this.screenPrintPos[0] / this.screenWidth * 2 - 1;
        const y = this.screenPrintPos[1] / this.screenHeight * -2 + 1;

        const p = vec3.set(DebugDraw.scratchVec3[0], x, y, 0.0);
        const colorR = color.r;
        const colorG = color.g;
        for (let i = 0; i < numQuads; i++) {
            const char = s.charAt(i);
            if (char === '\n') {
                p[0] = x;
                p[1] += sizeY;
                this.screenPrintPos[1] += advanceY;
                continue;
            }

            const index = this.fontTexture.getCharacterIndex(char);

            if (index >= 0) {
                const cellX = this.fontTexture.getCellX(index);
                const cellY = this.fontTexture.getCellY(index);
                color.r = colorR + cellX * 2;
                color.g = colorG + cellY * 2;

                // TL, TR, BR, BL
                page.vertexPCF(p, color, options);

                vec3.copy(DebugDraw.scratchVec3[1], p);
                DebugDraw.scratchVec3[1][0] += sizeX;
                page.vertexPCF(DebugDraw.scratchVec3[1], color, options);

                DebugDraw.scratchVec3[1][1] += sizeY;
                page.vertexPCF(DebugDraw.scratchVec3[1], color, options);

                vec3.copy(DebugDraw.scratchVec3[1], p);
                DebugDraw.scratchVec3[1][1] += sizeY;
                page.vertexPCF(DebugDraw.scratchVec3[1], color, options);

                page.index(baseVertex + 0);
                page.index(baseVertex + 1);
                page.index(baseVertex + 2);
                page.index(baseVertex + 0);
                page.index(baseVertex + 2);
                page.index(baseVertex + 3);
                baseVertex += 4;
            }

            p[0] += sizeX;
        }

        color.r = colorR;
        color.g = colorG;
        this.screenPrintPos[1] += advanceY;
    }

    private uploadPages(): number {
        let types = 0;
        for (let i = 0; i < this.pages.length; i++) {
            const page = this.pages[i];
            if (!page.uploadData(this.device))
                this.pages.splice(i--, 1);
            types |= 1 << page.behaviorType;
        }
        return types;
    }

    private drawPages(pass: GPURenderPassEncoder, behaviorTypes: number): void {
        for (let i = 0; i < this.pages.length; i++) {
            const page = this.pages[i];
            if (!((1 << page.behaviorType) & behaviorTypes))
                continue;

            const indexCount = page.getDrawCount();
            if (indexCount === 0)
                continue;

            pass.setPipeline(page.pso);
            pass.setVertexBuffer(0, page.vertexBuffer);
            pass.setIndexBuffer(page.indexBuffer, 'uint16');

            const isLines = page.behaviorType === BehaviorType.LinesDepthWrite || page.behaviorType === BehaviorType.LinesDepthTint;
            const instanceCount = isLines ? 1 + (this.lineThickness - 1) * 4 : 1;
            pass.drawIndexed(indexCount, instanceCount);

            // Reset for next frame.
            page.endFrame();
        }
    }

    public beginFrame(screenWidth: number, screenHeight: number, mouseX: number, mouseY: number, mouseButtons: number): void {
        this.screenWidth = screenWidth;
        this.screenHeight = screenHeight;
        // Initialize to 10, 10
        vec2.set(this.screenPrintPos, 10, 10);
        this.debugDrawGPU.beginFrame(mouseX, mouseY, mouseButtons);
    }

    public endFrame(cmd: GPUCommandEncoder, clipFromViewMatrix: ReadonlyMat4, viewFromWorldMatrix: ReadonlyMat4, colorTextureView: GPUTextureView, depthTextureView: GPUTextureView): void {
        this.debugDrawGPU.endFrame(cmd);

        const behaviorTypes = this.uploadPages();
        if (behaviorTypes === 0)
            return;

        const data = new Float32Array(this.uniformBuffer.size / 4);
        data.set(clipFromViewMatrix, 0);
        data.set(viewFromWorldMatrix, 16);
        data[32] = 1 / this.screenWidth;
        data[33] = 1 / this.screenHeight;
        data.set(this.fontTexture.fontParams, 36);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

        // First, check if we have any solid stuff (needs proper depth).
        if (behaviorTypes & ((1 << BehaviorType.Lines) | (1 << BehaviorType.LinesDepthWrite) | (1 << BehaviorType.Solid) | (1 << BehaviorType.SolidDepthWrite) | (1 << BehaviorType.Font))) {
            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: this.fontTexture.gpuTexture.createView() },
                    { binding: 2, resource: this.fontSampler },
                    { binding: 3, resource: this.dummyDepthBuffer.createView() },
                ],
                label: `DebugDraw DepthWrite`,
            });

            const renderPass = cmd.beginRenderPass({
                colorAttachments: [{ view: colorTextureView, loadOp: 'load', storeOp: 'store' }],
                depthStencilAttachment: { view: depthTextureView, depthLoadOp: 'load', depthStoreOp: 'store' },
                label: `DebugDraw DepthTint`,
            });

            renderPass.setBindGroup(0, bindGroup);
            this.drawPages(renderPass, (1 << BehaviorType.Lines) | (1 << BehaviorType.LinesDepthWrite) | (1 << BehaviorType.Solid) | (1 << BehaviorType.SolidDepthWrite) | (1 << BehaviorType.Font));
            renderPass.end();
        }

        if (behaviorTypes & ((1 << BehaviorType.LinesDepthTint) | (1 << BehaviorType.SolidDepthTint))) {
            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: this.fontTexture.gpuTexture.createView() },
                    { binding: 2, resource: this.fontSampler },
                    { binding: 3, resource: depthTextureView },
                ],
                label: `DebugDraw DepthTint`,
            });

            const renderPass = cmd.beginRenderPass({
                colorAttachments: [{ view: colorTextureView, loadOp: 'load', storeOp: 'store' }],
                label: `DebugDraw DepthTint`,
            });

            renderPass.setBindGroup(0, bindGroup);
            this.drawPages(renderPass, (1 << BehaviorType.LinesDepthTint) | (1 << BehaviorType.SolidDepthTint));
            renderPass.end();
        }
    }

    public destroy(): void {
        for (let i = 0; i < this.pages.length; i++)
            this.pages[i].destroy();
    }
}

class DebugDrawGPU {
    private framePool: GPUBuffer[] = [];
    private submittedFrames: GPUBuffer[] = [];
    private mouseHoverPos = vec2.fromValues(-1, -1);
    private mousePressPos = vec2.fromValues(-1, -1);
    public gpuBuffer: GPUBuffer;

    constructor(private debugDraw: DebugDraw, private device: GPUDevice, private size: number = 8196) {
        this.gpuBuffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label: `DebugDrawGPU GPU Buffer` });
    }

    private parseDebugDrawBuffer(view: DataView): void {
        const count = view.getUint32(0 * 4, true);

        let flags: DebugDrawFlags = DebugDrawFlags.Default;

        let offs = 8;
        const end = offs + count;
        while (offs < end) {
            const messageType: DebugDrawMessageType = view.getUint32(offs++ * 4, true);
            switch (messageType) {
            case DebugDrawMessageType.setDepthTint:
                {
                    const v = !!view.getUint32(offs++ * 4, true);
                    flags &= ~DebugDrawFlags.DepthTint;
                    flags |= v ? DebugDrawFlags.DepthTint : 0;
                }
                break;
            case DebugDrawMessageType.setSpace:
                {
                    const space = view.getUint32(offs++ * 4, true);
                    flags &= ~SpaceMask;
                    flags |= space;
                }
                break;
            case DebugDrawMessageType.drawSphere:
                {
                    const p0 = vec3.fromValues(view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true));
                    const r = view.getFloat32(offs++ * 4, true);
                    const color = { r: view.getFloat32(offs++ * 4, true), g: view.getFloat32(offs++ * 4, true), b: view.getFloat32(offs++ * 4, true), a: view.getFloat32(offs++ * 4, true) };
                    this.debugDraw.drawSphereLine(p0, r, color, 32, { flags });
                }
                break;
            case DebugDrawMessageType.drawLine:
                {
                    const p0 = vec3.fromValues(view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true));
                    const p1 = vec3.fromValues(view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true));
                    const color = { r: view.getFloat32(offs++ * 4, true), g: view.getFloat32(offs++ * 4, true), b: view.getFloat32(offs++ * 4, true), a: view.getFloat32(offs++ * 4, true) };
                    this.debugDraw.drawLine(p0, p1, color, color, { flags });
                }
                break;
            case DebugDrawMessageType.drawLocator:
                {
                    const p0 = vec3.fromValues(view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true));
                    const mag = view.getFloat32(offs++ * 4, true);
                    const color = { r: view.getFloat32(offs++ * 4, true), g: view.getFloat32(offs++ * 4, true), b: view.getFloat32(offs++ * 4, true), a: view.getFloat32(offs++ * 4, true) };
                    this.debugDraw.drawLocator(p0, mag, color, { flags });
                }
                break;
            case DebugDrawMessageType.screenPrintFloat1:
                {
                    const v0 = view.getFloat32(offs++ * 4, true);
                    this.debugDraw.screenPrintText(`${v0.toFixed(2)}`);
                }
                break;
            case DebugDrawMessageType.screenPrintFloat2:
                {
                    const v0 = view.getFloat32(offs++ * 4, true), v1 = view.getFloat32(offs++ * 4, true);
                    this.debugDraw.screenPrintText(`${v0.toFixed(2)}, ${v1.toFixed(2)}`);
                }
                break;
            case DebugDrawMessageType.screenPrintFloat3:
                {
                    const v0 = view.getFloat32(offs++ * 4, true), v1 = view.getFloat32(offs++ * 4, true), v2 = view.getFloat32(offs++ * 4, true);
                    this.debugDraw.screenPrintText(`${v0.toFixed(2)}, ${v1.toFixed(2)}, ${v2.toFixed(2)}`);
                }
                break;
            case DebugDrawMessageType.screenPrintFloat4:
                {
                    const v0 = view.getFloat32(offs++ * 4, true), v1 = view.getFloat32(offs++ * 4, true), v2 = view.getFloat32(offs++ * 4, true), v3 = view.getFloat32(offs++ * 4, true);
                    this.debugDraw.screenPrintText(`${v0.toFixed(2)}, ${v1.toFixed(2)}, ${v2.toFixed(2)}, ${v3.toFixed(2)}`);
                }
                break;
            }
        }
    }

    private mapSubmittedFrames(): void {
        for (let i = 0; i < this.submittedFrames.length; i++) {
            const frame = this.submittedFrames[i];
            if (frame.mapState === 'unmapped') {
                frame.mapAsync(GPUMapMode.READ);
            }
        }
    }

    private updateFromFinishedFrames(): void {
        for (let i = 0; i < this.submittedFrames.length; i++) {
            const frame = this.submittedFrames[i];
            if (frame.mapState === 'mapped') {
                const results = new DataView(frame.getMappedRange());
                this.parseDebugDrawBuffer(results);
                frame.unmap();

                this.submittedFrames.splice(i--, 1);
                this.framePool.push(frame);
            }
        }
    }

    private getFrame(): GPUBuffer {
        if (this.framePool.length > 0)
            return this.framePool.pop()!;
        else
            return this.device.createBuffer({ size: this.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: `DebugDrawGPU CPU Readback Buffer` });
    }

    public beginFrame(mouseX: number, mouseY: number, mouseButtons: number): void {
        this.mapSubmittedFrames();

        vec2.set(this.mouseHoverPos, mouseX, mouseY);
        if (mouseButtons !== 0)
            vec2.set(this.mousePressPos, mouseX, mouseY);

        // Overwrite the header with new data.
        const headerData = new Uint32Array(8);
        headerData[4] = this.mouseHoverPos[0];
        headerData[5] = this.mouseHoverPos[1];
        headerData[6] = this.mousePressPos[0];
        headerData[7] = this.mousePressPos[1];
        this.device.queue.writeBuffer(this.gpuBuffer, 0, headerData);
    }

    public endFrame(cmd: GPUCommandEncoder): void {
        this.mapSubmittedFrames();
        this.updateFromFinishedFrames();

        const readbackFrame = this.getFrame();
        cmd.copyBufferToBuffer(this.gpuBuffer, 0, readbackFrame, 0, this.size);

        this.submittedFrames.push(readbackFrame);
    }
}

const enum DebugDrawMessageType {
    setDepthTint,
    setSpace,
    drawLine,
    drawSphere,
    drawLocator,
    screenPrintFloat1,
    screenPrintFloat2,
    screenPrintFloat3,
    screenPrintFloat4,
}

export const gpuShaderCode = `
struct DebugDraw_Buffer {
    @size(16)
    size: atomic<u32>, // u32 count

    mouse_hover_pos: vec2i, // pixel position of the current mouse
    mouse_press_pos: vec2i, // pixel position of the last held mouse

    data: array<u32>,
};

// user code needs to write:
// @group(x) @binding(y) var<storage, read_write> gDebugDraw_buffer: DebugDraw_Buffer;

fn DebugDraw_getMouseHoverPos() -> vec2i {
    return gDebugDraw_buffer.mouse_hover_pos;
}

fn DebugDraw_getMousePressPos() -> vec2i {
    return gDebugDraw_buffer.mouse_press_pos;
}

/**
 * Sets all future debug draws to use depth tinting (if v is true), or traditional depth testing (if v is false).
 */
fn DebugDraw_setDepthTint(v: bool) {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 2u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.setDepthTint};
    gDebugDraw_buffer.data[offs + 1u] = select(0u, 1u, v);
}

/**
 * Sets all future debug draws to use a world-space coordinate system.
 */
fn DebugDraw_setWorldSpace() {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 2u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.setSpace};
    gDebugDraw_buffer.data[offs + 1u] = ${DebugDrawFlags.WorldSpace};
}

/**
 * Sets all future debug draws to use a view-space coordinate system,
 * where +z points towards the camera.
 */
fn DebugDraw_setViewSpace() {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 2u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.setSpace};
    gDebugDraw_buffer.data[offs + 1u] = ${DebugDrawFlags.ViewSpace};
}

/**
 * Sets all future debug draws to use a screen-space coordinate system,
 * where all draws are at the near plane, and x and y range from -1,1 (top left) to 1,1 (bottom right).
 */
fn DebugDraw_setScreenSpace() {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 2u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.setSpace};
    gDebugDraw_buffer.data[offs + 1u] = ${DebugDrawFlags.ScreenSpace};
}

fn DebugDraw_drawLine(p0: vec3f, p1: vec3f, color0: vec4f) {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 11u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.drawLine};
    gDebugDraw_buffer.data[offs + 1u] = bitcast<u32>(p0.x);
    gDebugDraw_buffer.data[offs + 2u] = bitcast<u32>(p0.y);
    gDebugDraw_buffer.data[offs + 3u] = bitcast<u32>(p0.z);
    gDebugDraw_buffer.data[offs + 4u] = bitcast<u32>(p1.x);
    gDebugDraw_buffer.data[offs + 5u] = bitcast<u32>(p1.y);
    gDebugDraw_buffer.data[offs + 6u] = bitcast<u32>(p1.z);
    gDebugDraw_buffer.data[offs + 7u] = bitcast<u32>(color0.x);
    gDebugDraw_buffer.data[offs + 8u] = bitcast<u32>(color0.y);
    gDebugDraw_buffer.data[offs + 9u] = bitcast<u32>(color0.z);
    gDebugDraw_buffer.data[offs + 10u] = bitcast<u32>(color0.w);
}

fn DebugDraw_drawSphere(p0: vec3f, radius: f32, color0: vec4f) {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 9u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.drawSphere};
    gDebugDraw_buffer.data[offs + 1u] = bitcast<u32>(p0.x);
    gDebugDraw_buffer.data[offs + 2u] = bitcast<u32>(p0.y);
    gDebugDraw_buffer.data[offs + 3u] = bitcast<u32>(p0.z);
    gDebugDraw_buffer.data[offs + 4u] = bitcast<u32>(radius);
    gDebugDraw_buffer.data[offs + 5u] = bitcast<u32>(color0.x);
    gDebugDraw_buffer.data[offs + 6u] = bitcast<u32>(color0.y);
    gDebugDraw_buffer.data[offs + 7u] = bitcast<u32>(color0.z);
    gDebugDraw_buffer.data[offs + 8u] = bitcast<u32>(color0.w);
}

fn DebugDraw_drawLocator(p0: vec3f, mag: f32, color0: vec4f) {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 9u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.drawLocator};
    gDebugDraw_buffer.data[offs + 1u] = bitcast<u32>(p0.x);
    gDebugDraw_buffer.data[offs + 2u] = bitcast<u32>(p0.y);
    gDebugDraw_buffer.data[offs + 3u] = bitcast<u32>(p0.z);
    gDebugDraw_buffer.data[offs + 4u] = bitcast<u32>(mag);
    gDebugDraw_buffer.data[offs + 5u] = bitcast<u32>(color0.x);
    gDebugDraw_buffer.data[offs + 6u] = bitcast<u32>(color0.y);
    gDebugDraw_buffer.data[offs + 7u] = bitcast<u32>(color0.z);
    gDebugDraw_buffer.data[offs + 8u] = bitcast<u32>(color0.w);
}

fn DebugDraw_screenPrintFloat1(v: f32) {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 2u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.screenPrintFloat1};
    gDebugDraw_buffer.data[offs + 1u] = bitcast<u32>(v);
}

fn DebugDraw_screenPrintFloat2(v: vec2f) {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 3u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.screenPrintFloat2};
    gDebugDraw_buffer.data[offs + 1u] = bitcast<u32>(v.x);
    gDebugDraw_buffer.data[offs + 2u] = bitcast<u32>(v.y);
}

fn DebugDraw_screenPrintFloat3(v: vec3f) {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 4u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.screenPrintFloat3};
    gDebugDraw_buffer.data[offs + 1u] = bitcast<u32>(v.x);
    gDebugDraw_buffer.data[offs + 2u] = bitcast<u32>(v.y);
    gDebugDraw_buffer.data[offs + 3u] = bitcast<u32>(v.z);
}

fn DebugDraw_screenPrintFloat4(v: vec4f) {
    var offs = atomicAdd(&gDebugDraw_buffer.size, 5u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.screenPrintFloat4};
    gDebugDraw_buffer.data[offs + 1u] = bitcast<u32>(v.x);
    gDebugDraw_buffer.data[offs + 2u] = bitcast<u32>(v.y);
    gDebugDraw_buffer.data[offs + 3u] = bitcast<u32>(v.z);
    gDebugDraw_buffer.data[offs + 4u] = bitcast<u32>(v.w);
}
`;
