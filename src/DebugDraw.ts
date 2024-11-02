
import { mat4, ReadonlyMat4, ReadonlyVec3, vec2, vec3 } from "gl-matrix";

interface DebugDrawOptions {
    flags?: DebugDrawFlags;
};

export const enum DebugDrawFlags {
    // TODO(jstpierre): Implement these?
    WorldSpace = 0,
    ViewSpace = 1 << 0,
    ScreenSpace = 1 << 1,

    DepthTint = 1 << 2,

    Default = DepthTint,
};

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
    Opaque,
    Transparent,
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
        let flags = options.flags ?? DebugDrawFlags.Default;
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

    public static scratchVec3 = [vec3.create(), vec3.create(), vec3.create(), vec3.create()];

    constructor(private device: GPUDevice, private colorTextureFormat: GPUTextureFormat) {
        this.shaderModule = this.createShaderModule();
        this.uniformBuffer = device.createBuffer({ usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, size: 64+64+16, label: `DebugDraw Uniforms` });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
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

        this.debugDrawGPU = new DebugDrawGPU(this, this.device);
    }

    public getGPUBuffer(): GPUBuffer {
        return this.debugDrawGPU.gpuBuffer;
    }

    private createShaderModule(): GPUShaderModule {
        const code = `
struct ViewData {
    clip_from_view: mat4x4f,
    view_from_world: mat4x4f,
    misc: vec4f,
};

@id(0) override behavior_type: u32 = ${BehaviorType.Lines};

@group(0) @binding(0) var<uniform> view_data: ViewData;
@group(0) @binding(1) var depth_buffer: texture_depth_2d;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) @interpolate(flat) flags: u32,
};

@vertex
fn main_vs(@location(0) position: vec3f, @location(1) color: vec4f, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    // Flags are packed in the integer component of the alpha.
    var flags = u32(color.a);

    var out: VertexOutput;
    out.position = view_data.clip_from_view * view_data.view_from_world * vec4f(position, 1.0f);

    if (behavior_type == ${BehaviorType.Lines}) {
        // Hacky thick line support.
        if (instance_index >= 1) {
            var line_idx = instance_index - 1;
            var offs: vec2f;
            offs.x = select(1.0f, -1.0f, (line_idx & 1u) != 0u);
            offs.y = select(1.0f, -1.0f, (line_idx & 2u) != 0u);
            offs *= f32((line_idx / 4) + 1);

            out.position += vec4f(offs / vec2f(textureDimensions(depth_buffer)), 0.0f, 0.0f) * out.position.w;
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

    if (behavior_type != ${BehaviorType.Opaque}) {
        // Do manual depth testing so we can do a depth tint.
        if ((vertex.flags & ${DebugDrawFlags.DepthTint}) != 0) {
            var depth = textureLoad(depth_buffer, vec2u(vertex.position.xy), 0);

            if (depth > vertex.position.z) {
                color *= 0.2f;
            }
        }
    }

    return color;
}
`;
        return this.device.createShaderModule({ code, label: `DebugDraw` });
    }

    private createPipeline(behaviorType: BehaviorType): GPURenderPipeline {
        // Create the PSO based on our state.
        const desc: GPURenderPipelineDescriptor = {
            layout: this.pipelineLayout,
            primitive: {
                topology: behaviorType === BehaviorType.Lines ? 'line-list' : 'triangle-list',
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
            depthStencil: behaviorType === BehaviorType.Opaque ? {
                format: 'depth24plus',
                depthCompare: 'greater',
                depthWriteEnabled: true,
            } : undefined,
            label: `DebugDraw ${BehaviorType[behaviorType]}`,
        };

        return this.device.createRenderPipeline(desc);
    }

    private beginBatch(behaviorType: BehaviorType, vertexCount: number, indexCount: number): void {
        assert(this.currentPage === null);
        this.currentPage = this.findPage(behaviorType, vertexCount, indexCount);
    }

    public beginBatchLine(numSegments: number): void {
        this.beginBatch(BehaviorType.Lines, numSegments * 2, numSegments * 2);
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
        const page = this.findPage(BehaviorType.Lines, 2, 2);

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
        const page = this.findPage(BehaviorType.Lines, 6, 6);

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

    public drawDiscLineN(center: ReadonlyVec3, n: ReadonlyVec3, r: number, color: GfxColor, sides = 32, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        branchlessONB(DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], n);
        this.drawDiscSolidRU(center, DebugDraw.scratchVec3[0], DebugDraw.scratchVec3[1], r, color, sides, options);
    }

    public drawDiscLineRU(center: ReadonlyVec3, right: ReadonlyVec3, up: ReadonlyVec3, r: number, color: GfxColor, sides = 32, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(BehaviorType.Lines, sides, sides * 2);

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
        const behaviorType = color.a < 1.0 ? BehaviorType.Transparent : BehaviorType.Opaque;
        const page = this.findPage(behaviorType, sides + 1, sides * 3);

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
        const behaviorType = color.a < 1.0 ? BehaviorType.Transparent : BehaviorType.Opaque;
        const page = this.findPage(behaviorType, 3, 3);

        const baseVertex = page.getCurrentVertexID();
        page.vertexPCF(p0, color, options);
        page.vertexPCF(p1, color, options);
        page.vertexPCF(p2, color, options);

        for (let i = 0; i < 3; i++)
            page.index(baseVertex + i);
    }

    public drawRectLineP(p0: ReadonlyVec3, p1: ReadonlyVec3, p2: ReadonlyVec3, p3: ReadonlyVec3, color: GfxColor, options: DebugDrawOptions = { flags: DebugDrawFlags.Default }): void {
        const page = this.findPage(BehaviorType.Lines, 4, 8);

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
        const behaviorType = color.a < 1.0 ? BehaviorType.Transparent : BehaviorType.Opaque;
        const page = this.findPage(behaviorType, 4, 6);

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

    private drawPages(pass: GPURenderPassEncoder, behaviorType: BehaviorType): void {
        for (let i = 0; i < this.pages.length; i++) {
            const page = this.pages[i];
            if (page.behaviorType !== behaviorType)
                continue;

            const indexCount = page.getDrawCount();
            if (indexCount === 0)
                continue;

            pass.setPipeline(page.pso);
            pass.setVertexBuffer(0, page.vertexBuffer);
            pass.setIndexBuffer(page.indexBuffer, 'uint16');

            const instanceCount = behaviorType === BehaviorType.Lines ? 1 + (this.lineThickness - 1) * 4 : 1;
            pass.drawIndexed(indexCount, instanceCount);

            // Reset for next frame.
            page.endFrame();
        }
    }

    public beginFrame(mouseX: number, mouseY: number, mouseButtons: number): void {
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
        this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

        // First, check if we have any solid stuff (needs proper depth).
        if (behaviorTypes & (1 << BehaviorType.Opaque)) {
            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: this.dummyDepthBuffer.createView() },
                ],
                label: `DebugDraw`,
            });

            const renderPass = cmd.beginRenderPass({
                colorAttachments: [{ view: colorTextureView, loadOp: 'load', storeOp: 'store' }],
                depthStencilAttachment: { view: depthTextureView, depthLoadOp: 'load', depthStoreOp: 'store' },
            });

            renderPass.setBindGroup(0, bindGroup);
            this.drawPages(renderPass, BehaviorType.Opaque);
            renderPass.end();
        }

        if (behaviorTypes & ((1 << BehaviorType.Transparent) | (1 << BehaviorType.Lines))) {
            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: depthTextureView },
                ],
                label: `DebugDraw`,
            });

            const renderPass = cmd.beginRenderPass({
                colorAttachments: [{ view: colorTextureView, loadOp: 'load', storeOp: 'store' }],
            });

            renderPass.setBindGroup(0, bindGroup);
            this.drawPages(renderPass, BehaviorType.Transparent);
            this.drawPages(renderPass, BehaviorType.Lines);
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

        let offs = 8;
        for (let i = 0; i < count; i++) {
            const messageType: DebugDrawMessageType = view.getUint32(offs++ * 4, true);
            switch (messageType) {
            case DebugDrawMessageType.drawSphere:
                {
                    const p0 = vec3.fromValues(view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true));
                    const r = view.getFloat32(offs++ * 4, true);
                    const color = { r: view.getFloat32(offs++ * 4, true), g: view.getFloat32(offs++ * 4, true), b: view.getFloat32(offs++ * 4, true), a: view.getFloat32(offs++ * 4, true) };
                    this.debugDraw.drawSphereLine(p0, r, color);
                }
                break;
            case DebugDrawMessageType.drawLine:
                {
                    const p0 = vec3.fromValues(view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true));
                    const p1 = vec3.fromValues(view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true));
                    const color = { r: view.getFloat32(offs++ * 4, true), g: view.getFloat32(offs++ * 4, true), b: view.getFloat32(offs++ * 4, true), a: view.getFloat32(offs++ * 4, true) };
                    this.debugDraw.drawLine(p0, p1, color);
                }
                break;
            case DebugDrawMessageType.drawBasis:
                {
                    const p0 = vec3.fromValues(view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true), view.getFloat32(offs++ * 4, true));
                    this.debugDraw.drawBasis(mat4.fromTranslation(mat4.create(), p0));
                }
                break;
            }
        }
    }

    private updateFromFinishedFrames(): void {
        for (let i = 0; i < this.submittedFrames.length; i++) {
            const frame = this.submittedFrames[i];
            if (frame.mapState === 'unmapped') {
                frame.mapAsync(GPUMapMode.READ);
            }

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
        this.updateFromFinishedFrames();

        const readbackFrame = this.getFrame();
        cmd.copyBufferToBuffer(this.gpuBuffer, 0, readbackFrame, 0, this.size);

        this.submittedFrames.push(readbackFrame);
    }
}

const enum DebugDrawMessageType {
    drawLine,
    drawSphere,
    drawBasis,
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

fn DebugDraw_drawLine(p0: vec3f, p1: vec3f, color0: vec4f) {
    var offs = atomicAdd(&(gDebugDraw_buffer.size), 11);
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
    var offs = atomicAdd(&(gDebugDraw_buffer.size), 9u);
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

fn DebugDraw_drawBasis(p0: vec3f) {
    var offs = atomicAdd(&(gDebugDraw_buffer.size), 4u);
    gDebugDraw_buffer.data[offs + 0u] = ${DebugDrawMessageType.drawBasis};
    gDebugDraw_buffer.data[offs + 1u] = bitcast<u32>(p0.x);
    gDebugDraw_buffer.data[offs + 2u] = bitcast<u32>(p0.y);
    gDebugDraw_buffer.data[offs + 3u] = bitcast<u32>(p0.z);
}
`;
