import { mat4, ReadonlyMat4, vec3 } from "gl-matrix";
import { DebugDraw, DebugDrawFlags, Green, Red } from "./DebugDraw";

class Plane {
    private vertexBuffer: GPUBuffer;
    private indexBuffer: GPUBuffer;
    private texture: GPUTexture;
    private sampler: GPUSampler;
    private pso: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup;
    private worldFromModelMatrix = mat4.create();

    constructor(device: GPUDevice, colorTextureFormat: GPUTextureFormat) {
        this.vertexBuffer = device.createBuffer({ size: 5 * 4 * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: `Plane` });
        this.indexBuffer = device.createBuffer({ size: 6 * 2, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, label: `Plane` });

        const vertexData = new Float32Array([
            -1, 0, -1,  0, 0,
             1, 0, -1,  1, 0,
             1, 0,  1,  1, 1,
            -1, 0,  1,  0, 1,
        ]);
        device.queue.writeBuffer(this.vertexBuffer, 0, vertexData);

        const indexData = new Uint16Array([ 0, 1, 2, 0, 2, 3 ]);
        device.queue.writeBuffer(this.indexBuffer, 0, indexData);

        const textureData = new Uint8Array([
            0xCC, 0xCC, 0xCC, 0xCC,
            0x77, 0x77, 0x77, 0x77,
            0x77, 0x77, 0x77, 0x77,
            0xCC, 0xCC, 0xCC, 0xCC,
        ]);
        this.texture = device.createTexture({ 
            size: [2, 2],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            label: `Checkerboard`,
        });
        device.queue.writeTexture({ texture: this.texture }, textureData, { bytesPerRow: 8 }, [2, 2]);

        const shaderModule = this.createShaderModule(device);
        this.pso = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'main_vs',
                buffers: [{
                    attributes: [
                        { shaderLocation: 0, format: 'float32x3', offset: 0 },
                        { shaderLocation: 1, format: 'float32x2', offset: 3 * 4 },
                    ],
                    arrayStride: 5 * 4,
                }],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'main_ps',
                targets: [{ format: colorTextureFormat }],
            },
            depthStencil: {
                format: 'depth24plus',
                depthCompare: 'greater',
                depthWriteEnabled: true,
            },
            label: `Plane`,
        });

        this.sampler = device.createSampler({
            addressModeU: `repeat`,
            addressModeV: `repeat`,
            minFilter: `nearest`,
            magFilter: `nearest`,
            label: `Plane`,
        })

        this.uniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        this.bindGroup = device.createBindGroup({
            layout: this.pso.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.texture.createView() },
                { binding: 2, resource: this.sampler },
            ],
        });

        mat4.scale(this.worldFromModelMatrix, this.worldFromModelMatrix, [10, 10, 10]);
    }

    public draw(device: GPUDevice, clipFromWorldMatrix: ReadonlyMat4, pass: GPURenderPassEncoder): void {
        const clipFromModelMatrix = mat4.mul(mat4.create(), clipFromWorldMatrix, this.worldFromModelMatrix);
        device.queue.writeBuffer(this.uniformBuffer, 0, clipFromModelMatrix as Float32Array);

        pass.setPipeline(this.pso);
        pass.setBindGroup(0, this.bindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.setIndexBuffer(this.indexBuffer, 'uint16');
        pass.drawIndexed(6);
    }

    private createShaderModule(device: GPUDevice): GPUShaderModule {
        const code = `
struct ViewData {
    clip_from_model: mat4x4f,
};

@group(0) @binding(0) var<uniform> view_data: ViewData;
@group(0) @binding(1) var plane_texture: texture_2d<f32>;
@group(0) @binding(2) var point_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn main_vs(@location(0) position: vec3f, @location(1) uv: vec2f) -> VertexOutput {
    var out: VertexOutput;
    out.position = view_data.clip_from_model * vec4f(position, 1.0f);
    out.uv = uv * vec2f(10.0f);
    return out;
}

@fragment
fn main_ps(vertex: VertexOutput) -> @location(0) vec4f {
    return textureSample(plane_texture, point_sampler, vertex.uv);
}
`;
        return device.createShaderModule({ code, label: `Plane `});
    }
}

class Main {
    private canvas: HTMLCanvasElement;
    private device: GPUDevice;
    private ctx: GPUCanvasContext;
    private depthBuffer: GPUTexture;
    private plane: Plane;
    private clipFromViewMatrix = mat4.create();
    private viewFromWorldMatrix = mat4.create();
    private clipFromWorldMatrix = mat4.create();

    private debugDraw: DebugDraw;

    constructor() {
        this.canvas = document.createElement('canvas');
        this.canvas.width = 1920;
        this.canvas.height = 1080;
        document.body.appendChild(this.canvas);

        this.init();
    }

    private async init() {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter?.requestDevice();
        if (device === undefined)
            throw "whoops";

        const colorTextureFormat = navigator.gpu.getPreferredCanvasFormat();

        this.device = device;
        this.ctx = this.canvas.getContext('webgpu') as GPUCanvasContext;
        this.ctx.configure({ device, format: colorTextureFormat });

        this.depthBuffer = device.createTexture({ 
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        })

        this.plane = new Plane(device, colorTextureFormat);
        this.debugDraw = new DebugDraw(device, colorTextureFormat);

        requestAnimationFrame(this.update);
    }

    private updateCamera(): void {
        mat4.lookAt(this.viewFromWorldMatrix, [0, 10, 20], [0, 0, 0], [0, 1, 0]);

        mat4.perspectiveZO(this.clipFromViewMatrix, Math.PI / 3, this.canvas.width/this.canvas.height, 0.1, Infinity);

        // reverse depth
        mat4.mul(this.clipFromViewMatrix, mat4.fromValues(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, -1, 0,
            0, 0, 1, 1,
        ), this.clipFromViewMatrix);

        mat4.mul(this.clipFromWorldMatrix, this.clipFromViewMatrix, this.viewFromWorldMatrix);
    }

    private update = () => {
        this.updateCamera();

        const colorTexture = this.ctx.getCurrentTexture();
        const renderPass: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: colorTexture.createView(),
                clearValue: [0.5, 0.5, 0.75, 1.0],
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.depthBuffer.createView(),
                depthClearValue: 0.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            }
        };

        const cmd = this.device.createCommandEncoder();

        const pass = cmd.beginRenderPass(renderPass);
        this.plane.draw(this.device, this.clipFromWorldMatrix, pass);
        pass.end();

        // DebugDraw example.
        const time = window.performance.now();
        this.debugDraw.drawSphereLine(vec3.fromValues(Math.sin(time / 200) * 1.5, Math.sin(time / 300) * 0.2, Math.sin(time / 300 + 400)), 1, Red, 32, { flags: DebugDrawFlags.DepthTint });
        this.debugDraw.endFrame(cmd, this.clipFromViewMatrix, this.viewFromWorldMatrix, colorTexture.createView(), this.depthBuffer.createView());

        this.device.queue.submit([cmd.finish()]);

        requestAnimationFrame(this.update);
    };
}

function main() {
    const main = new Main();
    (window as any).main = main;
}

main();