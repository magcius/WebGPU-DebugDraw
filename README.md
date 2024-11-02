
# GPU-driven DebugDraw in WebGPU

* A basic demonstration showing how to draw a lot of different shapes to help debug WebGPU: lines, axes, spheres, discs (solid and lines), rectangles (solid and lines).
* Supports depth tinting (shader-driven depth buffer testing to let you "see" the other side of your shapes through objects), and "thick" lines (through instancing).
* A basic monospace text printer as well for debugging.
* Shows a "GPU-driven" mode allowing you to draw shapes and print float vectors from shaders, using a storage buffer and appending draw commands. Helpful for debugging geometry from shaders.
