import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setGaussianMultiplier: (value: number) => void;
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================


  // Each intermediate splat occupies 48 bytes:
  // vec2<f32> center at offset 0, vec3<f32> conic at offset 16, f32 opacity at offset 28, and vec3<f32> color at offset 32, including padding required by WGSL storage-buffer alignment
  const splat_buffer = createBuffer(
    device,
    'projected Gaussian splats',
    pc.num_points * 48,
    GPUBufferUsage.STORAGE
  );

  const nulling_data = new Uint32Array([0]);





  // Gaussian rendering settings: gaussian_scaling and sh_deg
  // Allocate 16 bytes to satisfy uniform-buffer alignment requirements
  const render_settings_buffer = createBuffer(
    device,
    'gaussian render settings',
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1.0, pc.sh_deg, 0.0, 0.0])
  );





  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
        shDegree: pc.sh_deg,
      },
    },
  });




  // Bind packed Gaussian attributes, camera matrices, projected splats, and spherical harmonics coefficients
  // These bindings correspond to @group(0) @binding(0..3) in preprocess.wgsl
  const preprocess_bind_group = device.createBindGroup({
    label: 'gaussian preprocessing resources',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: pc.gaussian_3d_buffer },
      },
      {
        binding: 1,
        resource: { buffer: camera_buffer },
      },
      {
        binding: 2,
        resource: { buffer: splat_buffer },
      },
      {
        binding: 3,
        resource: { buffer: pc.sh_buffer },
      },
      {
        binding: 4,
        resource: { buffer: render_settings_buffer },
      },
    ],
  });




  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  // Four u32 values required by WebGPU's drawIndirect command:
  // vertexCount, instanceCount, firstVertex, firstInstance.
  //
  // Initially, draw six vertices per Gaussian (two triangles per quad).
  // Preprocessing will eventually determine the visible instance count.
  const indirect_draw_buffer = createBuffer(
    device,
    'gaussian indirect draw arguments',
    4 * Uint32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    new Uint32Array([6, pc.num_points, 0, 0])
  );

  const render_shader = device.createShaderModule({
    label: 'gaussian render shader',
    code: renderWGSL,
  });

  const render_pipeline = device.createRenderPipeline({
  label: 'gaussian render pipeline',
  layout: 'auto',
  vertex: {
    module: render_shader,
    entryPoint: 'vs_main',
  },
  fragment: {
    module: render_shader,
    entryPoint: 'fs_main',
    targets: [{
      format: presentation_format,
      blend: {
        color: {
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
          operation: 'add',
        },
        alpha: {
          srcFactor: 'one',
          dstFactor: 'one-minus-src-alpha',
          operation: 'add',
        },
      },
    }],
  },
  primitive: {
    topology: 'triangle-list',
  },
});



  // Bind the projected Gaussian centers to the vertex shader
  // This corresponds to @group(0) @binding(0) in gaussian.wgsl
  const render_bind_group = device.createBindGroup({
    label: 'gaussian rendering resources',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: splat_buffer },
      },
      {
        binding: 1,
        resource: {
          buffer: sorter.ping_pong[0].sort_indices_buffer,
        },
      },
    ],
  });
    

  // ===============================================
  //    Command Encoder Functions
  // ===============================================


  // Dispatch one preprocessing invocation per Gaussian
  // The workgroup size is 256, matching the specialization constant supplied when creating the preprocessing compute pipeline
  // The shader performs a bounds check for excess invocations



  const preprocess = (encoder: GPUCommandEncoder) => {
    // Reset the visible-Gaussian counter before each preprocessing dispatch
    encoder.clearBuffer(sorter.sort_info_buffer, 0, Uint32Array.BYTES_PER_ELEMENT);


    // Reset the padded sorting length (second u32 in SortInfos)
    encoder.clearBuffer(sorter.sort_info_buffer, Uint32Array.BYTES_PER_ELEMENT, Uint32Array.BYTES_PER_ELEMENT);

    // Reset the indirect sorting workgroup count before each frame
    encoder.clearBuffer(sorter.sort_dispatch_indirect_buffer, 0, Uint32Array.BYTES_PER_ELEMENT);
    
    const pass = encoder.beginComputePass({
      label: 'gaussian preprocessing'
    });



    pass.setPipeline(preprocess_pipeline);

    // Bind Gaussian inputs, camera matrices, and projected output centers
    pass.setBindGroup(0, preprocess_bind_group);



    // Bind the existing radix-sort resources required by the compute pipeline
    pass.setBindGroup(2, sort_bind_group);



    // Round up to cover all Gaussians, including the final partial workgroup
    const workgroup_count = Math.ceil(pc.num_points / C.histogram_wg_size);



    pass.dispatchWorkgroups(workgroup_count);
    pass.end();


    // Copy the visible-Gaussian count into the indirect draw instance count
    encoder.copyBufferToBuffer(
      sorter.sort_info_buffer,
      0,
      indirect_draw_buffer,
      Uint32Array.BYTES_PER_ELEMENT,
      Uint32Array.BYTES_PER_ELEMENT
    );



  };

  // Record a render pass that draws Gaussian quads indirectly
  // The instance count currently equals the full point count
  const render = (
    encoder: GPUCommandEncoder,
    texture_view: GPUTextureView
  ) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian quad render',
      colorAttachments: [{
        view: texture_view,
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });

    pass.setPipeline(render_pipeline);

    // Make the projected Gaussian centers available to the vertex shader
    pass.setBindGroup(0, render_bind_group);



    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // Project Gaussians and generate depth keys
      preprocess(encoder);


      // Sort visible Gaussians by depth using the provided GPU radix sorter
      sorter.sort(encoder);

      

      // Render Gaussian splats
      render(encoder, texture_view);
    },
    camera_buffer,

    setGaussianMultiplier: (value: number) => {
      device.queue.writeBuffer(
        render_settings_buffer,
        0,
        new Float32Array([value])
      );
    }
  };
}
