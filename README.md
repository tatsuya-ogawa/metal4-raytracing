# MetalRaytracing

A raytracing sample app implementing Apple's latest Metal features, based on [JaapWijnen/metal-raytracing](https://github.com/JaapWijnen/metal-raytracing).

This project extends the original sample to experiment with **Metal4FX** capabilities (available on M3/A17 Pro and later) and adds interactive controls to explore the scene. **It also includes a fallback path for older devices, so it runs on any device that supports Metal4.**

![Demo](movies/demo.gif)

### Key Features

- **Metal 4 Support**:
  - Utilizes `MTL4CommandQueue`, `MTL4CommandBuffer`, and `MTL4ArgumentTable` for modern, low-overhead resource binding and submission.
  - Implements a hybrid rendering path that falls back to legacy Metal 3 APIs on unsupported devices.
  
- **Interactive Controls**:
  - **Orbit Camera**: Full 3D control to rotate, zoom, and pan around the scene.
  - **MetalFX Control**: Real-time toggling of MetalFX upscaling.
  
- **Advanced Rendering Pipeline**:
  - **MetalFX Integration**: Support for both Spatial and Temporal upscaling, toggleable at runtime.
  - **USDZ Support**: Seamless loading and raytracing of USDZ models.
  - **Skinning Animation**: GPU-accelerated skeletal animation for dynamic characters.
  - **PBR Materials**: Full Physically Based Rendering support (Base Color, Normal, Roughness, Metallic, AO, Emission).

### Requirements

- **Xcode 15.0+** (Swift 5.9+)
- **macOS 14.0+** or **iOS 17.0+**
- A device with Metal 3 support
  - *Metal 4 features require an Apple M3, A17 Pro, or newer chip, but the app will automatically fall back to Metal 3 on older hardware.*

### License

* This library is released under the MIT license. See [LICENSE](LICENSE) for details.
* The model file (`AssetResources/robot.usdz`) includes a separate license. Please refer to [AssetResources/LICENSE.txt](AssetResources/LICENSE.txt) for the model-specific terms and conditions.