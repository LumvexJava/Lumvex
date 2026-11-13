# Third-Party Notices

Lumvex is licensed under the [MIT License](LICENSE). It contains code and data from
the third-party projects listed below. Their license texts are in the [`licenses/`](licenses)
folder and are also packaged into the jar under `META-INF/licenses/`.

---

## kajiya

- **Project:** kajiya – real-time global illumination renderer (Rust/Vulkan)
- **Source:** https://github.com/EmbarkStudios/kajiya
- **Copyright:** Copyright (c) 2019 Embark Studios
- **License:** MIT OR Apache-2.0 (dual-licensed). See [`licenses/kajiya/LICENSE-MIT`](licenses/kajiya/LICENSE-MIT)
  and [`licenses/kajiya/LICENSE-APACHE`](licenses/kajiya/LICENSE-APACHE).

Lumvex's render pipeline is a Java port of kajiya's renderer. Used from kajiya:

| In Lumvex | Origin in kajiya | Kind |
|---|---|---|
| `src/main/java/dev/lumvex/kajiya/**` | `crates/lib/kajiya`, `kajiya-backend`, `kajiya-rg` (Rust) | Rewritten in Java (port) |
| `src/main/resources/kajiya/shaders/**` | `assets/shaders/**` | HLSL shaders, copied, some adapted |
| `src/main/resources/kajiya/spv/**` | compiled from the shaders above | SPIR-V build output |
| `src/main/resources/kajiya/data/*.bin` | `assets/` / blue-noise sampler tables | Data tables |
| `tools/compile_kajiya_shaders.sh`, `tools/kajiya_shaders.txt` | `assets/shaders` layout | Build helper |

Changes made for Lumvex: rewritten from Rust to Java, adapted to Lumvex's Vulkan backend
(LWJGL), scene system and asset loading. Some features were removed (see README).

---

## Components bundled within kajiya's shaders and assets

### FidelityFX Shadow Denoiser (AMD)
- **Files:** `src/main/resources/kajiya/shaders/shadow_denoise/ffx/*`
- **Copyright:** Copyright (c) 2021 Advanced Micro Devices, Inc.
- **License:** MIT, see [`licenses/amd-ffx-shadows/LICENSE.txt`](licenses/amd-ffx-shadows/LICENSE.txt)
  (also kept next to the shaders).

### MinimalAtmosphere (Felix Westin)
- **Files:** `src/main/resources/kajiya/shaders/inc/atmosphere_felix.hlsl`
- **Source:** https://github.com/Fewes/MinimalAtmosphere
- **Copyright:** Copyright (c) 2021 Felix Westin
- **License:** MIT (the full license text is in the file header).

### Blue noise textures (Christoph Peters)
- **Files:** `src/main/resources/kajiya/images/LDR_RGBA_0.png`
- **Source:** http://momentsingraphics.de/BlueNoise.html
- **License:** CC0 1.0 (public domain dedication), see [`licenses/bluenoise/LICENSE.txt`](licenses/bluenoise/LICENSE.txt).

### Blue-noise sampler tables (`blue-noise-sampler` crate)
- **Files:** `src/main/resources/kajiya/data/rtr_*.bin` (Sobol, ranking and scrambling tiles, `spp64`)
- **Source:** https://github.com/Jasper-Bekkers/blue-noise-sampler, a dependency of kajiya. Based on
  "A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space"
  (Heitz, Belcour, Ostromoukhov, Coeurjolly, Iehl – 2019).
- **License:** see the crate's repository.

---

Many thanks to Embark Studios and all kajiya contributors for making kajiya open source.
