// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class ShaderLib {
    private static Path root;

    private static synchronized Path extract() throws IOException {
        if (root != null) return root;
        Path dir = Files.createTempDirectory("lumvex-kajiya-shaders");
        try (InputStream idx = ShaderLib.class.getResourceAsStream("/kajiya/shaders.index")) {
            if (idx == null) throw new IOException("kajiya/shaders.index missing; run tools/compile_kajiya_shaders.sh");
            for (String line : new String(idx.readAllBytes()).split("\\R")) {
                if (line.isBlank()) continue;
                Path out = dir.resolve(line.trim());
                Files.createDirectories(out.getParent());
                try (InputStream in = ShaderLib.class.getResourceAsStream("/kajiya/shaders/" + line.trim())) {
                    Files.write(out, in.readAllBytes());
                }
            }
        }
        return root = dir;
    }

    public static byte[] compile(String shader) {
        try {
            Path dir = extract();
            Path out = Files.createTempFile("kj", ".spv");
            String profile;
            boolean lib = false;
            if (shader.endsWith(".rgen.hlsl") || shader.endsWith(".rchit.hlsl") || shader.endsWith(".rmiss.hlsl")) {
                profile = "lib_6_4";
                lib = true;
            } else if (shader.endsWith("_vs.hlsl")) profile = "vs_6_4";
            else if (shader.endsWith("_ps.hlsl")) profile = "ps_6_4";
            else profile = "cs_6_4";
            List<String> cmd = new ArrayList<>(List.of(dxc(), "-spirv", "-fspv-target-env=vulkan1.2", "-WX", "-Ges", "-HV", "2021", "-T", profile));
            if (!lib) cmd.addAll(List.of("-E", "main"));
            cmd.addAll(List.of("-Fo", out.toString(), dir.resolve(shader).toString()));
            Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            String log = new String(p.getInputStream().readAllBytes());
            if (p.waitFor() != 0) throw new IllegalStateException("dxc failed for " + shader + ":\n" + log);
            byte[] code = Files.readAllBytes(out);
            Files.deleteIfExists(out);
            return code;
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException("shader compilation failed: " + shader, e);
        }
    }

    private static String dxc() {
        String sdk = System.getenv("VULKAN_SDK");
        if (sdk != null) {
            for (String n : new String[]{"Bin/dxc.exe", "Bin/dxc", "bin/dxc"}) {
                Path p = Path.of(sdk, n);
                if (Files.isExecutable(p)) return p.toString();
            }
        }
        return "dxc";
    }
}
