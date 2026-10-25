package dev.lumvex.texture;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.lwjgl.system.MemoryUtil.memAlloc;

/**
 * Builds {@link TextureSet}s from PBR image maps, from glTF materials or texture folders such as Poly Haven
 * downloads. Roughness, metallic and occlusion are packed into ORM, and all maps are resampled to a common square
 * power-of-two size.
 */
public final class PbrImages {
    private PbrImages() {}

    public static BufferedImage read(Path file) throws IOException {
        BufferedImage img = ImageIO.read(file.toFile());
        if (img == null) throw new IOException("unsupported image: " + file);
        return img;
    }

    public static BufferedImage read(byte[] data) throws IOException {
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(data));
        if (img == null) throw new IOException("unsupported embedded image");
        return img;
    }

    /** The image's largest side rounded down to a power of two, clamped to [one tile, maxSize]. */
    public static int sizeFor(BufferedImage reference, int maxSize) {
        int s = Integer.highestOneBit(Math.max(reference.getWidth(), reference.getHeight()));
        return Math.clamp(s, TextureSet.TILE, Integer.highestOneBit(Math.max(maxSize, TextureSet.TILE)));
    }

    static int[] pixels(BufferedImage img, int size) {
        BufferedImage dst = img;
        if (img.getWidth() != size || img.getHeight() != size || img.getType() != BufferedImage.TYPE_INT_ARGB) {
            dst = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = dst.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.setComposite(AlphaComposite.Src);
            // Halve repeatedly; a single bilinear step of more than 2x aliases.
            BufferedImage cur = img;
            while (cur.getWidth() / 2 >= size * 2 || cur.getHeight() / 2 >= size * 2) {
                BufferedImage half = new BufferedImage(Math.max(cur.getWidth() / 2, 1), Math.max(cur.getHeight() / 2, 1), BufferedImage.TYPE_INT_ARGB);
                Graphics2D hg = half.createGraphics();
                hg.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                hg.setComposite(AlphaComposite.Src);
                hg.drawImage(cur, 0, 0, half.getWidth(), half.getHeight(), null);
                hg.dispose();
                cur = half;
            }
            g.drawImage(cur, 0, 0, size, size, null);
            g.dispose();
        }
        int[] px = new int[size * size];
        dst.getRGB(0, 0, size, size, px, 0, size);
        return px;
    }

    /**
     * Builds a texture set from separate maps, all resampled to {@code size}.
     *
     * @param albedo    sRGB albedo, alpha used for cutouts
     * @param normalGl  normal map in OpenGL convention, or null for flat
     * @param roughness image with roughness in channel {@code roughChannel} (0 = R, 2 = B), or null for 1
     * @param metallic  image with metallic in channel {@code metalChannel}, or null for 0
     * @param ao        ambient occlusion in the red channel, or null for 1
     */
    public static TextureSet build(String name, int size, BufferedImage albedo, BufferedImage normalGl,
                                   BufferedImage roughness, int roughChannel, BufferedImage metallic, int metalChannel,
                                   BufferedImage ao) {
        int n = size * size;
        ByteBuffer a = memAlloc(n * 4), nm = memAlloc(n * 4), orm = memAlloc(n * 4);
        int[] pa = pixels(albedo, size);
        for (int i = 0; i < n; i++) {
            int c = pa[i];
            a.put(i * 4, (byte) (c >> 16)).put(i * 4 + 1, (byte) (c >> 8)).put(i * 4 + 2, (byte) c).put(i * 4 + 3, (byte) (c >>> 24));
        }
        int[] pn = normalGl != null ? pixels(normalGl, size) : null;
        for (int i = 0; i < n; i++) {
            int c = pn != null ? pn[i] : 0xFF8080FF;
            // The engine's tangent frame has v pointing down, so the green channel of OpenGL maps is flipped.
            nm.put(i * 4, (byte) (c >> 16)).put(i * 4 + 1, (byte) (255 - ((c >> 8) & 0xFF))).put(i * 4 + 2, (byte) c).put(i * 4 + 3, (byte) 255);
        }
        int[] pr = roughness != null ? pixels(roughness, size) : null;
        int[] pm = metallic != null ? (metallic == roughness ? pr : pixels(metallic, size)) : null;
        int[] po = ao != null ? pixels(ao, size) : null;
        for (int i = 0; i < n; i++) {
            int r = pr != null ? (pr[i] >> (16 - 8 * roughChannel)) & 0xFF : 255;
            int m = pm != null ? (pm[i] >> (16 - 8 * metalChannel)) & 0xFF : 0;
            int o = po != null ? (po[i] >> 16) & 0xFF : 255;
            orm.put(i * 4, (byte) o).put(i * 4 + 1, (byte) r).put(i * 4 + 2, (byte) m).put(i * 4 + 3, (byte) 255);
        }
        return TextureSet.of(name, size, a, nm, orm);
    }

    // Role keywords in file names. Names are split at _ - . and spaces; two adjacent words may form one keyword.
    private static final String[] ALBEDO = {"diff", "diffuse", "albedo", "basecolor", "base_color", "color", "col", "basecolour", "colour"};
    private static final String[] NORMAL_DX = {"nordx", "normaldx", "nrmdx"};
    private static final String[] NORMAL = {"norgl", "normalgl", "nor", "normal", "nrm", "norm"};
    private static final String[] ROUGH = {"rough", "roughness", "rgh"};
    private static final String[] METAL = {"metal", "metallic", "metalness", "mtl"};
    private static final String[] AO = {"ao", "ambientocclusion", "occlusion", "occ"};
    private static final String[] ARM = {"arm"};
    private static final Set<String> IMAGE_TYPES = Set.of("png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif");

    /**
     * Loads a texture folder. Maps are recognized by file name: Poly Haven style ({@code _diff_}, {@code _nor_gl_},
     * {@code _arm_}, ...) as well as common names like basecolor, normal, roughness or metallic. A packed ARM map
     * (occlusion, roughness, metallic) is used when the separate maps are missing.
     */
    public static TextureSet loadFolder(Path folder, int maxSize) throws IOException {
        List<Path> images = new ArrayList<>();
        try (var s = Files.list(folder)) {
            // Empty files from broken downloads are skipped so that another map can take their place.
            s.filter(p -> IMAGE_TYPES.contains(ext(p)) && sizeOf(p) > 0).sorted().forEach(images::add);
        }
        Path diff = role(images, ALBEDO), norDx = role(images, NORMAL_DX), nor = norDx != null ? norDx : role(images, NORMAL);
        Path rough = role(images, ROUGH), metal = role(images, METAL), ao = role(images, AO), arm = role(images, ARM);
        if (diff == null) throw new IOException("no albedo / color image in " + folder + " (names like *_diff_*, *basecolor*, *albedo*)");
        long t0 = System.nanoTime();
        BufferedImage d = read(diff);
        int size = sizeFor(d, maxSize);
        BufferedImage n = nor != null ? optional(nor) : null;
        // DirectX to OpenGL convention
        if (n != null && norDx != null) n = flipGreen(n);
        BufferedImage r, m, o;
        int rc = 0, mc = 0;
        if (arm != null && rough == null) {
            BufferedImage packed = optional(arm);
            r = packed; m = packed; o = packed;
            rc = 1; mc = 2;
        } else {
            r = rough != null ? optional(rough) : null;
            m = metal != null ? optional(metal) : null;
            o = ao != null ? optional(ao) : null;
        }
        TextureSet set = build(folder.getFileName().toString(), size, d, n, r, rc, m, mc, o);
        System.out.printf("[textures] %-30s %dx%d (from %dx%d) in %.1f s%n", set.name, size, size, d.getWidth(), d.getHeight(),
                (System.nanoTime() - t0) / 1e9);
        return set;
    }

    /** Reads an optional map. An unreadable file is logged and skipped instead of failing the whole set. */
    private static BufferedImage optional(Path p) {
        try {
            return read(p);
        } catch (IOException ex) {
            System.out.println("[textures] warning: " + p.getFileName() + " is not readable - left out (" + ex.getMessage() + ")");
            return null;
        }
    }

    private static long sizeOf(Path p) {
        try {
            return Files.size(p);
        } catch (IOException e) {
            return 0;
        }
    }

    /** Whether the folder contains a recognizable albedo image. */
    public static boolean isTextureFolder(Path folder) throws IOException {
        List<Path> images = new ArrayList<>();
        try (var s = Files.list(folder)) {
            s.filter(p -> IMAGE_TYPES.contains(ext(p))).forEach(images::add);
        }
        return role(images, ALBEDO) != null;
    }

    private static String ext(Path p) {
        String n = p.getFileName().toString();
        int i = n.lastIndexOf('.');
        return i < 0 ? "" : n.substring(i + 1).toLowerCase(Locale.ROOT);
    }

    /** The first image whose name contains one of the keywords, or null. Previews and thumbnails are skipped. */
    private static Path role(List<Path> images, String[] tokens) {
        Set<String> want = Set.of(tokens);
        for (Path p : images) {
            String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
            if (n.contains("preview") || n.contains("thumb")) continue;
            n = n.substring(0, n.lastIndexOf('.'));
            // The set name may itself contain keywords ("rough_linen", "metal_plate"), so it is stripped first.
            String set = p.getParent() == null ? "" : p.getParent().getFileName().toString().toLowerCase(Locale.ROOT);
            if (!set.isEmpty() && n.startsWith(set + "_")) n = n.substring(set.length() + 1);
            String[] w = n.split("[_ .-]+");
            for (int i = 0; i < w.length; i++) {
                if (want.contains(w[i])) return p;
                if (i + 1 < w.length && want.contains(w[i] + w[i + 1])) return p;
            }
        }
        return null;
    }

    private static BufferedImage flipGreen(BufferedImage src) {
        BufferedImage out = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        int[] row = new int[src.getWidth()];
        for (int y = 0; y < src.getHeight(); y++) {
            src.getRGB(0, y, row.length, 1, row, 0, row.length);
            for (int x = 0; x < row.length; x++) row[x] = (row[x] & 0xFF00FF) | ((255 - ((row[x] >> 8) & 0xFF)) << 8);
            out.setRGB(0, y, row.length, 1, row, 0, row.length);
        }
        return out;
    }
}
