package dev.lumvex.game;

/**
 * A linear RGB color with opacity. Components are in 0..1; emission strengths are given separately, see
 * {@link Material#withEmission(Color, float)}. For 0..255 values use {@link #rgb(int, int, int)}; surfaces clamp
 * their albedo to 1.
 *
 * @param r red
 * @param g green
 * @param b blue
 * @param a opacity, 1 is opaque
 */
public record Color(float r, float g, float b, float a) {
    public static final Color WHITE = new Color(1, 1, 1);
    public static final Color BLACK = new Color(0, 0, 0);
    public static final Color TRANSPARENT = new Color(0, 0, 0, 0);

    /** An opaque color. */
    public Color(float r, float g, float b) { this(r, g, b, 1); }

    /**
     * An opaque color from 8-bit sRGB values (0 to 255), as color pickers show them, converted to linear. Same as
     * {@link #hex(String)}.
     */
    public static Color rgb(int r, int g, int b) {
        return new Color(linear(r), linear(g), linear(b));
    }

    /**
     * An opaque color from a hex string such as {@code "#FFC857"}, converted from sRGB to linear. This matches
     * colors picked in an image editor.
     */
    public static Color hex(String hex) {
        String h = hex.startsWith("#") ? hex.substring(1) : hex;
        if (h.length() != 6) throw new IllegalArgumentException("expected #RRGGBB, got " + hex);
        int v = Integer.parseInt(h, 16);
        return rgb((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
    }

    /** The same color with another opacity. */
    public Color withAlpha(float alpha) { return new Color(r, g, b, alpha); }

    float[] rgb() { return new float[]{r, g, b}; }

    float[] rgba() { return new float[]{r, g, b, a}; }

    static Color of(float[] c) { return c.length > 3 ? new Color(c[0], c[1], c[2], c[3]) : new Color(c[0], c[1], c[2]); }

    private static float linear(int channel) {
        float c = Math.clamp(channel, 0, 255) / 255f;
        return c <= 0.04045f ? c / 12.92f : (float) Math.pow((c + 0.055f) / 1.055f, 2.4);
    }
}
