package dev.lumvex.ui;

import dev.lumvex.ui.UiDoc.Element;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.nio.file.Path;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Lays out and draws UI canvases with Java2D.
 *
 * <p>The canvas is scaled uniformly from its reference resolution to fit the target and centered. Each element is
 * positioned at its anchor point within the parent rectangle, offset by (x, y), and uses the matching point of its
 * own rectangle as pivot, so a bottom-right element grows up and to the left.
 */
public final class UiPainter {
    private UiPainter() { }

    /** Ids of the hovered and pressed elements, for button highlighting. */
    public record State(String hover, String pressed) {
        public static final State NONE = new State(null, null);
    }

    /** Element rectangles in target pixels for a canvas drawn at {@code w} x {@code h}, keyed by element id. */
    public static Map<String, Rectangle2D.Float> layout(UiDoc doc, float w, float h) {
        float s = scale(doc, w, h);
        Rectangle2D.Float root = new Rectangle2D.Float((w - doc.refWidth * s) / 2, (h - doc.refHeight * s) / 2, doc.refWidth * s, doc.refHeight * s);
        Map<String, Rectangle2D.Float> out = new LinkedHashMap<>();
        Deque<Element> todo = new ArrayDeque<>(doc.children(null));
        while (!todo.isEmpty()) {
            Element e = todo.poll();
            if (out.containsKey(e.id)) continue;
            Rectangle2D.Float p = e.parent == null ? root : out.get(e.parent);
            if (p == null) p = root;
            out.put(e.id, rect(e, p, s));
            todo.addAll(doc.children(e.id));
        }
        return out;
    }

    public static float scale(UiDoc doc, float w, float h) { return Math.min(w / doc.refWidth, h / doc.refHeight); }

    static Rectangle2D.Float rect(Element e, Rectangle2D.Float p, float s) {
        if ("stretch".equals(e.anchor)) {
            return new Rectangle2D.Float(p.x + e.x * s, p.y + e.y * s, Math.max(0, p.width - 2 * e.x * s), Math.max(0, p.height - 2 * e.y * s));
        }
        float[] a = anchor(e.anchor);
        float ew = e.width * s, eh = e.height * s;
        float ax = p.x + a[0] * p.width, ay = p.y + a[1] * p.height;
        return new Rectangle2D.Float(ax + e.x * s - a[0] * ew, ay + e.y * s - a[1] * eh, ew, eh);
    }

    /** Normalized position of an anchor within its parent; null means center. */
    public static float[] anchor(String a) {
        if (a == null) return new float[]{0.5f, 0.5f};
        float x = a.endsWith("left") ? 0 : a.endsWith("right") ? 1 : 0.5f;
        float y = a.startsWith("top") ? 0 : a.startsWith("bottom") ? 1 : 0.5f;
        if (a.equals("left")) { x = 0; y = 0.5f; }
        if (a.equals("right")) { x = 1; y = 0.5f; }
        return new float[]{x, y};
    }

    /**
     * Draws all visible canvases, lowest order first, into a premultiplied ARGB buffer. Returns null if no canvas is
     * visible.
     */
    public static int[] render(List<UiDoc> canvases, Map<UiDoc, State> states, int w, int h, Function<String, Path> assets) {
        List<UiDoc> shown = new ArrayList<>();
        for (UiDoc d : canvases) if (d.visible) shown.add(d);
        if (shown.isEmpty() || w <= 0 || h <= 0) return null;
        shown.sort(Comparator.comparingInt(d -> d.order));
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB_PRE);
        Graphics2D g = img.createGraphics();
        for (UiDoc d : shown) paint(g, d, states.getOrDefault(d, State.NONE), w, h, assets);
        g.dispose();
        return ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
    }

    public static void paint(Graphics2D g, UiDoc doc, State state, float w, float h, Function<String, Path> assets) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        float s = scale(doc, w, h);
        Map<String, Rectangle2D.Float> rects = layout(doc, w, h);
        for (Element e : drawOrder(doc)) {
            if (!doc.shown(e)) continue;
            Rectangle2D.Float r = rects.get(e.id);
            if (r == null) continue;
            draw(g, e, r, s, state, assets);
        }
    }

    /** Elements in paint order: depth first, parents before children, siblings in list order. */
    public static List<Element> drawOrder(UiDoc doc) {
        List<Element> out = new ArrayList<>();
        Deque<Element> stack = new ArrayDeque<>();
        List<Element> roots = doc.children(null);
        for (int i = roots.size() - 1; i >= 0; i--) stack.push(roots.get(i));
        Set<String> seen = new HashSet<>();
        while (!stack.isEmpty()) {
            Element e = stack.pop();
            if (!seen.add(e.id)) continue;
            out.add(e);
            List<Element> c = doc.children(e.id);
            for (int i = c.size() - 1; i >= 0; i--) stack.push(c.get(i));
        }
        return out;
    }

    private static void draw(Graphics2D g, Element e, Rectangle2D.Float r, float s, State st, Function<String, Path> assets) {
        boolean button = UiDoc.BUTTON.equals(e.type);
        float[] bg = e.background;
        if (button && !e.enabled) bg = mix(bg, new float[]{0.3f, 0.3f, 0.3f, bg[3]}, 0.6f);
        else if (button && e.id.equals(st.pressed())) bg = e.pressedColor != null ? e.pressedColor : mix(bg, new float[]{0, 0, 0, bg[3]}, 0.25f);
        else if (button && e.id.equals(st.hover())) bg = e.hoverColor != null ? e.hoverColor : mix(bg, new float[]{1, 1, 1, bg[3]}, 0.18f);
        float rad = e.cornerRadius * s;
        Shape shape = rad > 0 ? new RoundRectangle2D.Float(r.x, r.y, r.width, r.height, 2 * rad, 2 * rad) : r;
        if (bg != null && bg.length >= 4 && bg[3] > 0) {
            g.setColor(color(bg));
            g.fill(shape);
        }
        if (e.image != null && assets != null) {
            BufferedImage im = image(assets.apply(e.image));
            if (im != null) {
                Shape clip = g.getClip();
                g.clip(shape);
                g.drawImage(im, Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height), null);
                g.setClip(clip);
            }
        }
        if (e.borderWidth > 0 && e.borderColor != null) {
            g.setColor(color(e.borderColor));
            g.setStroke(new BasicStroke(e.borderWidth * s));
            g.draw(shape);
        }
        if (e.text != null && !e.text.isEmpty()) {
            float[] tc = button && !e.enabled ? mix(e.textColor, new float[]{0.6f, 0.6f, 0.6f, 1}, 0.6f) : e.textColor;
            g.setColor(color(tc));
            g.setFont(new Font(Font.SANS_SERIF, e.bold ? Font.BOLD : Font.PLAIN, 1).deriveFont(Math.max(1f, e.fontSize * s)));
            text(g, e, r, s);
        }
    }

    // Word-wraps the text to the element width and centers the block vertically.
    private static void text(Graphics2D g, Element e, Rectangle2D.Float r, float s) {
        FontMetrics fm = g.getFontMetrics();
        float pad = UiDoc.TEXT.equals(e.type) ? 0 : 8 * s;
        float maxW = Math.max(1, r.width - 2 * pad);
        List<String> lines = new ArrayList<>();
        for (String para : e.text.split("\n", -1)) {
            StringBuilder line = new StringBuilder();
            for (String word : para.split(" ")) {
                String t = line.isEmpty() ? word : line + " " + word;
                if (fm.stringWidth(t) > maxW && !line.isEmpty()) {
                    lines.add(line.toString());
                    line = new StringBuilder(word);
                } else {
                    line = new StringBuilder(t);
                }
            }
            lines.add(line.toString());
        }
        float lh = fm.getHeight();
        float total = lh * lines.size();
        float y = r.y + (r.height - total) / 2 + fm.getAscent();
        for (String l : lines) {
            float lw = fm.stringWidth(l);
            float x = "left".equals(e.align) ? r.x + pad : "right".equals(e.align) ? r.x + r.width - pad - lw : r.x + (r.width - lw) / 2;
            g.drawString(l, x, y);
            y += lh;
        }
    }

    /** The topmost visible, enabled and clickable element at a point, or null. */
    public static Element hit(UiDoc doc, float x, float y, float w, float h) {
        Map<String, Rectangle2D.Float> rects = layout(doc, w, h);
        List<Element> order = drawOrder(doc);
        for (int i = order.size() - 1; i >= 0; i--) {
            Element e = order.get(i);
            if (!e.clickable() || !e.enabled || !doc.shown(e)) continue;
            Rectangle2D.Float r = rects.get(e.id);
            if (r != null && r.contains(x, y)) return e;
        }
        return null;
    }

    // Decoded images; an empty Optional marks a file that could not be read.
    private static final Map<Path, Optional<BufferedImage>> IMAGES = new ConcurrentHashMap<>();

    private static BufferedImage image(Path p) {
        if (p == null) return null;
        return IMAGES.computeIfAbsent(p, k -> {
            try {
                return Optional.ofNullable(ImageIO.read(k.toFile()));
            } catch (Exception ex) {
                return Optional.empty();
            }
        }).orElse(null);
    }

    /** Converts RGB or RGBA in 0..1 to a color; null gives transparent. */
    public static Color color(float[] c) {
        if (c == null) return new Color(0, 0, 0, 0);
        return new Color(clamp(c[0]), clamp(c[1]), clamp(c[2]), c.length > 3 ? clamp(c[3]) : 1f);
    }

    private static float clamp(float v) { return Math.max(0, Math.min(1, v)); }

    // Blends the color of a towards b by t; the alpha of a is kept.
    private static float[] mix(float[] a, float[] b, float t) {
        if (a == null) return null;
        float[] o = new float[4];
        for (int i = 0; i < 4; i++) o[i] = (i < a.length ? a[i] : 1) * (1 - t) + b[i] * t;
        o[3] = a.length > 3 ? a[3] : 1;
        return o;
    }
}
