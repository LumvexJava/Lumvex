package dev.lumvex.ui;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import dev.lumvex.engine.SceneDoc;
import dev.lumvex.engine.SceneDoc.ScriptComponent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * A UI canvas as stored in {@code assets/ui/*.ui.json}.
 *
 * <p>Elements are laid out in a reference resolution and the whole canvas is scaled uniformly to the screen. The
 * element tree is flat in the file; {@link Element#parent} links children to their parent. Scripts can be attached
 * to the canvas and to individual elements.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public final class UiDoc {
    public String name = "Canvas";
    /** Reference resolution that positions and sizes are expressed in. */
    public int refWidth = 1920, refHeight = 1080;
    /** Whether the canvas is visible when the scene starts. */
    public boolean visible = true;
    /** A modal canvas (a menu) frees the mouse while visible and blocks clicks into the world. */
    public boolean modal;
    /** Draw order between canvases; higher is drawn on top. */
    public int order;
    /** Scripts on the canvas. They receive the clicks of all elements. */
    public List<ScriptComponent> scripts = new ArrayList<>();
    public List<Element> elements = new ArrayList<>();

    public static final String PANEL = "panel", TEXT = "text", BUTTON = "button", IMAGE = "image";

    /** A UI element: layout relative to its parent, appearance, and attached scripts. */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static final class Element {
        public String id = UUID.randomUUID().toString();
        public String name = "Element";
        public String type = PANEL;
        /** Id of the parent element; null for elements directly on the canvas. */
        public String parent;
        public String anchor = "center";
        /** Offset from the anchor in reference pixels; for {@code stretch}, the margins on each side. */
        public float x, y;
        public float width = 300, height = 80;
        public boolean visible = true;
        /** A disabled button is grayed out and ignores clicks. */
        public boolean enabled = true;

        public String text;
        public float fontSize = 32;
        public boolean bold;
        /** {@code left}, {@code center} or {@code right}. */
        public String align = "center";
        public float[] textColor = {1, 1, 1, 1};
        public float[] background = {0, 0, 0, 0};
        public float[] hoverColor;
        public float[] pressedColor;
        public float[] borderColor;
        public float borderWidth;
        public float cornerRadius;
        /** Image asset for image elements, or a background image for buttons. */
        public String image;
        public List<ScriptComponent> scripts = new ArrayList<>();

        public Element() { }

        public Element(String name, String type) {
            this.name = name;
            this.type = type;
        }

        /** Set for elements whose scripts were attached in code rather than listed in {@link #scripts}. */
        @JsonIgnore public boolean hasCodeScripts;

        /** Buttons and elements with scripts react to clicks. */
        @JsonIgnore public boolean clickable() { return BUTTON.equals(type) || !scripts.isEmpty() || hasCodeScripts; }
    }

    public Element find(String id) {
        if (id == null) return null;
        for (Element e : elements) if (e.id.equals(id)) return e;
        return null;
    }

    /** The first element with the given name, or null. */
    public Element byName(String name) {
        for (Element e : elements) if (e.name.equals(name)) return e;
        return null;
    }

    /** Direct children of an element; {@code null} returns the top-level elements. */
    public List<Element> children(String parentId) {
        List<Element> out = new ArrayList<>();
        for (Element e : elements) if (Objects.equals(e.parent, parentId)) out.add(e);
        return out;
    }

    /** Whether the element and all of its ancestors are visible. */
    public boolean shown(Element e) {
        Set<String> seen = new HashSet<>();
        for (Element p = e; p != null && seen.add(p.id); p = find(p.parent)) if (!p.visible) return false;
        return true;
    }

    public UiDoc copy() {
        return SceneDoc.JSON.convertValue(SceneDoc.JSON.valueToTree(this), UiDoc.class);
    }

    public static UiDoc load(Path file) throws IOException { return SceneDoc.JSON.readValue(file.toFile(), UiDoc.class); }

    public void save(Path file) throws IOException {
        Files.createDirectories(file.toAbsolutePath().getParent());
        SceneDoc.JSON.writeValue(file.toFile(), this);
    }

    /** Asset id for a canvas name, e.g. {@code ui/Main menu.ui.json}. Unsupported characters become {@code _}. */
    public static String assetId(String name) { return "ui/" + name.replaceAll("[^\\p{L}\\p{N}_\\- ]+", "_").trim() + ".ui.json"; }
}
