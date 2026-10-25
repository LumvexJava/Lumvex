package dev.lumvex.game;

import java.awt.event.KeyEvent;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Keyboard and mouse state, available to scripts as {@code input()}.
 *
 * <p>Key names are {@code A}-{@code Z}, {@code 0}-{@code 9}, {@code F1}-{@code F12}, {@code SPACE},
 * {@code ENTER}, {@code ESCAPE}, {@code TAB}, {@code BACKSPACE}, {@code SHIFT}, {@code CTRL}, {@code ALT},
 * {@code UP}, {@code DOWN}, {@code LEFT} and {@code RIGHT}; mouse buttons are {@code LEFT}, {@code RIGHT} and
 * {@code MIDDLE}. For event-style handling see {@link OnKey}, {@link OnMouse} and {@link OnClick}.
 *
 * <p>The host window feeds raw input from the UI thread through the public methods at the bottom; the script
 * thread consumes it once per frame.
 */
public final class Input extends SystemScript {
    record Raw(int type, String name, boolean down, float x, float y, String entity) {}

    static final int KEY = 0, BUTTON = 1, CLICK = 2;

    // Written by the UI thread.
    private final ConcurrentLinkedQueue<Raw> queue = new ConcurrentLinkedQueue<>();
    private float accDX, accDY;
    private volatile float mx = 0.5f, my = 0.5f;
    private volatile boolean lockRequested;
    /** Set while a modal canvas (a menu) is visible; the cursor must stay free then. */
    volatile boolean uiModal;

    // Per-frame state, script thread only.
    private final Set<String> down = new HashSet<>(), pressed = new HashSet<>(), released = new HashSet<>();
    private float dx, dy;

    public Input() { }

    /** Whether the key is held down, e.g. {@code isDown("W")}. */
    public boolean isDown(String key) { return down.contains(key.toUpperCase(Locale.ROOT)); }

    /** Whether the key went down during this frame. */
    public boolean pressed(String key) { return pressed.contains(key.toUpperCase(Locale.ROOT)); }

    /** Whether the key went up during this frame. */
    public boolean released(String key) { return released.contains(key.toUpperCase(Locale.ROOT)); }

    /** -1, 0 or 1 depending on which of the two keys is held, e.g. {@code axis("A", "D")}. */
    public float axis(String negative, String positive) { return (isDown(positive) ? 1 : 0) - (isDown(negative) ? 1 : 0); }

    /** Horizontal mouse movement during this frame in pixels. Also works while the mouse is captured. */
    public float mouseDX() { return dx; }

    /** Vertical mouse movement during this frame in pixels. */
    public float mouseDY() { return dy; }

    /** Cursor x in the view, 0 (left) to 1 (right). */
    public float mouseX() { return mx; }

    /** Cursor y in the view, 0 (top) to 1 (bottom). */
    public float mouseY() { return my; }

    public boolean mouseDown(String button) { return down.contains("MOUSE_" + button.toUpperCase(Locale.ROOT)); }

    /** Captures the mouse: the cursor is hidden and mouse movement is unbounded. Escape releases it. */
    public void lockMouse(boolean on) { lockRequested = on; }

    public boolean mouseLocked() { return lockRequested; }

    /** Reports a key by AWT key code. Keys without a script name are ignored. */
    public void key(int awtKeyCode, boolean isDown) {
        String n = keyName(awtKeyCode);
        if (n != null) queue.add(new Raw(KEY, n, isDown, 0, 0, null));
    }

    /** Reports a mouse button at a normalized view position. */
    public void button(String button, boolean isDown, float x, float y) {
        queue.add(new Raw(BUTTON, button, isDown, x, y, null));
    }

    /** Reports the entity under the cursor at a left click; null if the click hit nothing. */
    public void clicked(String entityId) { queue.add(new Raw(CLICK, null, true, 0, 0, entityId)); }

    /** Reports mouse movement. Deltas accumulate until the next frame. */
    public synchronized void mouseMoved(float deltaX, float deltaY, float x, float y) {
        accDX += deltaX;
        accDY += deltaY;
        mx = x;
        my = y;
    }

    /** Whether a menu is open, in which case the host should show the cursor and not capture it. */
    public boolean uiWantsMouse() { return uiModal; }

    /** Called by the host when the user released the captured mouse. */
    public void unlock() { lockRequested = false; }

    /** Drains the queued input into this frame's state and returns the events to dispatch. */
    List<Raw> beginFrame() {
        pressed.clear();
        released.clear();
        synchronized (this) {
            dx = accDX;
            dy = accDY;
            accDX = accDY = 0;
        }
        List<Raw> events = new ArrayList<>();
        Raw r;
        while ((r = queue.poll()) != null) {
            String n = r.type == BUTTON ? "MOUSE_" + r.name : r.name;
            if (r.type != CLICK) {
                if (r.down) {
                    if (!down.add(n)) continue;       // auto-repeat of a held key
                    pressed.add(n);
                } else {
                    if (!down.remove(n)) continue;
                    released.add(n);
                }
            }
            events.add(r);
        }
        return events;
    }

    void releaseAll() { down.clear(); }

    /** Script name of an AWT key code, or null if the key is not supported. Numpad digits map to digits. */
    public static String keyName(int c) {
        if (c >= KeyEvent.VK_A && c <= KeyEvent.VK_Z) return String.valueOf((char) ('A' + c - KeyEvent.VK_A));
        if (c >= KeyEvent.VK_0 && c <= KeyEvent.VK_9) return String.valueOf((char) ('0' + c - KeyEvent.VK_0));
        if (c >= KeyEvent.VK_NUMPAD0 && c <= KeyEvent.VK_NUMPAD9) return String.valueOf((char) ('0' + c - KeyEvent.VK_NUMPAD0));
        if (c >= KeyEvent.VK_F1 && c <= KeyEvent.VK_F12) return "F" + (c - KeyEvent.VK_F1 + 1);
        return switch (c) {
            case KeyEvent.VK_SPACE -> "SPACE";
            case KeyEvent.VK_ENTER -> "ENTER";
            case KeyEvent.VK_ESCAPE -> "ESCAPE";
            case KeyEvent.VK_TAB -> "TAB";
            case KeyEvent.VK_BACK_SPACE -> "BACKSPACE";
            case KeyEvent.VK_SHIFT -> "SHIFT";
            case KeyEvent.VK_CONTROL -> "CTRL";
            case KeyEvent.VK_ALT -> "ALT";
            case KeyEvent.VK_UP -> "UP";
            case KeyEvent.VK_DOWN -> "DOWN";
            case KeyEvent.VK_LEFT -> "LEFT";
            case KeyEvent.VK_RIGHT -> "RIGHT";
            default -> null;
        };
    }
}
