package dev.lumvex.game;

/**
 * Game time, available to scripts as {@code time()}. It advances once per frame and stands still while the game is
 * paused.
 */
public final class Time extends SystemScript {
    float time, dt;
    long frame;

    Time() { }

    /** Seconds since the scene was loaded. */
    public float time() { return time; }

    /** Duration of the last frame in seconds. */
    public float dt() { return dt; }

    /** Number of frames since the scene was loaded. */
    public long frame() { return frame; }

    void advance(float d) {
        dt = d;
        time += d;
        frame++;
    }
}
