package dev.lumvex.render;

//DEV

public final class RenderTuning {
    public static volatile float targetMs = Float.parseFloat(System.getProperty("lumvex.targetms", "0"));
    public static volatile float fixedScale = 0f;
    public static volatile float minScale = 0.4f, maxScale = 1.0f;
    public static volatile boolean motionBlur = true;
    public static volatile boolean autoExposure = Boolean.parseBoolean(System.getProperty("lumvex.autoexposure", "false"));
    public static volatile float evShift = 0f;

    public static volatile double fps, gpuMs, frameMs, worstMs, cpuMs;
    public static volatile int renderW, renderH, outW, outH;
    //DEV: per frame raster statistics and a screenshot hook for game code (benchmark bots)
    public static volatile int drawnInstances, culledInstances;
    public static volatile long drawnTriangles;
    public static volatile java.nio.file.Path screenshotRequest;

    public static boolean adaptive() { return targetMs > 0 || fixedScale > 0; }

    private RenderTuning() { }
}
