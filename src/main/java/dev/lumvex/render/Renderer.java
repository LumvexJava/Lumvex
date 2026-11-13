package dev.lumvex.render;

//DEV

import dev.lumvex.kajiya.KjRenderer;
import dev.lumvex.kajiya.gfx.Img;
import dev.lumvex.kajiya.render.Fmt;
import dev.lumvex.kajiya.world.SceneAdapter;
import dev.lumvex.kajiya.world.View;
import dev.lumvex.render.scene.Camera;
import dev.lumvex.render.scene.Scene;
import dev.lumvex.vk.*;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK13.*;

public final class Renderer implements AutoCloseable {

    public enum RenderMode {
        HYBRID, PATH, RASTER;

        public static RenderMode parse(String name) {
            if (name == null) return HYBRID;
            return switch (name.trim().toLowerCase(java.util.Locale.ROOT)) {
                case "path", "pathtracing", "path_tracing" -> PATH;
                case "raster", "rasterization" -> RASTER;
                default -> HYBRID;
            };
        }
    }

    public static boolean compileSer = false;
    private float adaptScale = Float.parseFloat(System.getProperty("lumvex.startscale", "0.6"));
    private double emaMs;
    private boolean adaptOnly;
    private long lastFrameNs;
    private double worstWindow;
    private int windowFrames;
    private double fastEma; //DEV
    private int adaptCooldown = 90;
    private static final boolean GPU_TIMES = Boolean.getBoolean("lumvex.gputimes");
    private static final String AUTOSHOT = System.getProperty("lumvex.autoshot");
    private static final int AUTOSHOT_FRAME = Integer.getInteger("lumvex.autoshotFrame", 120);

    public int maxBounces = 8;
    public int samplesPerFrame = 1;
    public float exposure = 1.0f;
    public float flareStrength = 1.0f;
    public int debugMode = 0;
    public boolean vsync;
    public int sampleLimit = 100;
    public final double[] lastGpuMs = new double[4];
    public final double[] lastCpuMs = new double[6];
    public int swapchainRecreations;
    public int lastPresentResult;
    public final long[] lastStats = new long[32];

    private final RenderSurface window;
    private final VkContext ctx;
    private final Scene scene;
    private final KjRenderer kj;
    private final SceneAdapter adapter;
    private RenderMode renderMode = RenderMode.HYBRID;

    private Swapchain swapchain;
    private int fbW, fbH, rW, rH;
    private float renderScale = Float.parseFloat(System.getProperty("lumvex.renderscale", "0"));
    private static final double AUTO_PIXEL_BUDGET = 1920.0 * 1080.0;
    private boolean sizeChanged = true;
    private Img target, ui;
    private VkCommandBuffer cmd;
    private long imageAvailable, inFlight, queryPool;
    private boolean timestampsPending;
    private final VkCommandBuffer[] cmds = new VkCommandBuffer[2];
    private final long[] avails = new long[2], flights = new long[2], pools = new long[2];
    private final boolean[] pend = new boolean[2];
    private int slot;
    private double gpuTotalMs;
    private Path pendingScreenshot;
    private int frame;
    private Camera currentCamera;

    private volatile int[] uiPending;
    private volatile int uiPendingW, uiPendingH;
    private volatile boolean uiChanged;

    private int[] pickPending;
    private Consumer<float[]> pickCallback;

    public Renderer(RenderSurface window, Scene scene, boolean validation, boolean vsync) {
        this.window = window;
        this.scene = scene;
        this.vsync = vsync;
        ctx = new VkContext(window, validation || Boolean.getBoolean("lumvex.validation")); //DEV
        System.out.println("[lumvex] GPU: " + ctx.deviceName + (ctx.rayTracingSupported ? " | Kajiya pipeline (hardware ray tracing)"
                : " | no hardware ray tracing: Kajiya pipeline without ray traced lighting"));
        if (!ctx.rayTracingSupported) renderMode = RenderMode.RASTER;
        kj = new KjRenderer(ctx);
        adapter = new SceneAdapter(kj.g, kj.world, scene);
        try (MemoryStack stack = stackPush()) {
            for (int k = 0; k < 2; k++) {
                org.lwjgl.PointerBuffer pp = stack.mallocPointer(1);
                check(vkAllocateCommandBuffers(ctx.device, VkCommandBufferAllocateInfo.calloc(stack).sType$Default().commandPool(ctx.commandPool)
                        .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1), pp), "alloc cmd");
                cmds[k] = new VkCommandBuffer(pp.get(0), ctx.device);
                java.nio.LongBuffer lp = stack.mallocLong(1);
                check(vkCreateSemaphore(ctx.device, VkSemaphoreCreateInfo.calloc(stack).sType$Default(), null, lp), "semaphore");
                avails[k] = lp.get(0);
                check(vkCreateFence(ctx.device, VkFenceCreateInfo.calloc(stack).sType$Default().flags(VK_FENCE_CREATE_SIGNALED_BIT), null, lp), "fence");
                flights[k] = lp.get(0);
                check(vkCreateQueryPool(ctx.device, VkQueryPoolCreateInfo.calloc(stack).sType$Default().queryType(VK_QUERY_TYPE_TIMESTAMP).queryCount(2), null, lp), "query pool");
                pools[k] = lp.get(0);
            }
            selectSlot(0);
        }
        createSizeDependent(VK_NULL_HANDLE);
    }

    public String deviceName() { return ctx.deviceName; }

    public int width() { return fbW; }

    public int height() { return fbH; }

    public RenderMode renderMode() { return renderMode; }

    public void setRenderMode(RenderMode mode) { renderMode = ctx.rayTracingSupported ? mode : RenderMode.RASTER; }

    public void setOverscan(float fraction) { }

    public float overscan() { return 0f; }

    public void setDisocclusionFix(float strength) { }

    public void resetAccumulation() { }

    public boolean idle() { return false; }

    public boolean sampleLimitReached() { return false; }

    public int framesToLimit() { return 0; }

    public int accumulatedSamples() { return frame; }

    public double gpuTotalMs() { return gpuTotalMs; }

    public double gpuTraceMs() { return gpuTotalMs; }

    public double gpuDenoiseMs() { return 0; }

    public double gpuTrainMs() { return 0; }

    public int tlasBuilds() { return frame; }

    public void requestScreenshot(Path file) { pendingScreenshot = file; }

    public void setRenderScale(float scale) {
        float s = scale <= 0 ? 0 : Math.clamp(scale, 0.25f, 1f);
        if (s != renderScale) {
            renderScale = s;
            sizeChanged = true;
        }
    }

    public float renderScale() { return fbW == 0 ? 1 : (float) rW / fbW; }

    private float effectiveRenderScale(int width, int height) {
        if (RenderTuning.adaptive()) return adaptScale; //DEV
        if (renderScale > 0) return renderScale;
        return (float) Math.min(1, Math.sqrt(AUTO_PIXEL_BUDGET / ((double) width * height)));
    }

    public void setUiImage(int[] argbPre, int w, int h) {
        uiPending = argbPre;
        uiPendingW = w;
        uiPendingH = h;
        uiChanged = true;
    }

    public void requestPick(int x, int y, Consumer<float[]> callback) {
        pickPending = new int[]{x, y};
        pickCallback = callback;
    }

    private void createSizeDependent(long oldSwapchain) {
        int[] fb = window.framebufferSize();
        swapchain = window.headless() ? null : new Swapchain(ctx, fb[0], fb[1], vsync, oldSwapchain);
        fbW = swapchain != null ? swapchain.width : fb[0];
        fbH = swapchain != null ? swapchain.height : fb[1];
        float scale = effectiveRenderScale(fbW, fbH);
        rW = Math.max(1, Math.round(fbW * scale));
        rH = Math.max(1, Math.round(fbH * scale));
        if (rW != fbW) System.out.printf("[lumvex] rendering %d x %d for a %d x %d window%n", rW, rH, fbW, fbH);
        if (target != null) {
            vkDeviceWaitIdle(ctx.device);
            target.close();
            ui.close();
            kj.g.flushPools();
        }
        target = new Img(ctx, new Img.Desc(fbW, fbH, Fmt.RGBA8));
        ui = new Img(ctx, new Img.Desc(fbW, fbH, org.lwjgl.vulkan.VK13.VK_FORMAT_B8G8R8A8_UNORM)); //DEV: ARGB ints in memory are BGRA bytes
        uiChanged = uiPending != null;
        sizeChanged = false;
    }

    public void recreateSwapchain() {
        vkDeviceWaitIdle(ctx.device);
        long old = swapchain != null ? swapchain.handle : VK_NULL_HANDLE;
        Swapchain previous = swapchain;
        window.waitWhileEmpty();
        createSizeDependent(old);
        if (previous != null) previous.close();
        swapchainRecreations++;
    }

    private int seenEnvironment = -1;

    private void applySettings() {
        if (seenEnvironment != scene.environmentRevision) {
            seenEnvironment = scene.environmentRevision;
            kj.setEnvironment(scene.environment);
        }
        kj.evShift = (float) (Math.log(Math.max(1e-4f, exposure * scene.exposure)) / Math.log(2.0));
        kj.sunDirection.set(scene.sunDirection);
        float k = scene.sunIntensity / 3f;
        kj.sunColorMultiplier.set(scene.sunColor).mul(k);
        kj.dynamicExposure = RenderTuning.autoExposure; //DEV
        kj.useMotionBlur = RenderTuning.motionBlur;
        kj.evShift += RenderTuning.evShift;
        kj.debugMode(debugMode);
    }

    private final Buffer[] uiUploads = new Buffer[2];

    private void uploadUi(MemoryStack stack) {
        if (!uiChanged) return;
        uiChanged = false;
        int[] px = uiPending;
        int w = uiPendingW, h = uiPendingH;
        long size = (long) fbW * fbH * 4;
        Buffer up = uiUploads[slot];
        if (up == null || up.size < size) {
            if (up != null) {
                vkDeviceWaitIdle(ctx.device);
                up.close();
            }
            up = uiUploads[slot] = new Buffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.HOST);
        }
        ByteBuffer m = up.map();
        java.nio.IntBuffer ib = m.order(java.nio.ByteOrder.nativeOrder()).asIntBuffer();
        if (px == null) {
            org.lwjgl.system.MemoryUtil.memSet(m, 0);
        } else if (w == fbW && h == fbH) {
            ib.put(0, px, 0, w * h);
        } else {
            org.lwjgl.system.MemoryUtil.memSet(m, 0);
            int cw = Math.min(w, fbW), ch = Math.min(h, fbH);
            for (int y = 0; y < ch; y++) ib.put(y * fbW, px, y * w, cw);
        }
        VkBufferImageCopy.Buffer r = VkBufferImageCopy.calloc(1, stack);
        r.get(0).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
        r.get(0).imageExtent().set(fbW, fbH, 1);
        vkCmdCopyBufferToImage(cmd, up.handle, ui.image, VK_IMAGE_LAYOUT_GENERAL, r);
    }

    private void selectSlot(int k) {
        slot = k;
        cmd = cmds[k];
        imageAvailable = avails[k];
        inFlight = flights[k];
        queryPool = pools[k];
        timestampsPending = pend[k];
    }

    private void readTimestamps(MemoryStack stack) {
        if (!timestampsPending) return;
        java.nio.LongBuffer lb = stack.mallocLong(2);
        if (vkGetQueryPoolResults(ctx.device, queryPool, 0, 2, lb, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            gpuTotalMs = (lb.get(1) - lb.get(0)) * ctx.timestampPeriodNs / 1e6;
            if (GPU_TIMES && frame % 120 == 0) System.out.printf("[lumvex] frame %d: GPU %.2f ms%n", frame, gpuTotalMs); //DEV
            lastGpuMs[2] = gpuTotalMs;
            if (RenderTuning.adaptive() && gpuTotalMs > 0.1) {
                emaMs = emaMs == 0 ? gpuTotalMs : emaMs * 0.95 + gpuTotalMs * 0.05;
                fastEma = fastEma == 0 ? gpuTotalMs : fastEma * 0.8 + gpuTotalMs * 0.2; //DEV: reacts within a few frames when the view suddenly gets heavy (falling, looking down a canyon)
                if (--adaptCooldown <= 0) {
                    float next = adaptScale;
                    float fixed = RenderTuning.fixedScale;
                    boolean urgent = false;
                    if (fixed > 0) next = Math.max(0.25f, Math.min(1f, fixed));
                    else {
                        float target = RenderTuning.targetMs;
                        if (fastEma > target * 1.25) { next = Math.max(RenderTuning.minScale, adaptScale - 0.08f); urgent = true; }
                        else if (emaMs > target * 1.04) next = Math.max(RenderTuning.minScale, adaptScale - 0.05f);
                        else if (emaMs < target * 0.72) next = Math.min(RenderTuning.maxScale, adaptScale + 0.05f);
                    }
                    adaptCooldown = 30;
                    if (Math.abs(next - adaptScale) > 1e-3f) {
                        adaptScale = next;
                        emaMs = 0;
                        fastEma = 0;
                        adaptCooldown = fixed > 0 ? 30 : urgent ? 50 : 360;
                        adaptOnly = true;
                    }
                }
            }
        }
        timestampsPending = false;
    }

    public void render(Camera camera, boolean animated) {
        try (MemoryStack stack = stackPush()) {
            selectSlot(frame & 1);
            long s0 = System.nanoTime();
            vkWaitForFences(ctx.device, inFlight, true, Long.MAX_VALUE);
            lastCpuMs[0] = (System.nanoTime() - s0) / 1e6;
            if (adaptOnly && !sizeChanged) {
                adaptOnly = false;
                rW = Math.max(1, Math.round(fbW * adaptScale));
                rH = Math.max(1, Math.round(fbH * adaptScale));
                kj.g.flushPools();
                System.out.printf("[lumvex] adaptive resolution: %d x %d (%.0f%%)%n", rW, rH, adaptScale * 100);
            }
            if (sizeChanged) {
                recreateSwapchain();
                return;
            }
            IntBuffer ip = stack.mallocInt(1);
            int acquire = VK_SUCCESS;
            if (swapchain != null) acquire = vkAcquireNextImageKHR(ctx.device, swapchain.handle, Long.MAX_VALUE, imageAvailable, VK_NULL_HANDLE, ip);
            else ip.put(0, 0);
            if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchain();
                return;
            }
            if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) check(acquire, "vkAcquireNextImageKHR");
            int imageIndex = ip.get(0);
            vkResetFences(ctx.device, inFlight);
            readTimestamps(stack);
            currentCamera = camera;
            long nowNs = System.nanoTime();
            if (lastFrameNs != 0) {
                double ms = (nowNs - lastFrameNs) / 1e6;
                RenderTuning.frameMs = RenderTuning.frameMs * 0.9 + ms * 0.1;
                RenderTuning.fps = 1000.0 / Math.max(0.01, RenderTuning.frameMs);
                worstWindow = Math.max(worstWindow, ms);
                if (++windowFrames >= 120) {
                    RenderTuning.worstMs = worstWindow;
                    worstWindow = 0;
                    windowFrames = 0;
                }
            }
            lastFrameNs = nowNs;
            RenderTuning.gpuMs = gpuTotalMs;
            RenderTuning.renderW = rW;
            RenderTuning.renderH = rH;
            RenderTuning.outW = fbW;
            RenderTuning.outH = fbH;
            adapter.sync();
            applySettings();
            readPick(camera);

            View view = View.of(camera.view(), camera.fovYDegrees, (float) fbW / fbH, 0.01f);
            vkResetCommandBuffer(cmd, 0);
            check(vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack).sType$Default().flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)), "begin cmd");
            vkCmdResetQueryPool(cmd, queryPool, 0, 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
            uploadUi(stack);
            long cpu0 = System.nanoTime();
            Img main = kj.record(cmd, slot, view, rW, rH, fbW, fbH, 1f / 60f);
            kj.blit(main, ui, target);
            if (pendingScreenshot == null && RenderTuning.screenshotRequest != null) { pendingScreenshot = RenderTuning.screenshotRequest; RenderTuning.screenshotRequest = null; } //DEV
            boolean screenshot = pendingScreenshot != null;
            presentCopy(stack, imageIndex);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
            check(vkEndCommandBuffer(cmd), "end cmd");

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(stack.pointers(cmd));
            if (swapchain != null) {
                submit.waitSemaphoreCount(1).pWaitSemaphores(stack.longs(imageAvailable)).pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_TRANSFER_BIT))
                        .pSignalSemaphores(stack.longs(swapchain.renderFinished[imageIndex]));
            }
            check(vkQueueSubmit(ctx.queue, submit, inFlight), "vkQueueSubmit");
            timestampsPending = ctx.timestampsSupported;
            pend[slot] = timestampsPending;
            RenderTuning.cpuMs = RenderTuning.cpuMs * 0.9 + (System.nanoTime() - cpu0) / 1e6 * 0.1;

            int pr = VK_SUCCESS;
            if (swapchain != null) {
                VkPresentInfoKHR present = VkPresentInfoKHR.calloc(stack).sType$Default().pWaitSemaphores(stack.longs(swapchain.renderFinished[imageIndex]))
                        .swapchainCount(1).pSwapchains(stack.longs(swapchain.handle)).pImageIndices(stack.ints(imageIndex));
                pr = vkQueuePresentKHR(ctx.queue, present);
                if (pr == VK_SUCCESS) window.framePresented();
            }
            if (pr != VK_SUCCESS) lastPresentResult = pr;
            frame++;
            if (AUTOSHOT != null && frame == AUTOSHOT_FRAME && pendingScreenshot == null) pendingScreenshot = Path.of(AUTOSHOT);
            if (screenshot) {
                Path file = pendingScreenshot;
                pendingScreenshot = null;
                vkQueueWaitIdle(ctx.queue);
                saveScreenshot(file);
                if (AUTOSHOT != null && Boolean.getBoolean("lumvex.autoshotExit") && file.equals(Path.of(AUTOSHOT))) System.exit(0);
            }
            if (swapchain != null && (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || window.consumeResized())) recreateSwapchain();
            else if (pr != VK_SUCCESS) check(pr, "vkQueuePresentKHR");
        }
    }

    private void presentCopy(MemoryStack stack, int imageIndex) {
        if (swapchain == null) return;
        long swapImage = swapchain.images[imageIndex];
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default().srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, mb, null, null);
        imageBarrier(stack, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkImageBlit.Buffer blit = VkImageBlit.calloc(1, stack);
        blit.get(0).srcSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
        blit.get(0).srcOffsets(1).set(fbW, fbH, 1);
        blit.get(0).dstSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
        blit.get(0).dstOffsets(1).set(fbW, fbH, 1);
        vkCmdBlitImage(cmd, target.image, VK_IMAGE_LAYOUT_GENERAL, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blit, VK_FILTER_NEAREST);
        imageBarrier(stack, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    private void imageBarrier(MemoryStack stack, long image, int oldLayout, int newLayout, int srcAccess, int dstAccess, int srcStage, int dstStage) {
        VkImageMemoryBarrier.Buffer b = VkImageMemoryBarrier.calloc(1, stack);
        b.get(0).sType$Default().srcAccessMask(srcAccess).dstAccessMask(dstAccess).oldLayout(oldLayout).newLayout(newLayout)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).image(image)
                .subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, null, null, b);
    }

    private void saveScreenshot(Path file) {
        float[] px = kj.g.readback(target);
        BufferedImage img = new BufferedImage(fbW, fbH, BufferedImage.TYPE_INT_RGB);
        int[] row = new int[fbW];
        for (int y = 0; y < fbH; y++) {
            for (int x = 0; x < fbW; x++) {
                int o = (y * fbW + x) * 4;
                row[x] = (Math.round(px[o] * 255) << 16) | (Math.round(px[o + 1] * 255) << 8) | Math.round(px[o + 2] * 255);
            }
            img.setRGB(0, y, fbW, 1, row, 0, fbW);
        }
        try {
            Path abs = file.toAbsolutePath();
            if (abs.getParent() != null) Files.createDirectories(abs.getParent());
            ImageIO.write(img, "png", abs.toFile());
            System.out.println("[lumvex] screenshot saved: " + abs);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Casts a ray through the pixel against the scene's triangles; reports {instance + 1, distance}, 0 for sky. */
    private void readPick(Camera camera) {
        if (pickPending == null || pickCallback == null) return;
        int px = pickPending[0], py = pickPending[1];
        Consumer<float[]> cb = pickCallback;
        pickPending = null;
        pickCallback = null;
        float nx = (px + 0.5f) / fbW * 2f - 1f, ny = 1f - (py + 0.5f) / fbH * 2f;
        Matrix4f inv = new Matrix4f();
        camera.viewProjection((float) fbW / fbH, inv).invert();
        Vector4f a = inv.transform(new Vector4f(nx, ny, 0f, 1f)), b = inv.transform(new Vector4f(nx, ny, 1f, 1f));
        Vector3f o = camera.position, d = new Vector3f(b.x / b.w - a.x / a.w, b.y / b.w - a.y / a.w, b.z / b.w - a.z / a.w).normalize();
        float best = Float.MAX_VALUE;
        int hit = -1;
        Matrix4f inverse = new Matrix4f();
        Vector3f lo = new Vector3f(), ld = new Vector3f();
        for (int i = 0; i < scene.instances.size(); i++) {
            Scene.Instance inst = scene.instances.get(i);
            inst.transform.invertAffine(inverse);
            inverse.transformPosition(o, lo);
            inverse.transformDirection(d, ld);
            float t = intersect(scene.meshes.get(inst.mesh), lo, ld);
            if (t < Float.MAX_VALUE) {
                Vector3f wp = new Vector3f(ld).mul(t).add(lo);
                inst.transform.transformPosition(wp);
                float dist = wp.distance(o);
                if (dist < best) {
                    best = dist;
                    hit = i;
                }
            }
        }
        cb.accept(new float[]{hit + 1, hit < 0 ? 0 : best});
    }

    private static float intersect(dev.lumvex.render.scene.Mesh m, Vector3f o, Vector3f d) {
        float[] bb = m.bounds();
        float t0 = 0, t1 = Float.MAX_VALUE;
        float[] oo = {o.x, o.y, o.z}, dd = {d.x, d.y, d.z};
        for (int k = 0; k < 3; k++) {
            float inv = 1f / (dd[k] == 0 ? 1e-30f : dd[k]);
            float a = (bb[k] - oo[k]) * inv, b2 = (bb[3 + k] - oo[k]) * inv;
            t0 = Math.max(t0, Math.min(a, b2));
            t1 = Math.min(t1, Math.max(a, b2));
        }
        if (t0 > t1) return Float.MAX_VALUE;
        float best = Float.MAX_VALUE;
        float[] v = m.vertices;
        int[] idx = m.indices;
        for (int i = 0; i + 2 < idx.length; i += 3) {
            int a = idx[i] * 8, b = idx[i + 1] * 8, c = idx[i + 2] * 8;
            float e1x = v[b] - v[a], e1y = v[b + 1] - v[a + 1], e1z = v[b + 2] - v[a + 2];
            float e2x = v[c] - v[a], e2y = v[c + 1] - v[a + 1], e2z = v[c + 2] - v[a + 2];
            float px = d.y * e2z - d.z * e2y, py = d.z * e2x - d.x * e2z, pz = d.x * e2y - d.y * e2x;
            float det = e1x * px + e1y * py + e1z * pz;
            if (Math.abs(det) < 1e-12f) continue;
            float id = 1f / det;
            float tx = o.x - v[a], ty = o.y - v[a + 1], tz = o.z - v[a + 2];
            float u = (tx * px + ty * py + tz * pz) * id;
            if (u < 0 || u > 1) continue;
            float qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
            float w = (d.x * qx + d.y * qy + d.z * qz) * id;
            if (w < 0 || u + w > 1) continue;
            float t = (e2x * qx + e2y * qy + e2z * qz) * id;
            if (t > 1e-4f && t < best) best = t;
        }
        return best;
    }

    @Override
    public void close() {
        vkDeviceWaitIdle(ctx.device);
        adapter.close();
        kj.close();
        target.close();
        ui.close();
        for (Buffer ub : uiUploads) if (ub != null) ub.close();
        if (swapchain != null) swapchain.close();
        for (int k = 0; k < 2; k++) {
            vkDestroyFence(ctx.device, flights[k], null);
            vkDestroySemaphore(ctx.device, avails[k], null);
            vkDestroyQueryPool(ctx.device, pools[k], null);
        }
        ctx.close();
    }
}
