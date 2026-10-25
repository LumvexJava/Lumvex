package dev.lumvex.vk;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.EXTDebugUtils.*;
import static org.lwjgl.vulkan.EXTMemoryBudget.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
import static org.lwjgl.vulkan.EXTShaderAtomicFloat.VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRAccelerationStructure.*;
import static org.lwjgl.vulkan.KHRDeferredHostOperations.VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.*;
import static org.lwjgl.vulkan.KHRSurface.*;
import static org.lwjgl.vulkan.KHRSwapchain.VK_KHR_SWAPCHAIN_EXTENSION_NAME;
import static org.lwjgl.vulkan.VK13.*;

/**
 * Vulkan instance, surface and logical device with hardware ray tracing, plus the single queue (graphics, compute
 * and present) and a command pool.
 */
public final class VkContext implements AutoCloseable {

    private static final List<String> DEVICE_EXTENSIONS = List.of(
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            // visibility rays from compute shaders (lens flares)
            KHRRayQuery.VK_KHR_RAY_QUERY_EXTENSION_NAME
    );

    public final VkInstance instance;
    public final long surface;
    public final VkPhysicalDevice physicalDevice;
    public final VkDevice device;
    public final int queueFamily;
    public final VkQueue queue;
    public final long commandPool;
    public final String deviceName;

    public final VkPhysicalDeviceMemoryProperties memoryProperties;
    public final int shaderGroupHandleSize;
    public final int shaderGroupHandleAlignment;
    public final int shaderGroupBaseAlignment;
    public final int maxRayRecursionDepth;
    public final int minScratchOffsetAlignment;
    public final float timestampPeriodNs;
    public final boolean timestampsSupported;
    public final boolean memoryBudgetSupported;
    /** Float atomic add on storage buffers, needed by the radiance cache. */
    public final boolean atomicFloatAddSupported;
    /** Shader execution reordering ({@code VK_EXT_ray_tracing_invocation_reorder}). */
    public final boolean invocationReorderSupported;
    /** BC1-BC7 texture formats. */
    public final boolean textureCompressionBC;

    private final long debugMessenger;
    private final VkDebugUtilsMessengerCallbackEXT debugCallback;

    /** Validation is enabled only if requested and the Khronos layer is installed. */
    public VkContext(RenderSurface target, boolean requestValidation) {
        try (MemoryStack stack = stackPush()) {
            boolean validation = requestValidation && validationLayerAvailable();
            if (requestValidation && !validation) {
                System.err.println("[vk] VK_LAYER_KHRONOS_validation not found - is the Vulkan SDK installed? Continuing without validation.");
            }

            List<String> surfaceExt = target.instanceExtensions();
            PointerBuffer extensions = stack.mallocPointer(surfaceExt.size() + 1);
            for (String e : surfaceExt) extensions.put(stack.UTF8(e));
            if (validation) extensions.put(stack.UTF8(VK_EXT_DEBUG_UTILS_EXTENSION_NAME));
            extensions.flip();

            VkApplicationInfo appInfo = VkApplicationInfo.calloc(stack)
                    .sType$Default()
                    .pApplicationName(stack.UTF8("Lumvex"))
                    .applicationVersion(VK_MAKE_API_VERSION(0, 0, 1, 0))
                    .pEngineName(stack.UTF8("Lumvex"))
                    .engineVersion(VK_MAKE_API_VERSION(0, 0, 1, 0))
                    .apiVersion(VK_API_VERSION_1_3);

            VkInstanceCreateInfo instanceInfo = VkInstanceCreateInfo.calloc(stack)
                    .sType$Default()
                    .pApplicationInfo(appInfo)
                    .ppEnabledExtensionNames(extensions);
            if (validation) {
                instanceInfo.ppEnabledLayerNames(stack.pointers(stack.UTF8("VK_LAYER_KHRONOS_validation")));
            }

            PointerBuffer pp = stack.mallocPointer(1);
            check(vkCreateInstance(instanceInfo, null, pp), "vkCreateInstance");
            instance = new VkInstance(pp.get(0), instanceInfo);

            if (validation) {
                debugCallback = VkDebugUtilsMessengerCallbackEXT.create((severity, types, pData, pUser) -> {
                    VkDebugUtilsMessengerCallbackDataEXT data = VkDebugUtilsMessengerCallbackDataEXT.create(pData);
                    String level = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0 ? "ERROR" : "WARN";
                    System.err.println("[vk-validation " + level + "] " + data.pMessageString());
                    return VK_FALSE;
                });
                VkDebugUtilsMessengerCreateInfoEXT dbgInfo = VkDebugUtilsMessengerCreateInfoEXT.calloc(stack)
                        .sType$Default()
                        .messageSeverity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                        .messageType(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
                        .pfnUserCallback(debugCallback);
                LongBuffer lp = stack.mallocLong(1);
                check(vkCreateDebugUtilsMessengerEXT(instance, dbgInfo, null, lp), "vkCreateDebugUtilsMessengerEXT");
                debugMessenger = lp.get(0);
            } else {
                debugCallback = null;
                debugMessenger = VK_NULL_HANDLE;
            }

            LongBuffer lp = stack.mallocLong(1);
            surface = target.headless() ? VK_NULL_HANDLE : target.createSurface(instance);

            IntBuffer ip = stack.mallocInt(1);
            vkEnumeratePhysicalDevices(instance, ip, null);
            if (ip.get(0) == 0) throw new IllegalStateException("No Vulkan devices found");
            PointerBuffer devices = stack.mallocPointer(ip.get(0));
            vkEnumeratePhysicalDevices(instance, ip, devices);

            VkPhysicalDevice chosen = null;
            int chosenFamily = -1;
            int bestScore = -1;
            for (int i = 0; i < devices.capacity(); i++) {
                VkPhysicalDevice pd = new VkPhysicalDevice(devices.get(i), instance);
                int family = findQueueFamily(pd, stack);
                if (family < 0 || !supportsExtensions(pd)) continue;
                VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(stack);
                vkGetPhysicalDeviceProperties(pd, props);
                int score = props.deviceType() == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
                if (score > bestScore) {
                    bestScore = score;
                    chosen = pd;
                    chosenFamily = family;
                }
            }
            if (chosen == null) {
                throw new IllegalStateException("No GPU with VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure found");
            }
            physicalDevice = chosen;
            queueFamily = chosenFamily;

            VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps = VkPhysicalDeviceRayTracingPipelinePropertiesKHR.calloc(stack).sType$Default();
            VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps = VkPhysicalDeviceAccelerationStructurePropertiesKHR.calloc(stack).sType$Default();
            rtProps.pNext(asProps.address());
            VkPhysicalDeviceProperties2 props2 = VkPhysicalDeviceProperties2.calloc(stack).sType$Default().pNext(rtProps.address());
            vkGetPhysicalDeviceProperties2(physicalDevice, props2);
            deviceName = props2.properties().deviceNameString();
            shaderGroupHandleSize = rtProps.shaderGroupHandleSize();
            shaderGroupHandleAlignment = rtProps.shaderGroupHandleAlignment();
            shaderGroupBaseAlignment = rtProps.shaderGroupBaseAlignment();
            maxRayRecursionDepth = rtProps.maxRayRecursionDepth();
            minScratchOffsetAlignment = asProps.minAccelerationStructureScratchOffsetAlignment();
            timestampPeriodNs = props2.properties().limits().timestampPeriod();
            timestampsSupported = queueTimestampBits(physicalDevice, queueFamily, stack) > 0;
            Set<String> available = deviceExtensions(physicalDevice);
            memoryBudgetSupported = available.contains(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
            boolean atomicFloat = false;
            if (available.contains(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
                VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT.calloc(stack).sType$Default();
                vkGetPhysicalDeviceFeatures2(physicalDevice, VkPhysicalDeviceFeatures2.calloc(stack).sType$Default().pNext(af.address()));
                atomicFloat = af.shaderBufferFloat32AtomicAdd();
            }
            atomicFloatAddSupported = atomicFloat;
            boolean reorder = false;
            if (available.contains(EXTRayTracingInvocationReorder.VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)) {
                VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT rf = VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT.calloc(stack).sType$Default();
                vkGetPhysicalDeviceFeatures2(physicalDevice, VkPhysicalDeviceFeatures2.calloc(stack).sType$Default().pNext(rf.address()));
                reorder = rf.rayTracingInvocationReorder();
            }
            invocationReorderSupported = reorder;
            VkPhysicalDeviceFeatures supported = VkPhysicalDeviceFeatures.calloc(stack);
            vkGetPhysicalDeviceFeatures(physicalDevice, supported);
            textureCompressionBC = supported.textureCompressionBC();

            memoryProperties = VkPhysicalDeviceMemoryProperties.malloc();
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, memoryProperties);

            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFeatures = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT.calloc(stack)
                    .sType$Default()
                    .shaderBufferFloat32AtomicAdd(atomicFloatAddSupported);
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures = VkPhysicalDeviceRayTracingPipelineFeaturesKHR.calloc(stack)
                    .sType$Default()
                    .rayTracingPipeline(true);
            if (atomicFloatAddSupported) rtFeatures.pNext(atomicFeatures.address());
            // Shader execution reordering (Ada and newer) sorts incoherent bounce rays before shading.
            VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT reorderFeatures = VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT.calloc(stack)
                    .sType$Default()
                    .rayTracingInvocationReorder(true)
                    .pNext(rtFeatures.address());
            VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures = VkPhysicalDeviceRayQueryFeaturesKHR.calloc(stack)
                    .sType$Default()
                    .rayQuery(true)
                    .pNext(invocationReorderSupported ? reorderFeatures.address() : rtFeatures.address());
            VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = VkPhysicalDeviceAccelerationStructureFeaturesKHR.calloc(stack)
                    .sType$Default()
                    .accelerationStructure(true)
                    .pNext(rqFeatures.address());
            VkPhysicalDeviceVulkan13Features v13 = VkPhysicalDeviceVulkan13Features.calloc(stack)
                    .sType$Default()
                    .synchronization2(true)
                    .maintenance4(true)
                    .pNext(asFeatures.address());
            VkPhysicalDeviceVulkan12Features v12 = VkPhysicalDeviceVulkan12Features.calloc(stack)
                    .sType$Default()
                    .bufferDeviceAddress(true)
                    .scalarBlockLayout(true)
                    .descriptorIndexing(true)
                    .runtimeDescriptorArray(true)
                    .shaderSampledImageArrayNonUniformIndexing(true)
                    .pNext(v13.address());
            VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                    .sType$Default()
                    .pNext(v12.address());
            features2.features().shaderInt64(true).textureCompressionBC(textureCompressionBC);

            VkDeviceQueueCreateInfo.Buffer queueInfo = VkDeviceQueueCreateInfo.calloc(1, stack);
            queueInfo.get(0)
                    .sType$Default()
                    .queueFamilyIndex(queueFamily)
                    .pQueuePriorities(stack.floats(1.0f));

            PointerBuffer extNames = stack.mallocPointer(DEVICE_EXTENSIONS.size() + 3);
            for (String e : DEVICE_EXTENSIONS) extNames.put(stack.UTF8(e));
            if (memoryBudgetSupported) extNames.put(stack.UTF8(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME));
            if (atomicFloatAddSupported) extNames.put(stack.UTF8(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME));
            if (invocationReorderSupported) extNames.put(stack.UTF8(EXTRayTracingInvocationReorder.VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME));
            extNames.flip();

            VkDeviceCreateInfo deviceInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType$Default()
                    .pNext(features2.address())
                    .pQueueCreateInfos(queueInfo)
                    .ppEnabledExtensionNames(extNames);
            check(vkCreateDevice(physicalDevice, deviceInfo, null, pp), "vkCreateDevice");
            device = new VkDevice(pp.get(0), physicalDevice, deviceInfo, VK_API_VERSION_1_3);

            vkGetDeviceQueue(device, queueFamily, 0, pp);
            queue = new VkQueue(pp.get(0), device);

            VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                    .sType$Default()
                    .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                    .queueFamilyIndex(queueFamily);
            check(vkCreateCommandPool(device, poolInfo, null, lp), "vkCreateCommandPool");
            commandPool = lp.get(0);
        }
    }

    private static boolean validationLayerAvailable() {
        try (MemoryStack stack = stackPush()) {
            IntBuffer ip = stack.mallocInt(1);
            vkEnumerateInstanceLayerProperties(ip, null);
            VkLayerProperties.Buffer layers = VkLayerProperties.malloc(ip.get(0));
            try {
                vkEnumerateInstanceLayerProperties(ip, layers);
                for (VkLayerProperties l : layers) {
                    if (l.layerNameString().equals("VK_LAYER_KHRONOS_validation")) return true;
                }
                return false;
            } finally {
                layers.free();
            }
        }
    }

    /** First queue family with graphics, compute and, unless headless, presentation; -1 if there is none. */
    private int findQueueFamily(VkPhysicalDevice pd, MemoryStack stack) {
        IntBuffer ip = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, ip, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.malloc(ip.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, ip, families);
        IntBuffer present = stack.mallocInt(1);
        for (int i = 0; i < families.capacity(); i++) {
            int flags = families.get(i).queueFlags();
            if ((flags & VK_QUEUE_GRAPHICS_BIT) == 0 || (flags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            if (surface == VK_NULL_HANDLE) return i;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, present);
            if (present.get(0) == VK_TRUE) return i;
        }
        return -1;
    }

    private static boolean supportsExtensions(VkPhysicalDevice pd) {
        return deviceExtensions(pd).containsAll(DEVICE_EXTENSIONS);
    }

    private static Set<String> deviceExtensions(VkPhysicalDevice pd) {
        try (MemoryStack stack = stackPush()) {
            IntBuffer ip = stack.mallocInt(1);
            vkEnumerateDeviceExtensionProperties(pd, (String) null, ip, null);
            // Heap-allocated: NVIDIA drivers report more than 250 extensions, too many for the 64 KB stack.
            VkExtensionProperties.Buffer exts = VkExtensionProperties.malloc(ip.get(0));
            try {
                vkEnumerateDeviceExtensionProperties(pd, (String) null, ip, exts);
                Set<String> names = new HashSet<>();
                for (VkExtensionProperties e : exts) names.add(e.extensionNameString());
                return names;
            } finally {
                exts.free();
            }
        }
    }

    private static int queueTimestampBits(VkPhysicalDevice pd, int family, MemoryStack stack) {
        IntBuffer ip = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, ip, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.malloc(ip.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, ip, families);
        return families.get(family).timestampValidBits();
    }

    /** {usage, budget} of device-local memory in bytes, or null without {@code VK_EXT_memory_budget}. */
    public long[] deviceLocalMemoryUsage() {
        if (!memoryBudgetSupported) return null;
        try (MemoryStack stack = stackPush()) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = VkPhysicalDeviceMemoryBudgetPropertiesEXT.calloc(stack).sType$Default();
            VkPhysicalDeviceMemoryProperties2 props = VkPhysicalDeviceMemoryProperties2.calloc(stack)
                    .sType$Default()
                    .pNext(budget.address());
            vkGetPhysicalDeviceMemoryProperties2(physicalDevice, props);
            long usage = 0, total = 0;
            VkPhysicalDeviceMemoryProperties mp = props.memoryProperties();
            for (int h = 0; h < mp.memoryHeapCount(); h++) {
                if ((mp.memoryHeaps(h).flags() & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
                usage += budget.heapUsage(h);
                total += budget.heapBudget(h);
            }
            return new long[]{usage, total};
        }
    }

    public int findMemoryType(int typeBits, int required) {
        for (int i = 0; i < memoryProperties.memoryTypeCount(); i++) {
            if ((typeBits & (1 << i)) != 0
                    && (memoryProperties.memoryTypes(i).propertyFlags() & required) == required) {
                return i;
            }
        }
        throw new IllegalStateException("No suitable memory type (flags=" + required + ")");
    }

    /** Records, submits and waits for a one-off command buffer. */
    public void submitImmediate(Consumer<VkCommandBuffer> recorder) {
        try (MemoryStack stack = stackPush()) {
            VkCommandBufferAllocateInfo alloc = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType$Default()
                    .commandPool(commandPool)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(1);
            PointerBuffer pp = stack.mallocPointer(1);
            check(vkAllocateCommandBuffers(device, alloc, pp), "vkAllocateCommandBuffers");
            VkCommandBuffer cmd = new VkCommandBuffer(pp.get(0), device);

            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack)
                    .sType$Default()
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));
            recorder.accept(cmd);
            check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(pp);
            check(vkQueueSubmit(queue, submit, VK_NULL_HANDLE), "vkQueueSubmit");
            check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
            vkFreeCommandBuffers(device, commandPool, pp);
        }
    }

    @Override
    public void close() {
        vkDestroyCommandPool(device, commandPool, null);
        vkDestroyDevice(device, null);
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, null);
        if (debugMessenger != VK_NULL_HANDLE) vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
        if (debugCallback != null) debugCallback.free();
        vkDestroyInstance(instance, null);
        memoryProperties.free();
    }
}
