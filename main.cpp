#include <iostream>
#include <webgpu/webgpu.h>

static void onAdapterRequest(
    WGPURequestAdapterStatus status,
    WGPUAdapter adapter,
    WGPUStringView message,
    void* userdata1,
    void* userdata2)
{
    if (status == WGPURequestAdapterStatus_Success) {
        *(WGPUAdapter*)userdata1 = adapter;
    } else {
        std::cout << "Adapter request failed." << std::endl;
    }
}

static void onDeviceRequest(
    WGPURequestDeviceStatus status,
    WGPUDevice device,
    WGPUStringView message,
    void* userdata1,
    void* userdata2)
{
    if (status == WGPURequestDeviceStatus_Success) {
        *(WGPUDevice*)userdata1 = device;
    } else {
        std::cout << "Device request failed." << std::endl;
    }
}

int main() {
    std::cout << "Initializing WebGPU..." << std::endl;

    WGPUInstanceDescriptor desc = {};
    WGPUInstance instance = wgpuCreateInstance(&desc);

    if (!instance) {
        std::cout << "Failed to create instance." << std::endl;
        return 1;
    }

    // ---- Request Adapter ----
    WGPUAdapter adapter = nullptr;

    WGPURequestAdapterOptions options = {};
    WGPURequestAdapterCallbackInfo callbackInfo = {};
    callbackInfo.callback = onAdapterRequest;
    callbackInfo.userdata1 = &adapter;
    callbackInfo.userdata2 = nullptr;

    wgpuInstanceRequestAdapter(instance, &options, callbackInfo);

    while (!adapter) {
        wgpuInstanceProcessEvents(instance);
    }

    std::cout << "Adapter acquired." << std::endl;

    // ---- Request Device ----
    WGPUDevice device = nullptr;

    WGPUDeviceDescriptor deviceDesc = {};
    WGPURequestDeviceCallbackInfo deviceCallbackInfo = {};
    deviceCallbackInfo.callback = onDeviceRequest;
    deviceCallbackInfo.userdata1 = &device;
    deviceCallbackInfo.userdata2 = nullptr;

    wgpuAdapterRequestDevice(adapter, &deviceDesc, deviceCallbackInfo);

    while (!device) {
        wgpuInstanceProcessEvents(instance);
    }

    std::cout << "Device created successfully!" << std::endl;

    WGPUQueue queue = wgpuDeviceGetQueue(device);
    std::cout << "Queue ready." << std::endl;

    return 0;
}