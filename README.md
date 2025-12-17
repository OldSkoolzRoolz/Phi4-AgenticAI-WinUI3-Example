# Phi4 AgenticAI WinUI3 Example

## Overview
This repository contains a WinUI 3 desktop chat client that runs entirely on-device against a local Phi-4 ONNX model. The UI is implemented with the Windows App SDK and CommunityToolkit.Mvvm, while inference is handled through Microsoft.ML.OnnxRuntimeGenAI with the DirectML execution provider. Microsoft Semantic Kernel maintains a rolling chat history so every response is grounded in previous turns.

## Key Components
- `ChatAppGenAI/ChatAppGenAI.csproj` – WinUI 3 application targeting `net8.0-windows10.0.19041.0`, referencing Windows App SDK 1.8, CommunityToolkit.Mvvm, Semantic Kernel, and OnnxRuntime GenAI packages.
- `ChatAppGenAI/MainWindow.xaml(.cs)` – Declarative chat surface plus the `VM` view-model that wires UI commands to the Phi runner.
- `ChatAppGenAI/Phi4Runner.cs` – Loads the Phi-4 weights, initializes Semantic Kernel, and exposes `GenerateStreamingResponseAsync` for token-by-token streaming output.
- `ChatAppGenAI/Message.cs` – Observable message model that the XAML view binds to for chat rendering.

## Prerequisites
- Windows 10/11 build 19041 or later.
- Visual Studio 2022 17.10+ with the ".NET Desktop Development" and "Windows App SDK" workloads, or the .NET 8 SDK plus Windows App SDK runtime.
- DirectX 12–capable GPU/driver for DirectML acceleration.
- Local Phi-4 ONNX model files (see below).

## Model Setup
1. Download a Phi-4 ONNX checkpoint that exposes `model.onnx`, tokenizer assets, and `genai_config.json` (for example, from the official Microsoft Hugging Face releases).
2. Place the files under `ChatAppGenAI/Phi4/` so the layout matches:
   ```
   ChatAppGenAI/Phi4/
   ├── model.onnx
   ├── tokenizer.json
   ├── tokenizer_config.json
   ├── special_tokens_map.json
   ├── added_tokens.json
   └── ...
   ```
3. If you prefer another location, update the `pathHead` constant in `ChatAppGenAI/Phi4Runner.cs` to point at your directory.

## Build & Run
1. Restore packages: `dotnet restore ChatAppGenAI/ChatAppGenAI.csproj`.
2. Build for x64 (DirectML requires it): `dotnet build ChatAppGenAI/ChatAppGenAI.csproj -c Debug -p:Platform=x64`.
3. Launch from Visual Studio (F5) or run `ChatAppGenAI/bin/Debug/net8.0-windows10.0.19041.0/win-x64/ChatAppGenAI.exe`.
4. When the window opens, wait for the "Model loaded" debug output, type a prompt, and press Enter. Use the **Start New Conversation** button to clear history and reapply the system message.

## Configuration Tips
- Replace the hard-coded `pathHead` with `AppContext.BaseDirectory` or expose a settings UI if you need portable builds.
- `Phi4Runner.MaxContextTokens` governs how aggressively old turns are trimmed; tweak it for smaller GPUs.
- The MVVM layer (`VM`) centralizes dispatcher access via a helper, which keeps UI updates safe when streaming from background threads.

## Troubleshooting
- **Model not found** – ensure `model.onnx` exists at `Phi4Runner.ModelPath` and the process has read permission.
- **No GPU acceleration** – verify DirectML-compatible drivers are installed; otherwise switch to the CPU provider by changing the NuGet dependency.
- **Stalled UI** – confirm the dispatcher helper in `MainWindow.xaml.cs` is receiving a valid `DispatcherQueue` (already fixed by initializing it before subscribing to events).
- **Package restore errors** – run `nuget locals all -clear`, then `dotnet restore` to refresh the cache.

Customize the runner, prompts, or Semantic Kernel integration to adapt the sample into your own on-device AI assistant.