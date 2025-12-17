# [Sample] Phi4 128k Agentic Model in WinUI3 App

## Set Up

You will need to have Visual Studio installed with the latest workloads for WinAppSDK and WinUI 3 development. You can find instructions on how to set up your environment [here.](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/set-up-your-development-environment?tabs=cs-vs-community%2Ccpp-vs-community%2Cvs-2022-17-1-a%2Cvs-2022-17-1-b#install-visual-studio)

Clone the repository or download the files and save them all into sub folder. and open the solution in Visual Studio. Before you can get started exploring the sample, you will need to get the Phi4 model files required for the project.
The model I used here is one of the Phi4-Mini-Instruct-onnx family models which features an 128k token context window and is optimized. The specific flavor is great for small environments.

This example uses Semantic Kernel to demonstrate how to turn these ready made AI models into a powerful coding partner or targeted system tool. With the large context window this configuration is ideal for large tasks or complex tasks. Each turn in the conversation is fed back into model with each subsequent turn.

## Downloading Phi4

The model can be downloaded from the following link:
- https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx

Huggingface models are in repositories which you can clone to get the model files. Clone the Phi3 model repository and copy the required files to the phi3 folder in the project.

Phi-3-mini-4k-instruct-onnx has 3 different versions inside it's repo. We are using the DirectML versions in this project.
Copy the contents of the "directml/directml-int4-awq-block-128" folder to the phi3 folder in the solution.

You don't need to modify the *.csproj, as it is already including all the files in the phi3 folder to the output directory.

The final folder structure should look like this:

```
ChatAppGenAI
├── phi3
│   ├── added_tokens.json
│   ├── genai_config.json
│   ├── model.onnx
│   ├── model.onnx.data
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer.model
│   ├── tokenizer_config.json
```