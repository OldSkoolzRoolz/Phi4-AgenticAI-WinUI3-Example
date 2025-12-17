using System;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace ChatAppGenAI;

/// <summary>
/// Handles loading the Phi-4 ONNX model and exposes helper methods for running streaming inference.
/// </summary>
public class Phi4Runner : IDisposable
{
    //Replace with your relative path to the model directory
    private const string pathHead = "f:\\repos\\LiveAI\\Phi4-AgenticAI-WinUI3-Example\\ChatAppGenAI";
    private const string ModelFileName = "model.onnx";
    private readonly string ModelDir = Path.Combine(pathHead, "Phi4");
    private string ModelPath => Path.Combine(ModelDir, ModelFileName);
    private const string SystemToken = "<|system|>";
    private const string UserToken = "<|user|>";
    private const string AssistantToken = "<|assistant|>";
    private const string EndToken = "<|end|>";
    private const int MaxContextTokens = 128_000;
    private const int AverageCharsPerToken = 4;
    private const string DefaultSystemMessage = "Starting new chat session.";

    private Kernel? kernel = null;
    private Model? model = null;
    private Tokenizer? tokenizer = null;
    private ChatHistory chatHistory = new();
    private readonly List<int> historyTokenCounts = new();
    private int historyTokenTotal;
    private readonly SemaphoreSlim initializationLock = new(1, 1);
    private Task? initializationTask;

    /// <summary>
    /// Raised when both the model and tokenizer are fully initialized.
    /// </summary>
    public event EventHandler? ModelLoaded = null;

    public Phi4Runner()
    {
        ResetConversationHistory(DefaultSystemMessage);
    }

    /// <summary>
    /// Indicates whether the model and tokenizer have been loaded and are ready for inference.
    /// </summary>
    [MemberNotNullWhen(true, nameof(model), nameof(tokenizer))]
    public bool IsReady => model != null && tokenizer != null;

    /// <summary>
    /// Releases model and tokenizer resources.
    /// </summary>
    public void Dispose()
    {
        model?.Dispose();
        tokenizer?.Dispose();
        initializationLock.Dispose();
    }

    /// <summary>
    /// Starts or restarts a conversation with an optional system message.
    /// </summary>
    public void StartNewConversation(string? systemMessage = null)
    {
        var message = string.IsNullOrWhiteSpace(systemMessage) ? DefaultSystemMessage : systemMessage!;
        ResetConversationHistory(message);
    }

    /// <summary>
    /// Initializes the Semantic Kernel, model, and tokenizer components if needed.
    /// </summary>
    public Task InitializeAsync(CancellationToken ct = default)
    {
        return initializationTask ??= InitializeInternalAsync(ct);
    }

    private async Task InitializeInternalAsync(CancellationToken ct)
    {
        if (IsReady)
        {
            return;
        }

        await initializationLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (IsReady)
            {
                return;
            }

            InitializeKernel();

            model = new Model(ModelDir);
            tokenizer = new Tokenizer(model);

            ModelLoaded?.Invoke(this, EventArgs.Empty);
        }
        finally
        {
            initializationLock.Release();
        }
    }

    private void InitializeKernel()
    {
        if (kernel != null)
        {
            return;
        }

        var builder = Kernel.CreateBuilder();
        builder.AddOnnxRuntimeGenAIChatCompletion("Phi4", ModelPath);
        kernel = builder.Build();
    }

    /// <summary>
    /// Generates a streaming response while keeping the Semantic Kernel chat history updated.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateStreamingResponseAsync(string userPrompt, [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (!IsReady)
        {
            throw new InvalidOperationException("Model is not ready");
        }

        if (string.IsNullOrWhiteSpace(userPrompt))
        {
            yield break;
        }

        AppendHistoryMessage(AuthorRole.User, userPrompt);

        var prompt = BuildPromptFromHistory();
        var responseBuilder = new StringBuilder();

        try
        {
            await foreach (var fragment in StreamFromPrompt(prompt, ct))
            {
                responseBuilder.Append(fragment);
                yield return fragment;
            }
        }
        finally
        {
            if (responseBuilder.Length > 0)
            {
                AppendHistoryMessage(AuthorRole.Assistant, responseBuilder.ToString().Trim());
            }
        }
    }

    private string BuildPromptFromHistory()
    {
        if (chatHistory.Count == 0)
        {
            throw new InvalidOperationException("Chat history is empty.");
        }

        var builder = new StringBuilder();
        var systemMessage = chatHistory.FirstOrDefault(m => m.Role == AuthorRole.System)?.Content ?? string.Empty;

        builder.Append(SystemToken).Append(systemMessage).Append(EndToken);

        foreach (var message in chatHistory.Where(m => m.Role != AuthorRole.System))
        {
            var roleToken = message.Role == AuthorRole.User ? UserToken : AssistantToken;
            builder.Append(roleToken)
                   .Append(message.Content ?? string.Empty)
                   .Append(EndToken);
        }

        builder.Append(AssistantToken);
        return builder.ToString();
    }

    private void AppendHistoryMessage(AuthorRole role, string? content)
    {
        var text = content?.Trim() ?? string.Empty;

        if (role == AuthorRole.User)
        {
            chatHistory.AddUserMessage(text);
        }
        else if (role == AuthorRole.Assistant)
        {
            chatHistory.AddAssistantMessage(text);
        }
        else
        {
            chatHistory.AddSystemMessage(text);
        }

        var tokenCount = EstimateTokenCount(text);
        historyTokenCounts.Add(tokenCount);
        historyTokenTotal += tokenCount;

        TrimHistoryToLimit();
    }

    private void TrimHistoryToLimit()
    {
        if (chatHistory.Count == 0)
        {
            return;
        }

        while (historyTokenTotal > MaxContextTokens && chatHistory.Count > 1)
        {
            historyTokenTotal -= historyTokenCounts[1];
            historyTokenCounts.RemoveAt(1);
            chatHistory.RemoveAt(1);
        }
    }

    private void ResetConversationHistory(string systemMessage)
    {
        chatHistory = new ChatHistory();
        historyTokenCounts.Clear();
        historyTokenTotal = 0;
        AppendHistoryMessage(AuthorRole.System, systemMessage);
    }

    private static int EstimateTokenCount(string text)
    {
        return Math.Max(1, text.Length / AverageCharsPerToken + 1);
    }

    /// <summary>
    /// Executes token-by-token streaming inference for a fully constructed Phi prompt.
    /// </summary>
    private async IAsyncEnumerable<string> StreamFromPrompt(string prompt, [EnumeratorCancellation] CancellationToken ct = default)
    {
        using var sequences = tokenizer!.Encode(prompt);

        var generatorParams = new GeneratorParams(model!);

        generatorParams.SetSearchOption("max_length", 1024);
        generatorParams.TryGraphCaptureWithMaxBatchSize(1);

        using var tokenizerStream = tokenizer.CreateStream();
        using var generator = new Generator(model!, generatorParams);

        generator.AppendTokenSequences(sequences);

        bool shouldStop = false;
        while (!generator.IsDone())
        {
            string part;
            string decodedPart = string.Empty;
            try
            {
                if (ct.IsCancellationRequested)
                {
                    break;
                }

                generator.GenerateNextToken();
                part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
                decodedPart = DecodeResponseFragment(part);

                if (part.Contains(EndToken, StringComparison.Ordinal)
                    || part.Contains(UserToken, StringComparison.Ordinal)
                    || part.Contains(SystemToken, StringComparison.Ordinal))
                {
                    shouldStop = true;
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
                break;
            }

            if (!string.IsNullOrEmpty(decodedPart))
            {
                yield return decodedPart;
            }

            if (shouldStop)
            {
                break;
            }
        }
    }

    private static string DecodeResponseFragment(string fragment)
    {
        if (string.IsNullOrEmpty(fragment))
        {
            return string.Empty;
        }

        fragment = fragment.Replace(SystemToken, string.Empty, StringComparison.Ordinal);
        fragment = fragment.Replace(UserToken, string.Empty, StringComparison.Ordinal);
        fragment = fragment.Replace(AssistantToken, string.Empty, StringComparison.Ordinal);

        var endIndex = fragment.IndexOf(EndToken, StringComparison.Ordinal);
        if (endIndex >= 0)
        {
            fragment = fragment[..endIndex];
        }

        return fragment;
    }
}

/// <summary>
/// Distinguishes user inputs from assistant responses within the conversation history.
/// </summary>
public enum PhiMessageType
{
    User,
    Assistant
}