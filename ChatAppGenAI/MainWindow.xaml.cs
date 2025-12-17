using CommunityToolkit.Mvvm.ComponentModel;

using Microsoft.UI;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;

using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Threading.Tasks;

using Windows.UI;



namespace ChatAppGenAI;



public sealed partial class MainWindow : Window
{
    private VM VM;

    public MainWindow()
    {
        this.InitializeComponent();
        VM = new VM(DispatcherQueue);
    }

    private void StartNewConversation_Click(object sender, RoutedEventArgs e)
    {
        VM.StartNewConversation();
    }

    private void TextBox_KeyUp(object sender, KeyRoutedEventArgs e)
    {
        var textBox = sender as TextBox;
        if (e.Key == Windows.System.VirtualKey.Enter)
        {
            if (textBox.Text.Length > 0)
            {
                VM.AddMessage(textBox.Text);
                textBox.Text = string.Empty;
            }
        }
    }
    public static SolidColorBrush PhiMessageTypeToColor(PhiMessageType type)
    {
        return (type == PhiMessageType.User) ? new SolidColorBrush(Colors.White) : new SolidColorBrush(Color.FromArgb(255, 68, 228, 255));
    }

    public static SolidColorBrush PhiMessageTypeToForeground(PhiMessageType type)
    {
        return (type == PhiMessageType.User) ? new SolidColorBrush(Colors.Black) : new SolidColorBrush(Color.FromArgb(255, 80, 80, 80));
    }

    public static Visibility BoolToVisibleInversed(bool value)
    {
        return value ? Visibility.Collapsed : Visibility.Visible;
    }
}
public partial class VM: ObservableObject
{
    public ObservableCollection<Message> Messages = new();

    private bool acceptsMessages;
    public bool AcceptsMessages
    {
        get => acceptsMessages;
        set => SetProperty(ref acceptsMessages, value);
    }

    private Phi4Runner phi3 = new();
    private DispatcherQueue dispatcherQueue;

    public VM(DispatcherQueue dispatcherQueue)
    {
        this.dispatcherQueue = dispatcherQueue ?? throw new ArgumentNullException(nameof(dispatcherQueue));

        phi3.ModelLoaded += Phi3_ModelLoaded;
        phi3.StartNewConversation("Beginning new chat session.");
        _ = phi3.InitializeAsync();
    }

    private void Phi3_ModelLoaded(object sender, EventArgs e)
    {
        Debug.WriteLine("Model loaded");
        Dispatch(() => AcceptsMessages = true);
    }

    public void StartNewConversation()
    {
        Messages.Clear();
        phi3.StartNewConversation("Beginning new chat session.");
        AcceptsMessages = phi3.IsReady;
    }

    /// <summary>
    /// Method that is called to add a message from the user and get a response from the AI.
    /// </summary>
    /// <param name="text"></param>
    public void AddMessage(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        var sanitizedText = text.Trim();
        AcceptsMessages = false;
        Messages.Add(new Message(sanitizedText, DateTime.Now, PhiMessageType.User));

        Task.Run(async () =>
        {
            var responseMessage = new Message("...", DateTime.Now, PhiMessageType.Assistant);

            Dispatch(() =>
            {
                Messages.Add(responseMessage);
            });

            bool firstPart = true;

            try
            {
                await foreach (var messagePart in phi3.GenerateStreamingResponseAsync(sanitizedText))
                {
                    var part = messagePart;
                    Dispatch(() =>
                    {
                        if (firstPart)
                        {
                            responseMessage.MessageText = string.Empty;
                            firstPart = false;
                            part = messagePart.TrimStart();
                        }

                        responseMessage.MessageText += part;
                    });
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
            }
            finally
            {
                Dispatch(() =>
                {
                    AcceptsMessages = true;
                });
            }
        });
    }

    private void Dispatch(Action action)
    {
        if (dispatcherQueue == null)
        {
            action();
            return;
        }

        DispatcherQueueHandler handler = () => action();
        if (!dispatcherQueue.TryEnqueue(handler))
        {
            action();
        }
    }
}
