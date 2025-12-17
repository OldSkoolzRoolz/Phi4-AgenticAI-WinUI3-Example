using CommunityToolkit.Mvvm.ComponentModel;

using Microsoft.UI.Xaml;

using System;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace ChatAppGenAI;

public partial class Message : ObservableObject
{
    private string messageText;
    public string MessageText
    {
        get => messageText;
        set => SetProperty(ref messageText, value);
    }

    private DateTime msgDateTime;
    public DateTime MsgDateTime
    {
        get => msgDateTime;
        set => SetProperty(ref msgDateTime, value);
    }

    public PhiMessageType Type { get; set; }
    public HorizontalAlignment MsgAlignment => Type == PhiMessageType.User ? HorizontalAlignment.Right : HorizontalAlignment.Left;
    public Message(string text, DateTime dateTime, PhiMessageType type)
    {
        MessageText = text;
        MsgDateTime = dateTime;
        Type = type;
    }

    public override string ToString()
    {
        return MsgDateTime.ToString() + " " + MessageText;
    }
}
