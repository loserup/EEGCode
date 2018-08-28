using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Runtime.InteropServices;
using System.Windows.Threading;
using System.Threading;

namespace SyncKeyWPF
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        [DllImport("user32.dll", EntryPoint = "SendMessage")]
        private static extern int SendMessage(IntPtr hwnd, int msg, int wParam, int lParam);

        [DllImport("User32.dll", EntryPoint = "FindWindow")]
        private static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll")]
        private static extern short GetAsyncKeyState(int vKey);

        [DllImport("user32.dll")]
        private static extern bool AttachThreadInput(IntPtr idAttach, IntPtr idAttachTo, bool fAttach);

        [DllImport("Kernel32.dll")]
        private static extern IntPtr GetCurrentThreadId();

        [DllImport("user32.dll")]
        private static extern IntPtr GetWindowThreadProcessId(IntPtr hWnd, IntPtr lpdwProcessId);

        [DllImport("user32.dll")]
        private static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int x, int y, int cx, int cy, uint uFlags);

        // 虚拟按键码：https://docs.microsoft.com/zh-cn/windows/desktop/inputdev/virtual-key-codes
        private const int VK_LBUTTON = 0x01; // 鼠标左键
        private const int VK_F3 = 0x72; // F3键-脑电软件打标快捷键
        //private const int VK_RBUTTON = 0x02;
        private static bool pressed = false;

        private const int WM_KEYDOWN = 0x0100;
        private const int WM_KEYUP = 0x0101;

        private int counter = 0; //按键次数计数器，限制只能按两次
        private int trial_num = 0; //记录实验组数

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            PromptTextBox.AppendText("请按下【开始】按钮，准备打标\n");
        }

        private void PromptTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            PromptTextBox.ScrollToEnd(); //当通信窗口内容有变化时保持滚动条在最下面
        }

        private void button_start_Click(object sender, RoutedEventArgs e)
        {
            button_start.IsEnabled = false;
            button_end.IsEnabled = true;

            PromptTextBox.AppendText("\n已按下【开始】按钮，之后两次点击鼠标左键将会打标\n");
            Thread thread = new Thread(thread_Tick);
            thread.Start();
        }

        private void thread_Tick()
        {
            IntPtr getwnd = FindWindow(null, "ActiView703-Lores.vi"); // 捕捉脑电软件进程
            pressed = false;

            while (counter < 2)
            {
                if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) == 0x8000 || pressed) //鼠标左键按键按下
                {
                    pressed = true;
                    if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) == 0) //鼠标左键按键抬起
                    {
                        while (true)
                        {
                            if (!SetForegroundWindow(getwnd))
                            {
                                continue; //保证已设置好当前窗口为脑电软件时再往下执行
                            }
                            while (GetForegroundWindow() != getwnd) { } // while循环保证脑电软件变成当前窗口时再往下执行
                            break;
                        }
                        counter++;
                        SendMessage(getwnd, WM_KEYDOWN, VK_F3, 0);
                        pressed = false;
                    }
                }
            }
        }

        private void button_end_Click(object sender, RoutedEventArgs e)
        {
            trial_num++;
            counter = 0;
            PromptTextBox.AppendText("\n已完成【" + trial_num.ToString() + "】组实验\n");
            PromptTextBox.AppendText("请按下【开始】按钮，准备第【" + (trial_num+1).ToString() + "】组实验打标\n");
            button_start.IsEnabled = true;
            button_end.IsEnabled = false;
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            Environment.Exit(0);
        }
    }
}
