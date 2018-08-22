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

        // 虚拟按键码：https://docs.microsoft.com/zh-cn/windows/desktop/inputdev/virtual-key-codes
        private const int VK_LBUTTON = 0x01; // 鼠标左键
        private const int VK_F3 = 0x72; // F3键-脑电软件打标快捷键
        //private const int VK_RBUTTON = 0x02;
        private static bool pressed = false;

        private const int WM_KEYDOWN = 0x0100;
        private const int WM_KEYUP = 0x0101;

        private int counter = 0; //按键次数计数器，限制只能按两次

        private DispatcherTimer thread; //本控件监视鼠标左键是否按下的线程，由点击【开始】按钮启动

        private void button_start_Click(object sender, RoutedEventArgs e)
        {
            button_end.IsEnabled = true;
            button_start.IsEnabled = false;

            thread = new DispatcherTimer();
            thread.Tick += new EventHandler(thread_Tick); //Tick是超过计时器间隔时发生事件，此处为Tick增加了一个叫ShowCurTimer的取当前时间并扫描串口的委托
            thread.Interval = TimeSpan.FromMilliseconds(10);
            thread.Start();
        }

        private void thread_Tick(object sender, EventArgs e)
        {
            ////这里是当需要针对某个应用的时候可以先获取这个应用的句柄，接着就可以对这个应用发送想要的键盘操作
            IntPtr getwnd = FindWindow(null, "ActiView703-Lores.vi"); // 捕捉脑电软件进程

            while (counter < 2)
            {
                if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) == 0x8000 || pressed) //鼠标左键按键按下
                {
                    pressed = true;
                    if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) == 0) //鼠标左键按键抬起
                    {
                        //Console.Write("1111\n");
                        if (!SetForegroundWindow(getwnd))
                        {
                            // 若未找到进程则退出
                            //Console.Write("-1\n");
                            return;
                        }
                        while (GetForegroundWindow() != getwnd) { } // while循环保证脑电软件变成当前窗口时再往下执行
                        SendMessage(getwnd, WM_KEYDOWN, VK_F3, 0);
                        pressed = false;
                        counter += 1;
                    }
                }
                //System.Threading.Thread.Sleep(20);
            }
        }

        private void button_end_Click(object sender, RoutedEventArgs e)
        {
            counter = 0;

            thread.Stop();
            thread.Tick -= new EventHandler(thread_Tick);

            button_end.IsEnabled = false;
            button_start.IsEnabled = true;
        }
    }
}
