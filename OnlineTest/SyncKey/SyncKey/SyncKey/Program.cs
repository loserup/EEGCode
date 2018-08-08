using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

//这个例程与例程01的区别是平台是x64而不是x86，这里的private struct INPUT的字节数完全不同，是8个字节，而不是01例程的4个字节，需要注意！！！
//若使用01例程的4个字节虽然不会报错，但是不可用的，只会让鼠标左键点击好像出现一些bug，但没有键盘效果！！！
namespace SyncKey
{
    class Program
    {
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

        static void Main(string[] args)
        {
            ////这里是当需要针对某个应用的时候可以先获取这个应用的句柄，接着就可以对这个应用发送想要的键盘操作
            IntPtr getwnd = FindWindow(null, "ActiView703-Lores.vi"); // 捕捉脑电软件进程

            while (true)
            {
                if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) == 0x8000 || pressed) //鼠标左键按键按下
                {
                    pressed = true;
                    if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) == 0) //鼠标左键按键抬起
                    {
                        Console.Write("1111\n");
                        if (!SetForegroundWindow(getwnd))
                        {
                            // 若未找到进程则退出
                            Console.Write("-1\n");
                            return;
                        }
                        while (GetForegroundWindow() != getwnd) { }
                        SendMessage(getwnd, WM_KEYDOWN, VK_F3, 0);                       
                        pressed = false;
                    }
                }
                System.Threading.Thread.Sleep(20);
            }
        }
    }
}
