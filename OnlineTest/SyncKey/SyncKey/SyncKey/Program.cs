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
        [System.Runtime.InteropServices.DllImport("user32.dll", EntryPoint = "SendInput", SetLastError = true)]
        private static extern uint SendInput(uint numberOfInputs, INPUT[] inputs, int sizeOfInputStructure);

        [DllImport("User32.dll", EntryPoint = "FindWindow")]
        private static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern short GetAsyncKeyState(int vKey);

        // 虚拟按键码：https://docs.microsoft.com/zh-cn/windows/desktop/inputdev/virtual-key-codes
        private const int VK_LBUTTON = 0x01; // 鼠标左键
        private const int VK_F3 = 0x72; // F3键-脑电软件打标快捷键
        private const int VK_RBUTTON = 0x02;
        //private const int VK_TAB = 0x09;
        //private const int VK_WIN = 0x5B;
        private static bool pressed = false;

        [Flags()]
        private enum InputType
        {
            INPUT_MOUSE = 0,
            INPUT_KEYBOARD = 1,
            INPUT_HARDWARE = 2,
        }

        [Flags()]
        private enum KEYEVENTF
        {
            EXTENDEDKEY = 0x0001,
            KEYUP = 0x0002,
            UNICODE = 0x0004,
            SCANCODE = 0x0008,
        }

        /// <summary>
        /// 请特别注意这里！！！！！x64平台使用8个字节偏移，x86平台使用4个字节的偏移！！！
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        private struct INPUT
        {
            [FieldOffset(0)]
            public Int32 type;//0-MOUSEINPUT;1-KEYBDINPUT;2-HARDWAREINPUT     
            [FieldOffset(8)]
            public KEYBDINPUT ki;
            [FieldOffset(8)]
            public MOUSEINPUT mi;
            [FieldOffset(8)]
            public HARDWAREINPUT hi;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct KEYBDINPUT
        {
            public Int16 wVk;
            public Int16 wScan;
            public Int32 dwFlags;
            public Int32 time;
            public IntPtr dwExtraInfo;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct MOUSEINPUT
        {
            public Int32 dx;
            public Int32 dy;
            public Int32 mouseData;
            public Int32 dwFlags;
            public Int32 time;
            public IntPtr dwExtraInfo;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct HARDWAREINPUT
        {
            public Int32 uMsg;
            public Int16 wParamL;
            public Int16 wParamH;
        }

        static void Main(string[] args)
        {
            UInt32 result = 0;
            INPUT input_key_F3_down = new INPUT();
            INPUT input_key_F3_up = new INPUT();

            ////这里是当需要针对某个应用的时候可以先获取这个应用的句柄，接着就可以对这个应用发送想要的键盘操作
            IntPtr getwnd = FindWindow(null, "ActiView703-Lores.vi"); // 捕捉脑电软件进程
            //IntPtr getwnd = GetDesktopWindow();


            input_key_F3_down.type = input_key_F3_down.type = (int)InputType.INPUT_KEYBOARD;
            input_key_F3_down.ki.wVk = input_key_F3_down.ki.wVk = VK_F3;
            input_key_F3_down.ki.dwFlags = 0;
            input_key_F3_up.ki.dwFlags = (Int32)KEYEVENTF.KEYUP;

            INPUT[] inputs = new INPUT[] { input_key_F3_down, input_key_F3_up};

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
                        // 在脑电软件外部点击左键后会有那么一瞬间脑电软件非当前进程，造成虚拟F3按键没有在软件内按下，所以这里做个延时，延时越长成功率越大，但对同步率的影响也越大
                        System.Threading.Thread.Sleep(200); 
                        result = SendInput(2, inputs, Marshal.SizeOf(typeof(INPUT)));

                        pressed = false;
                    }
                }
                System.Threading.Thread.Sleep(20);
            }
        }
    }
}
