using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using MouseKeyboardLibrary;



// lxl
using System.Diagnostics;
using System.Runtime.InteropServices;

/*[DllImport("user32.dll")]
 public static extern void SwitchToThisWindow(IntPtr hWnd, Boolean fAltTab);

//[DllImport("USER32.DLL")]
public static extern void SwitchToThisWindow(IntPtr hwnd, Boolean fAltTab);

*/



namespace SampleApplication
{

    
    /*
     使用例子
     */
    public partial class HookTestForm : Form
    {

        MouseHook mouseHook = new MouseHook();
        KeyboardHook keyboardHook = new KeyboardHook();


        //lxl
        [DllImport("user32.dll ", SetLastError = true)]
        static extern void SwitchToThisWindow(IntPtr hWnd, bool fAltTab);

        [DllImport("user32.dll", EntryPoint = "keybd_event", SetLastError = true)]
        public static extern void keybd_event(Keys bVk, byte bScan, uint dwFlags, uint dwExtraInfo);
        //public static extern void keybd_event( byte bVk,byte bScan, int dwFlags, int dwExtraInfo );

        int Flag1 = 0;





        public HookTestForm()
        {
            InitializeComponent();
        }

        private void TestForm_Load(object sender, EventArgs e)
        {

            mouseHook.MouseMove += new MouseEventHandler(mouseHook_MouseMove);
            mouseHook.MouseDown += new MouseEventHandler(mouseHook_MouseDown);
            mouseHook.MouseUp += new MouseEventHandler(mouseHook_MouseUp);
            mouseHook.MouseWheel += new MouseEventHandler(mouseHook_MouseWheel);

            keyboardHook.KeyDown += new KeyEventHandler(keyboardHook_KeyDown);
            keyboardHook.KeyUp += new KeyEventHandler(keyboardHook_KeyUp);
            keyboardHook.KeyPress += new KeyPressEventHandler(keyboardHook_KeyPress);

            mouseHook.Start();
            keyboardHook.Start();

            SetXYLabel(MouseSimulator.X, MouseSimulator.Y);
            label2.Text = DateTime.Now.TimeOfDay.ToString();

        }

        void keyboardHook_KeyPress(object sender, KeyPressEventArgs e)
        {



        }

        void keyboardHook_KeyUp(object sender, KeyEventArgs e)
        {

            //AddKeyboardEvent(
            //    "KeyUp",
            //    e.KeyCode.ToString(),
            //    "",
            //    e.Shift.ToString(),
            //    e.Alt.ToString(),
            //    e.Control.ToString()
            //    );

        }

        void keyboardHook_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.L)
            {
                string pName1 = "ActiView703-Vari";
                Process[] temp1 = Process.GetProcessesByName(pName1);

                if (temp1.Length > 0)//如果查找到  
                {
                    IntPtr handle = temp1[0].MainWindowHandle;
                    SwitchToThisWindow(handle, true);    // 激活，显示在最前 
                    keybd_event(Keys.F3, 0, 0, 0); //按下F10
                                                   //keybd_event((byte)Keys.F10, 0, 0, 0); //按下F10
                                                   //keybd_event((byte)Keys.F10, 0, 2, 0); //按下后松开F10
                }
            }


            //AddKeyboardEvent(
            //    "KeyDown",
            //    e.KeyCode.ToString(),
            //    "",
            //    e.Shift.ToString(),
            //    e.Alt.ToString(),
            //    e.Control.ToString()
            //    );

        }

        void mouseHook_MouseWheel(object sender, MouseEventArgs e)
        {

            //AddMouseEvent(
            //    "MouseWheel",
            //    "",
            //    "",
            //    "",
            //    e.Delta.ToString()
            //    );

        }

        void mouseHook_MouseUp(object sender, MouseEventArgs e)
        {

      


        }

        void mouseHook_MouseDown(object sender, MouseEventArgs e)
        {
            
            if (Flag1> 0 && Flag1<3)
            {
                label1.Text = "记录中";
                string pName1 = "Axis Neuron";
                string pName2 = "ActiView703-Vari";
                Process[] temp1 = Process.GetProcessesByName(pName1);
                Process[] temp2 = Process.GetProcessesByName(pName2);

                if (temp2.Length > 0)//如果查找到  
                {
                    IntPtr handle = temp2[0].MainWindowHandle;
                    SwitchToThisWindow(handle, true);    // 激活，显示在最前 
                    keybd_event(Keys.F3, 0, 0, 0); //按下F10
                }
                Flag1 = Flag1 + 1;
            }
        //label2.Text = DateTime.Now.TimeOfDay.ToString();



        //if (Flag1 == 1)
        //{



        //    string pName1 = "ActiView703-Vari";
        //    Process[] temp1 = Process.GetProcessesByName(pName1);

        //    if (temp1.Length > 0)//如果查找到  
        //    {
        //        IntPtr handle = temp1[0].MainWindowHandle;
        //        SwitchToThisWindow(handle, true);    // 激活，显示在最前 
        //        keybd_event(Keys.F3, 0, 0, 0); //按下F10
        //                                       //keybd_event((byte)Keys.F10, 0, 0, 0); //按下F10
        //                                       //keybd_event((byte)Keys.F10, 0, 2, 0); //按下后松开F10
        //    }

        //string pName = "Axis Neuron";
        //Process[] temp = Process.GetProcessesByName(pName);

        //if (temp.Length > 0)//如果查找到  
        //{
        //    IntPtr handle = temp[0].MainWindowHandle;

        //    SwitchToThisWindow(handle, true);    // 激活，显示在最前 
        //    keybd_event(Keys.Enter, 0, 0, 0); //按下F10
        //                                      //keybd_event((byte)Keys.F10, 0, 0, 0); //按下F10
        //                                      //keybd_event((byte)Keys.F10, 0, 2, 0); //按下后松开F10
        //}

        //Flag1 = Flag1 + 1;
        //}



        //      


        //AddMouseEvent(
        //    "MouseDown",
        //    e.Button.ToString(),
        //    e.X.ToString(),
        //    e.Y.ToString(),
        //    ""
        //    );


    }

        void mouseHook_MouseMove(object sender, MouseEventArgs e)
        {

            //SetXYLabel(e.X, e.Y);

        }

        void SetXYLabel(int x, int y)
        {

            //curXYLabel.Text = String.Format("Current Mouse Point: X={0}, y={1}", x, y);

        }

        void AddMouseEvent(string eventType, string button, string x, string y, string delta)
        {

            listView1.Items.Insert(0,
                new ListViewItem(
                    new string[]{
                        eventType,
                        button,
                        x,
                        y,
                        delta
                    }));

        }

        void AddKeyboardEvent(string eventType, string keyCode, string keyChar, string shift, string alt, string control)
        {

            listView2.Items.Insert(0,
                 new ListViewItem(
                     new string[]{
                        eventType,
                        keyCode,
                        keyChar,
                        shift,
                        alt,
                        control
                }));

        }

        private void TestForm_FormClosed(object sender, FormClosedEventArgs e)
        {

            // Not necessary anymore, will stop when application exits

            //mouseHook.Stop();
            //keyboardHook.Stop();

        }

        private void listView1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            Flag1 = 1;
            //label1.Text = "记录中";
            //string pName1 = "Axis Neuron";
            //string pName2 = "ActiView703-Vari";
            //Process[] temp1 = Process.GetProcessesByName(pName1);
            //Process[] temp2 = Process.GetProcessesByName(pName2);

            //if (temp2.Length > 0)//如果查找到  
            //{
            //    IntPtr handle = temp2[0].MainWindowHandle;
            //    SwitchToThisWindow(handle, true);    // 激活，显示在最前 
            //    keybd_event(Keys.F3, 0, 0, 0); //按下F10
                                              //keybd_event((byte)Keys.F10, 0, 0, 0); //按下F10
                                              //keybd_event((byte)Keys.F10, 0, 2, 0); //按下后松开F10

                //    System.Threading.Thread.Sleep(1000);
                //    SwitchToThisWindow(handle, true);
                //    keybd_event(Keys.Enter, 0, 0, 0);


                //    if (temp2.Length > 0)//如果查找到  
                //    {
                //        handle = temp2[0].MainWindowHandle;
                //        SwitchToThisWindow(handle, true);    // 激活，显示在最前 
                //        keybd_event(Keys.F3, 0, 0, 0); //按下F10
                //    }
                //    //SwitchToThisWindow(handle, true);    // 激活，显示在最前 
                //    //keybd_event(Keys.Enter, 0, 0, 0);
                
            }

        private void button2_Click(object sender, EventArgs e)
        {
            Flag1 = 0;
            //label1.Text = "已结束";
            //string pName1 = "Axis Neuron";
            //string pName2 = "ActiView703-Vari";
            //Process[] temp1 = Process.GetProcessesByName(pName1);
            //Process[] temp2 = Process.GetProcessesByName(pName2);
            //IntPtr handle1 = temp1[0].MainWindowHandle;
            //IntPtr handle2 = temp2[0].MainWindowHandle;

            //if (temp1.Length > 0)//如果查找到  
            //{
                
            //    SwitchToThisWindow(handle1, true);    // 激活，显示在最前 
            //    keybd_event(Keys.R, 0, 0, 0); //按下F10
            //                                  //keybd_event((byte)Keys.F10, 0, 0, 0); //按下F10
            //                                  //keybd_event((byte)Keys.F10, 0, 2, 0); //按下后松开F10
            //                                  //SwitchToThisWindow(handle, true);    // 激活，显示在最前 
            //                                  //keybd_event(Keys.Enter, 0, 0, 0);
            //    //SwitchToThisWindow(handle, false);
            //    //if (temp2.Length > 0)//如果查找到  
            //    //{
            //    //    SwitchToThisWindow(handle2, true);    // 激活，显示在最前 
            //    //    keybd_event(Keys.F3, 0, 0, 0); //按下F10
            //    //}
            //}
        }
    }
}
