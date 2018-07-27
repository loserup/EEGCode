using System;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.IO;
using System.Windows.Threading;
using CMLCOMLib;
using System.Windows.Input;
using Microsoft.Research.DynamicDataDisplay;
using Microsoft.Research.DynamicDataDisplay.DataSources;
using System.Threading;
using LattePanda.Firmata;
using System.Threading.Tasks;
using System.Net.Sockets;
using System.Net;

namespace ExoGaitMonitorVer2
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
           
          #region 声明
        //主界面窗口

        //CMO
        public Motors motors = new Motors(); //声明电机类

        //手动操作设置
        private Manumotive manumotive = new Manumotive();

        //传感器
       
  
        private int PositionState = 0;
        private int Dete =2;
        private int ss=0;

        //PVT模式
        private PVT pvt = new PVT();

      //Stand up
     private Standup2 stand2 = new Standup2();
         
       Arduino arduino = new Arduino();

        DispatcherTimer Detection = new DispatcherTimer();
        public delegate void showData(string msg);//通信窗口输出
        private TcpClient client;
        private TcpListener server;
        private const int bufferSize = 8000;
        private double ProportionValue = 2 * Math.PI;
        private int TimeValue = 65;
        private int MiddleGaitTime = 70;
        private int LongGiatTime = 75;

        #endregion

        #region 界面初始化

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                motors.motors_Init();
                //cp.plotStart(motors, statusBar, statusInfoTextBlock);
            }
            catch(Exception)
            {
                MessageBox.Show("驱动器初始化失败");
                statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
                statusInfoTextBlock.Text = "窗口初始化失败！";
            }

            Thread shoutdown = new Thread(Select);
            shoutdown.Start();
        }

        private void Select()
        {
            while(true )
            {
                if (PositionState == 0 && Dete == -1 && ss == 0)
                {
                    PositionState = 1;
                    Dete = 2;
                    ss = 1;
                }
                if (PositionState == 0 && Dete == 1 && ss == 0)
                {
                    PositionState = 2;
                    Dete = 2;
                    ss = 2;
                }
                if (PositionState == 1 && Dete == -1 && ss == 0)
                {
                    PositionState = 3;
                    Dete = 2;
                    ss = 3;
                }
                if (PositionState == 1 && Dete == 1 && ss == 0)
                {
                    PositionState = 4;
                    Dete = 2;
                    ss = 4;
                }
                if (PositionState == 2 && Dete == -1 && ss == 0)
                {
                    PositionState = 3;
                    Dete = 2;
                    ss = 5;
                }
                if (PositionState == 2 && Dete == 1 && ss == 0)
                {
                    PositionState = 4;
                    Dete = 2;
                    ss = 6;
                }
                if (PositionState == 3 && Dete == -1 && ss == 0)
                {
                    PositionState = 1;
                    Dete = 2;
                    ss = 7;
                }
                if (PositionState == 3 && Dete == 1 && ss == 0)
                {
                    PositionState = 2;
                    Dete = 2;
                    ss = 8;
                }
                if (PositionState == 4 && Dete == -1 && ss == 0)
                {
                    PositionState = 1;
                    Dete = 2;
                    ss = 9;
                }
                if (PositionState == 4 && Dete == 1 && ss == 0)
                {
                    PositionState = 2;
                    Dete = 2;
                    ss = 10;
                }
                if (ss!=0)
                {
                    Detection.Stop();
                    switch (ss)
                    {
                        case 1:
                            //MessageBox.Show("1");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\左脚起始步低.txt", ProportionValue, 115, TimeValue);
                            }
                           catch(Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 2:
                            //MessageBox.Show("2");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\左脚起始步高.txt", ProportionValue, 115, MiddleGaitTime);
                            }
                          catch(Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 3:
                            //MessageBox.Show("3");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\左脚在前低到右脚前伸低.txt", ProportionValue, 115, TimeValue);
                            }
                            catch (Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 4:
                            //MessageBox.Show("4");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\左脚在前低到右脚前伸高.txt", ProportionValue, 115, MiddleGaitTime);
                            }
                            catch (Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 5:
                            //MessageBox.Show("5");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\左脚在前高到右脚前伸低.txt", ProportionValue, 115, MiddleGaitTime);
                            }
                            catch(Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 6:

                            //MessageBox.Show("6");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\左脚在前高到右脚前伸高.txt", ProportionValue, 115, LongGiatTime);
                            }
                            catch (Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                          break;
                        case 7:
                            //MessageBox.Show("7");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\右脚在前低到左脚前伸低.txt", ProportionValue, 115, TimeValue);
                            }
                            catch (Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 8:

                            //MessageBox.Show("8");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\右脚在前低到左脚前伸高.txt", ProportionValue, 115, MiddleGaitTime);
                            }
                            catch (Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 9:
                            //MessageBox.Show("9");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\右脚在前高到左脚前伸低.txt", ProportionValue, 115, MiddleGaitTime);
                            }
                            catch(Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 10:
                            //MessageBox.Show("10");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\右脚在前高到左脚前伸高.txt", ProportionValue, 115, LongGiatTime);
                            }
                            catch (Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 11:
                            MessageBox.Show("11");
                            try


                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\左脚向前低收步.txt", ProportionValue, 115, TimeValue);
                            }
                            catch (Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        case 12:
                            MessageBox.Show("12");
                            try
                            {
                                pvt.StartPVT(motors, "..\\..\\InputData\\左脚向前高收步.txt", ProportionValue, 115, TimeValue);
                            }
                            catch (Exception e)
                            {
                                MessageBox.Show(e.ToString());
                            }
                            break;
                        default:
                            break;
                    }
                    ss = 0;
                    Detection.Start();
                }
                Thread.Sleep(100);
            }
            


        }
    
      private void Window_Closed(object sender, EventArgs e)
        {
            //server.Stop();
        }

       #endregion
     #region 手动操作设置 Manumotive

        private void angleSetButton_Click(object sender, RoutedEventArgs e)//点击【执行命令】按钮时执行
        {
            angleSetButton.IsEnabled = false;
            emergencyStopButton.IsEnabled = true;
            getZeroPointButton.IsEnabled = false;
            zeroPointSetButton.IsEnabled = false;
            PVT_Button.IsEnabled = false;

            angleSetTextBox.IsReadOnly = true;
            motorNumberTextBox.IsReadOnly = true;


            int motorNumber = Convert.ToInt16(motorNumberTextBox.Text);
            int i = motorNumber - 1;

            motors.ampObj[i].PositionActual = 0;

            manumotive.angleSetStart(motors, Convert.ToDouble(angleSetTextBox.Text), Convert.ToInt16(motorNumberTextBox.Text), statusBar, statusInfoTextBlock, 
                                     angleSetButton, emergencyStopButton, getZeroPointButton, zeroPointSetButton, PVT_Button, angleSetTextBox, motorNumberTextBox);
        }

        private void emergencyStopButton_Click(object sender, RoutedEventArgs e)//点击【紧急停止】按钮时执行
        {
            emergencyStopButton.IsEnabled = false;
            angleSetButton.IsEnabled = true;
            getZeroPointButton.IsEnabled = true;
            angleSetTextBox.IsReadOnly = false;
            motorNumberTextBox.IsReadOnly = false;
            int motorNumber = Convert.ToInt16(motorNumberTextBox.Text);
            int i = motorNumber - 1;

            motors.ampObj[i].HaltMove();
            manumotive.angleSetStop();
        }
       private void zeroPointSetButton_Click(object sender, RoutedEventArgs e)//点击【设置原点】按钮时执行
        {
            motors.ampObj[0].PositionActual = 1330.683488;
            motors.ampObj[1].PositionActual = -665.3417438;
            motors.ampObj[2].PositionActual = 665.3417438;
            motors.ampObj[3].PositionActual = -1330.683488;

            zeroPointSetButton.IsEnabled = false;

            statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 0, 122, 204));
            statusInfoTextBlock.Text = "原点设置完毕";
        }

        private void getZeroPointButton_Click(object sender, RoutedEventArgs e)//点击【回归原点】按钮时执行
        {
            angleSetTextBox.IsReadOnly = true;
            motorNumberTextBox.IsReadOnly = true;
            PVT_Button.IsEnabled = false;
            getZeroPointButton.IsEnabled = false;
            angleSetButton.IsEnabled = false;
            emergencyStopButton.IsEnabled = false;
            zeroPointSetButton.IsEnabled = false;
            PositionState = 0;

            manumotive.getZeroPointStart(motors, statusBar, statusInfoTextBlock, angleSetButton, emergencyStopButton, getZeroPointButton, 
                                          zeroPointSetButton, PVT_Button, angleSetTextBox, motorNumberTextBox);
        }

        #endregion

        private void PVT_Button_Click(object sender, RoutedEventArgs e)//进入PVT模式
        {
            Button bt = sender as Button;
            double positon = motors.ampObj[3].PositionActual;
            if (bt.Content.ToString() == "PVT Mode")
            {
               
                angleSetButton.IsEnabled = false;
                getZeroPointButton.IsEnabled = false;
               

                statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
                statusInfoTextBlock.Text = "PVT模式";
                bt.Content = "Stop";
                if(positon<100000)
                {
                    try
                    {
                        pvt.StartPVT(motors, "..\\..\\InputData\\6步新.txt",360,45,15);
                    }
                   catch(Exception)
                    {
                        MessageBox.Show(e.ToString());
                    }
                }
               
            }

            else
            {
                angleSetButton.IsEnabled = true;
                getZeroPointButton.IsEnabled = true;

                motors.Linkage.HaltMove();

                statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 0, 122, 204));
                statusInfoTextBlock.Text = "PVT控制模式已停止";
                bt.Content = "PVT Mode";
            }
        }

       private void Sit_button_Click(object sender, RoutedEventArgs e)
        {
            Button bt = sender as Button;
            double positon = motors.ampObj[3].PositionActual;
            if(bt.Content.ToString()=="Sit Down")
            {
                PVT_Button.IsEnabled = false;
                Stand_up_Button.IsEnabled = false;
                angleSetButton.IsEnabled = false;
                getZeroPointButton.IsEnabled = false;
               
                statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
                statusInfoTextBlock.Text = "坐下模式";
                bt.Content = "Stop";
                if (positon < 430000)
                {
                    try
                    {
                        pvt.start_Sitdown2(motors);
                    }
                    catch(Exception)
                    { MessageBox.Show(e.ToString()); }
                }
               
            }
            else
            {
                angleSetButton.IsEnabled = true;
                getZeroPointButton.IsEnabled = true;
                PVT_Button.IsEnabled = true;
                Stand_up_Button.IsEnabled = true;

                //motors.Linkage.HaltMove();

                statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 0, 122, 204));
                statusInfoTextBlock.Text = "坐下模式已停止";
                bt.Content = "Sit Down";
            }
        }

        private void Stand_up_Button_Click(object sender, RoutedEventArgs e)
        {
            Button bt = sender as Button;
            double positon = motors.ampObj[3].PositionActual;
            if (bt.Content.ToString()=="Stand Up")
            {
                PVT_Button.IsEnabled = false;
                angleSetButton.IsEnabled = false;
                 Sit_button.IsEnabled = false;

                statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
                statusInfoTextBlock.Text = "起立模式";
                bt.Content = "Stop";
                if(positon>1000000)
                {
                    try
                    {
                        stand2.start_Standup2(motors);
                    }
                    catch(Exception)
                    {
                        MessageBox.Show(e.ToString());
                    }
                }
                
            }
            else
            {
                PVT_Button.IsEnabled = true;
                angleSetButton.IsEnabled = true;
               
                Sit_button.IsEnabled = true;

                //motors.Linkage.HaltMove();

                statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 0, 122, 204));
                statusInfoTextBlock.Text = "起立模式已结束";
                bt.Content = "Stand Up";
            }
        }

        private void Motor4_Pos_TextBox_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void button1_Click(object sender, RoutedEventArgs e)
        {
            Dete = 1;
        }

        private void button_1_Click(object sender, RoutedEventArgs e)
        {
            Dete = -1;
        }
        struct IpAndPort
        {
            public string Ip;
            public string Port;
        }

        private void switch_Click(object sender, RoutedEventArgs e)
        {
            if(IPAdressTextBox.Text.Trim()==string.Empty)
            {
                ComWinTextBox.Dispatcher.Invoke(new showData(ComWinTextBox.AppendText), "请填入服务器IP地址\n");
                return;
            }
            if (PortTextBox.Text.Trim() == string.Empty)
            {
                ComWinTextBox.Dispatcher.Invoke(new showData(ComWinTextBox.AppendText), "请填入服务器端口号\n");
                return;
            }
            Thread thread = new Thread(reciveAndListener);
            IpAndPort ipHePort = new IpAndPort();
            ipHePort.Ip = IPAdressTextBox.Text;
            ipHePort.Port = PortTextBox.Text;
            thread.Start((object)ipHePort);
        }

        private void btnSend_Click(object sender, RoutedEventArgs e)
        {
            if (stxtSendMsg.Text.Trim() != string.Empty)
            {
                NetworkStream sendStream = client.GetStream();//获得用于数据传输的流
                byte[] buffer = Encoding.Default.GetBytes(stxtSendMsg.Text.Trim());//将数据存在缓冲中
                sendStream.Write(buffer, 0, buffer.Length);//最终写入流中
                string showmsg = Encoding.Default.GetString(buffer, 0, buffer.Length);
                //ComWinTextBox1.AppendText("发送给服务端数据：" + showmsg + "\n");
                stxtSendMsg.Text = string.Empty;
            }
        }

        private void ComWinTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            ComWinTextBox.ScrollToEnd();//当通信窗口内容变化时滚动条定位在最下面
        }
        private void reciveAndListener(object ipAndPort)
        {
            IpAndPort ipHePort = (IpAndPort)ipAndPort;
            IPAddress ip = IPAddress.Parse(ipHePort.Ip);
            server = new TcpListener(ip, int.Parse(ipHePort.Port));
            Socket socketserver = server.Server;
            bool conma = !((socketserver.Poll(1000, SelectMode.SelectRead) && (socketserver.Available == 0)) || !socketserver.Connected);
            server.Start();//启动监听

            ComWinTextBox.Dispatcher.Invoke(new showData(ComWinTextBox.AppendText), "服务端开启侦听....\n");
            
            client = server.AcceptTcpClient();
            ComWinTextBox.Dispatcher.Invoke(new showData(ComWinTextBox.AppendText), "有客户端请求连接，连接已建立！");//AcceptTcpClient 是同步方法，会阻塞进程，得到连接对象后才会执行这一步
            //获取流
            NetworkStream reciveStream = client.GetStream();
            
            
            do
            {
                //获取连接的客户d端的对象
                //if (socketserver.Poll(10, SelectMode.SelectRead) == false)
                //{
                //    client = server.AcceptTcpClient();
                //    ComWinTextBox.Dispatcher.Invoke(new showData(ComWinTextBox.AppendText), "有客户端请求连接，连接已建立！");//AcceptTcpClient 是同步方法，会阻塞进程，得到连接对象后才会执行这一步
                //    reciveStream = client.GetStream();
                //}


                byte[] buffer = new byte[bufferSize];
                int msgSize;
                try
                {
                    lock (reciveStream)
                    {
                        msgSize = reciveStream.Read(buffer, 0, bufferSize);
                    }

                    if (msgSize == 0)
                    {
                        //获取连接的客户d端的对象
                        client = server.AcceptTcpClient();
                        //ComWinTextBox.Dispatcher.Invoke(new showData(ComWinTextBox.AppendText), "有客户端请求连接，连接已建立！");//AcceptTcpClient 是同步方法，会阻塞进程，得到连接对象后才会执行这一步
                        reciveStream = client.GetStream();
                        continue;
                    }
                    string msg = Encoding.Default.GetString(buffer, 0, bufferSize);
                    ComWinTextBox.Dispatcher.Invoke(new showData(ComWinTextBox.AppendText), "\n客户端曰：" +
                        Encoding.Default.GetString(buffer, 0, msgSize));
                    string de= Encoding.Default.GetString(buffer, 0, msgSize);
                    Dete = Convert.ToInt16(de);
                    
                }
                catch
                {
                    ComWinTextBox.Dispatcher.Invoke(new showData(ComWinTextBox.AppendText), "\n 出现异常：连接被迫关闭");
                    break;
                }
            } while (true);
            Thread.Sleep(100);
        }
        
        }
}
