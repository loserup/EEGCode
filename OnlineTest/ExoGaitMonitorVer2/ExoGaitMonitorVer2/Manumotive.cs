using System;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using CMLCOMLib;
using System.Windows.Controls.Primitives;

namespace ExoGaitMonitorVer2
{
    class Manumotive
    {
        #region 声明

        //设置角度
        private Motors motors = new Motors();
        private double angleSet;
        private int motorNumber;
        StatusBar statusBar;
        TextBlock statusInfoTextBlock;
        Button angleSetButton;
        Button emergencyStopButton;
        Button getZeroPointButton;
        Button zeroPointSetButton;
        Button PVT_Button;
        TextBox angleSetTextBox;
        TextBox motorNumberTextBox;

        static DispatcherTimer timer;

        const double FAST_VEL = 60000; //设置未到目标角度时较快的速度
        const double SLOW_VEL = 30000; //设置快到目标角度时较慢的速度
        const double ORIGIN_POINT = 2; //原点阈值
        const double TURN_POINT = 10; //快速转慢速的转变点

        #endregion

        #region 设置角度

        public void angleSetStart(Motors motorsIn, double angleSetIn, int motorNumberIn, StatusBar statusBarIn, TextBlock statusInfoTextBlockIn, 
                                  Button angleSetButtonIn, Button emergencyStopButtonIn, Button getZeroPointButtonIn, Button zeroPointSetButtonIn,
                                  Button PVT_ButtonIn, TextBox angleSetTextBoxIn, TextBox motorNumberTextBoxIn)//设置角度
        {
            motors = motorsIn;
            angleSet = angleSetIn;
            motorNumber = motorNumberIn;
            statusBar = statusBarIn;
            statusInfoTextBlock = statusInfoTextBlockIn;
            angleSetButton = angleSetButtonIn;
            emergencyStopButton = emergencyStopButtonIn;
            getZeroPointButton = getZeroPointButtonIn;
            zeroPointSetButton = zeroPointSetButtonIn;
            PVT_Button = PVT_ButtonIn;
            angleSetTextBox = angleSetTextBoxIn;
            motorNumberTextBox = motorNumberTextBoxIn;

            timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(10);
            timer.Tick += new EventHandler(angleSetTimer_Tick);
            timer.Start();
        }

        public void angleSetStop()
        {
            timer.Stop();
            timer.Tick -= new EventHandler(angleSetTimer_Tick);
        }

        private void angleSetTimer_Tick(object sender, EventArgs e)//电机按设置角度转动的委托
        {
            statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
            statusInfoTextBlock.Text = "正在执行";
            int i = motorNumber - 1;

            motors.profileSettingsObj.ProfileType = CML_PROFILE_TYPE.PROFILE_VELOCITY; // 选择速度模式控制电机

            if (angleSet > 0)
            {
                if (motors.ampObjAngleActual[i] < angleSet)
                {
                    motors.profileSettingsObj.ProfileVel = FAST_VEL;
                    motors.profileSettingsObj.ProfileAccel = FAST_VEL;
                    motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                    motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;

                    if (motors.ampObjAngleActual[i] > (angleSet - 5))
                    {
                        motors.profileSettingsObj.ProfileVel = SLOW_VEL;
                        motors.profileSettingsObj.ProfileAccel = SLOW_VEL;
                        motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                        motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;
                    }
                    try
                    {
                        motors.ampObj[i].MoveRel(1);
                    }
                    catch
                    {
                        statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
                        statusInfoTextBlock.Text = "电机" + (i + 1).ToString() + "已限位！";
                    }

                }
                else
                {
                    motors.ampObj[i].HaltMove();
                    motors.profileSettingsObj.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                    for (int j = 0; j < motors.motor_num; j++)
                    {
                        motors.profileSettingsObj.ProfileVel = 0;
                        motors.profileSettingsObj.ProfileAccel = 0;
                        motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                        motors.ampObj[j].ProfileSettings = motors.profileSettingsObj;
                    }
                    angleSetButton.IsEnabled = true;
                    emergencyStopButton.IsEnabled = false;
                    getZeroPointButton.IsEnabled = true;
                    zeroPointSetButton.IsEnabled = true;
                    PVT_Button.IsEnabled = true;

                    angleSetTextBox.IsReadOnly = false;
                    motorNumberTextBox.IsReadOnly = false;

                    timer.Stop();
                    timer.Tick -= new EventHandler(angleSetTimer_Tick);
                    statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 0, 122, 204));
                    statusInfoTextBlock.Text = "执行完毕";
                }
            }

            if (angleSet < 0)
            {
                if (motors.ampObjAngleActual[i] > angleSet)
                {
                    motors.profileSettingsObj.ProfileVel = FAST_VEL;
                    motors.profileSettingsObj.ProfileAccel = FAST_VEL;
                    motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                    motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;

                    if (motors.ampObjAngleActual[i] < (angleSet + 5))
                    {
                        motors.profileSettingsObj.ProfileVel = SLOW_VEL;
                        motors.profileSettingsObj.ProfileAccel = SLOW_VEL;
                        motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                        motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;
                    }
                    try
                    {
                        motors.ampObj[i].MoveRel(-1);
                    }
                    catch
                    {
                        statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
                        statusInfoTextBlock.Text = "电机" + (i + 1).ToString() + "已限位！"; ;
                    }
                }
                else
                {
                    motors.ampObj[i].HaltMove();
                    motors.profileSettingsObj.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                    for (int j = 0; j < motors.motor_num; j++)
                    {
                        motors.profileSettingsObj.ProfileVel = 0;
                        motors.profileSettingsObj.ProfileAccel = 0;
                        motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                        motors.ampObj[j].ProfileSettings = motors.profileSettingsObj;
                    }
                    angleSetButton.IsEnabled = true;
                    emergencyStopButton.IsEnabled = false;
                    getZeroPointButton.IsEnabled = true;
                    zeroPointSetButton.IsEnabled = true;
                    PVT_Button.IsEnabled = true;

                    angleSetTextBox.IsReadOnly = false;
                    motorNumberTextBox.IsReadOnly = false;

                    timer.Stop();
                    timer.Tick -= new EventHandler(angleSetTimer_Tick);
                    statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 0, 122, 204));
                    statusInfoTextBlock.Text = "执行完毕";
                }
            }
        }

        #endregion

        #region 回归原点

        public void getZeroPointStart(Motors motorsIn, StatusBar statusBarIn, TextBlock statusInfoTextBlockIn, Button angleSetButtonIn, Button emergencyStopButtonIn, Button getZeroPointButtonIn, Button zeroPointSetButtonIn,
                                      Button PVT_ButtonIn, TextBox angleSetTextBoxIn, TextBox motorNumberTextBoxIn)
        {
            motors = motorsIn;
            statusBar = statusBarIn;
            statusInfoTextBlock = statusInfoTextBlockIn;
            angleSetButton = angleSetButtonIn;
            emergencyStopButton = emergencyStopButtonIn;
            getZeroPointButton = getZeroPointButtonIn;
            zeroPointSetButton = zeroPointSetButtonIn;
            PVT_Button = PVT_ButtonIn;
            angleSetTextBox = angleSetTextBoxIn;
            motorNumberTextBox = motorNumberTextBoxIn;

            timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(10);
            timer.Tick += new EventHandler(getZeroPointTimer_Tick);
            timer.Start();
        }

        private void getZeroPointTimer_Tick(object sender, EventArgs e)//回归原点的委托
        {
            statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
            statusInfoTextBlock.Text = "正在回归原点";



            motors.profileSettingsObj.ProfileType = CML_PROFILE_TYPE.PROFILE_VELOCITY; // 选择速度模式控制电机

            for (int i = 0; i < motors.motor_num; i++)//电机回归原点
            {
                if (Math.Abs(motors.ampObjAngleActual[i]) > ORIGIN_POINT)
                {
                    if (motors.ampObjAngleActual[i] > 0)
                    {
                        if (motors.ampObjAngleActual[i] > TURN_POINT)
                        {
                            motors.profileSettingsObj.ProfileVel = FAST_VEL;
                            motors.profileSettingsObj.ProfileAccel = FAST_VEL;
                            motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                            motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;
                        }
                        else
                        {
                            motors.profileSettingsObj.ProfileVel = SLOW_VEL;
                            motors.profileSettingsObj.ProfileAccel = SLOW_VEL;
                            motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                            motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;
                        }
                        try
                        {
                            motors.ampObj[i].MoveRel(-1);
                        }
                        catch
                        {
                            statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
                            statusInfoTextBlock.Text = "电机" + (i + 1).ToString() + "已限位！";
                        }
                    }
                    else
                    {
                        if (motors.ampObjAngleActual[i] < -TURN_POINT)
                        {
                            motors.profileSettingsObj.ProfileVel = FAST_VEL;
                            motors.profileSettingsObj.ProfileAccel = FAST_VEL;
                            motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                            motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;
                        }
                        else
                        {
                            motors.profileSettingsObj.ProfileVel = SLOW_VEL;
                            motors.profileSettingsObj.ProfileAccel = SLOW_VEL;
                            motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                            motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;
                        }
                        try
                        {
                            motors.ampObj[i].MoveRel(1);
                        }
                        catch

                        {
                            statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 230, 20, 20));
                            statusInfoTextBlock.Text = "电机" + (i + 1).ToString() + "已限位！";
                        }
                    }
                }
                else
                {
                    motors.ampObj[i].HaltMove();
                }
            }

            if (Math.Abs(motors.ampObjAngleActual[0]) < ORIGIN_POINT && Math.Abs(motors.ampObjAngleActual[1]) < ORIGIN_POINT && Math.Abs(motors.ampObjAngleActual[2]) < ORIGIN_POINT && Math.Abs(motors.ampObjAngleActual[3]) < ORIGIN_POINT)
            {
                statusBar.Background = new SolidColorBrush(Color.FromArgb(255, 0, 122, 204));
                statusInfoTextBlock.Text = "回归原点完毕";
                motors.profileSettingsObj.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                for (int i = 0; i < motors.motor_num; i++)
                {
                    motors.profileSettingsObj.ProfileVel = 0;
                    motors.profileSettingsObj.ProfileAccel = 0;
                    motors.profileSettingsObj.ProfileDecel = motors.profileSettingsObj.ProfileAccel;
                    motors.ampObj[i].ProfileSettings = motors.profileSettingsObj;
                }
                angleSetButton.IsEnabled = true;
                emergencyStopButton.IsEnabled = false;
                zeroPointSetButton.IsEnabled = false;
                getZeroPointButton.IsEnabled = true;
                PVT_Button.IsEnabled = true;

                angleSetTextBox.IsReadOnly = false;
                motorNumberTextBox.IsReadOnly = false;

                timer.Stop();
                timer.Tick -= new EventHandler(getZeroPointTimer_Tick);
            }
        }
        #endregion
    }
}
