using System;
using System.Text;
using System.IO;
using CMLCOMLib;
using System.Windows.Threading;
using System.Threading;
using System.Threading.Tasks;


namespace ExoGaitMonitorVer2
{
    class PVT
    {
      public void StartPVT(Motors motors,  string adress,double unit,int timevalue1,int timevalue2)
        {
            #region 
            //计算轨迹位置，速度和时间间隔序列
            //原始数据
           
            string[] ral = File.ReadAllLines(adress, Encoding.Default); //相对目录是在bin/Debug下，所以要回溯到上两级目录
            int lineCounter = ral.Length; //获取步态数据行数
            string[] col = (ral[0] ?? string.Empty).Split(new char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
            int colCounter = col.Length; //获取步态数据列数
            double[,] pos0 = new double[lineCounter, colCounter]; //原始位置数据
            double[,] vel = new double[lineCounter, colCounter]; //速度
            int[] times = new int[lineCounter]; //时间间隔
            for (int i = 0; i < lineCounter; i++)
            {
                
                    times[i] =timevalue2 ; //【设置】时间间隔
                
                
                string[] str = (ral[i] ?? string.Empty).Split(new char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
                for (int j = 0; j < colCounter; j++)
                {
                    pos0[i, j] = double.Parse(str[j]) / (unit / motors.RATIO) * motors.userUnits[j] * -1;
                }
            }
            times[lineCounter - 1] = 0;
  

            for (int i = 0; i < lineCounter - 1; i++)
            {
                for (int j = 0; j < colCounter; j++)
                {

                    vel[i, j] = (pos0[i + 1, j] - pos0[i, j]) * 1000.0 / (times[i]);
                }
            }
            vel[lineCounter - 1, 0] = 0;
            vel[lineCounter - 1, 1] = 0;
            vel[lineCounter - 1, 2] = 0;
            vel[lineCounter - 1, 3] = 0;
            #endregion
          

            motors.Linkage.TrajectoryInitialize(pos0, vel, times, 500); //开始步态
            motors.ampObj[0].WaitMoveDone(10000);
            motors.ampObj[1].WaitMoveDone(10000);
            motors.ampObj[2].WaitMoveDone(10000);
            motors.ampObj[3].WaitMoveDone(10000);


        }
        public void start_Sitdown2(Motors motor)
        {
            #region
              double[,] KeyPos ={ { -56888.88889, 227555.5556, -227555.5556, 56888.88889 },
                                { -159288.8889, 599608.8889, -599608.8889, 159288.8889 },
                                { -349297.7778, 893155.5556, -893155.5556, 349297.7778 },
                                { -728177.7778, 1169635.556, -1169635.556, 728177.7778 },
                                { -989866.6667, 1365333.333, -1365333.333, 989866.6667 },
                                { -1035377.778, 1285688.889, -1285688.889, 1035377.778 },
                                { -1137777.778, 1196942.222, -1196942.222, 1137777.778 }};
                #endregion
                ProfileSettingsObj profileParameters = new ProfileSettingsObj();    //用于设置电机参数
                double MotorVelocity = 72;
                double MotorAcceleration = 20;
                double MotorDeceleration = 20;

                double[,] DeltaP = new double[7, 4];
                for (int s = 0; s < 7; s++)
                {
                    for (int j = 0; j < motor.motor_num; j++)
                    {
                        if (s == 0)
                        {
                            DeltaP[s, j] = Math.Abs(KeyPos[s, j] - 0);
                        }
                        else
                        {
                            DeltaP[s, j] = Math.Abs(KeyPos[s, j] - KeyPos[s - 1, j]);
                        }

                    }
                    for (int i = 0; i < motor.motor_num; i++)
                    {
                        double MaxDeltaP = DeltaP[s, 0];
                        if (MaxDeltaP < DeltaP[s, i])
                        {
                            MaxDeltaP = DeltaP[s, i];
                        }
                        profileParameters = motor.ampObj[i].ProfileSettings;
                        profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP[s, i] / MaxDeltaP;    //单位为°/s
                        profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                        profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                        profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                        motor.ampObj[i].ProfileSettings = profileParameters;
                        motor.ampObj[i].MoveAbs(KeyPos[s, i]);
                    }
                  
                }
                                   
        }
       }
    }
