using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CMLCOMLib;
using System.Windows.Threading;
using System.Threading;

namespace ExoGaitMonitorVer2
{
    class Standup2
    {
        #region
        double[] KeyPos1 = new double[4] { -1132088.889, 1228800, -1228800, 1132088.889 };
        double[] KeyPos2 = new double[4] { -1120711.111, 1251555.556, -1251555.556, 1120711.111 };
        double[] KeyPos3 = new double[4] { -1092266.667, 1308444.444, -1308444.444, 1092266.667 };
        double[] KeyPos4 = new double[4] { -1069511.111, 1388088.889, -1388088.889, 1069511.111 };
        double[] KeyPos5 = new double[4] { -1072011.111, 1389588.889, -1389588.889, 1072011.111 };
        double[] KeyPos6 = new double[4] { -932977.7778, 1331200, -1331200, 932977.7778 };
        double[] KeyPos7 = new double[4] { -762311.1111, 1228800, -1228800, 762311.1111 };
        double[] KeyPos8 = new double[4] { -568888.8889, 955733.3333, -955733.3333, 568888.8889 };
        double[] KeyPos9 = new double[4] { -443733.3333, 762311.1111, -762311.1111, 443733.3333 };
        double[] KeyPos10 = new double[4] { -113777.7778, 432355.5556, -432355.5556, 113777.7778 };
        double[] KeyPos11 = new double[4] { -56888.88889, 227555.5556, -227555.5556, 56888.88889 };
        double[] KeyPos12 = new double[4] { 0, 0, 0, 0 };
        #endregion
        ProfileSettingsObj profileParameters = new ProfileSettingsObj();    //用于设置电机参数
        double MotorVelocity = 80;
        double MotorAcceleration = 25;
        double MotorDeceleration = 25;
        // 求取从初始位置到第一个位置的电机转角变化量的绝对值

        public void start_Standup2(Motors motor)
        {
            double[] DeltaP1 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP1[i] = System.Math.Abs(KeyPos1[i] - 0);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP1 = DeltaP1.Max();

            //设置各电机运动参数，并在位置模式下运动到第一个下蹲位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP1[i] / MaxDeltaP1;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos1[i]);
            }


            //cm.MotorAmp_Array[9].WaitMoveDone(100000);
                        //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP2 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP2[i] = System.Math.Abs(KeyPos2[i] - KeyPos1[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP2 = DeltaP2.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚抬升位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP2[i] / MaxDeltaP2;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos2[i]);
            }
            //cm.MotorAmp_Array[9].WaitMoveDone(100000); 

            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP3 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP3[i] = System.Math.Abs(KeyPos3[i] - KeyPos2[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP3 = DeltaP3.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚落地位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP3[i] / MaxDeltaP3;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos3[i]);
            }
            //cm.MotorAmp_Array[9].WaitMoveDone(100000);
            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP4 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP4[i] = System.Math.Abs(KeyPos4[i] - KeyPos3[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP4 = DeltaP4.Max();

            //设置各电机运动参数,并在位置模式下左移重心
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP4[i] / MaxDeltaP4;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos4[i]);

            }
            //MotorAmp_Array[9].WaitMoveDone(100000);
            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP5 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP5[i] = System.Math.Abs(KeyPos5[i] - KeyPos4[i]);
            }
            //获取变化量的绝对值的最大值
            double MaxDeltaP5 = DeltaP5.Max();

            //设置各电机运动参数,并在位置模式下左移重心
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = 0 * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos5[i]);

            }
            Task task = Task.Run(() => {
                Thread.Sleep(1000);
               
            });
           
            task.Wait();
            //Thread.Sleep(2000);
            //motor.ampObj[3].WaitMoveDone(5000);
            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP6 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP6[i] = System.Math.Abs(KeyPos6[i] - KeyPos5[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP6 = DeltaP6.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚抬升位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP6[i] / MaxDeltaP6;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos6[i]);
            }
            //cm.MotorAmp_Array[9].WaitMoveDone(100000);
            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP7 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP7[i] = System.Math.Abs(KeyPos7[i] - KeyPos6[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP7 = DeltaP7.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚抬升位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP7[i] / MaxDeltaP7;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos7[i]);
            }
            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP8 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP8[i] = System.Math.Abs(KeyPos8[i] - KeyPos7[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP8 = DeltaP8.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚抬升位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP8[i] / MaxDeltaP8;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos8[i]);
            }
            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP9 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP9[i] = System.Math.Abs(KeyPos9[i] - KeyPos8[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP9 = DeltaP9.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚抬升位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP9[i] / MaxDeltaP9;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos9[i]);
            }
            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP10 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP10[i] = System.Math.Abs(KeyPos10[i] - KeyPos9[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP10 = DeltaP10.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚抬升位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP10[i] / MaxDeltaP10;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos10[i]);
            }
            //求取相对前一位置的电机转角变化量的绝对值
            double[] DeltaP11 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP11[i] = System.Math.Abs(KeyPos11[i] - KeyPos10[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP11 = DeltaP11.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚抬升位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP11[i] / MaxDeltaP11;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos11[i]);
            }
            double[] DeltaP12 = new double[4];
            for (int i = 0; i < motor.motor_num; i++)
            {
                DeltaP12[i] = System.Math.Abs(KeyPos12[i] - KeyPos11[i]);
            }

            //获取变化量的绝对值的最大值
            double MaxDeltaP12 = DeltaP12.Max();

            //设置各电机运动参数,并在位置模式下运动到第左脚抬升位置
            for (int i = 0; i < motor.motor_num; i++)
            {
                profileParameters = motor.ampObj[i].ProfileSettings;
                profileParameters.ProfileVel = MotorVelocity * 6400 * 4 * 160 / 360 * DeltaP12[i] / MaxDeltaP12;    //单位为°/s
                profileParameters.ProfileAccel = MotorAcceleration * 6400 * 4 * 160 / 360;    //单位为°/s2
                profileParameters.ProfileDecel = MotorDeceleration * 6400 * 4 * 160 / 360;    //单位为°/s
                profileParameters.ProfileType = CML_PROFILE_TYPE.PROFILE_TRAP;
                motor.ampObj[i].ProfileSettings = profileParameters;
                motor.ampObj[i].MoveAbs(KeyPos12[i]);

            }


        }
        }
}
