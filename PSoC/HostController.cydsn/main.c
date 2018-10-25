#include <project.h>
#include <math.h>

#include "time.h"
#include "debug.h"
#include "comms.h"

#define PERIOD 10000

#define STRAIN_LATEST 0
#define STRAIN_AVERAGE 1
#define STRAIN_LOWPASS 2
#define STRAIN_MODE STRAIN_LOWPASS
#define LOWPASS_BITS 10
#define LOWPASS_N (1<<LOWPASS_BITS)
#define LOWPASS_A (uint32)(0.995 * LOWPASS_N)
#define LOWPASS_B ((LOWPASS_N - LOWPASS_A)*LOWPASS_F)
#define LOWPASS_F 32

uint32 adc_sums[4] = {0};
uint16 adc_count = 0;
CY_ISR (Strain_ADC_ISR_LOC)
{
#if STRAIN_MODE == STRAIN_LATEST
    adc_sums[0] = Strain_ADC_finalArray[3];
    adc_sums[1] = Strain_ADC_finalArray[2];
    adc_sums[2] = Strain_ADC_finalArray[1];
    adc_sums[3] = Strain_ADC_finalArray[0];
    adc_count = 1;
#elif STRAIN_MODE == STRAIN_AVERAGE
    adc_sums[0] += Strain_ADC_finalArray[3];
    adc_sums[1] += Strain_ADC_finalArray[2];
    adc_sums[2] += Strain_ADC_finalArray[1];
    adc_sums[3] += Strain_ADC_finalArray[0];
    adc_count++;
#elif STRAIN_MODE == STRAIN_LOWPASS
    adc_sums[0] = (LOWPASS_A*adc_sums[0] + LOWPASS_B*Strain_ADC_finalArray[3]) >> LOWPASS_BITS;
    adc_sums[1] = (LOWPASS_A*adc_sums[1] + LOWPASS_B*Strain_ADC_finalArray[2]) >> LOWPASS_BITS;
    adc_sums[2] = (LOWPASS_A*adc_sums[2] + LOWPASS_B*Strain_ADC_finalArray[1]) >> LOWPASS_BITS;
    adc_sums[3] = (LOWPASS_A*adc_sums[3] + LOWPASS_B*Strain_ADC_finalArray[0]) >> LOWPASS_BITS;
    adc_count = LOWPASS_F;
#endif
}

/*CY_ISR (Strain_ADC2_ISR_LOC)
{
    adc_sums[0] += Strain_ADC2_GetResult16();
    adc_count++;
}*/

/*CY_ISR (Strain_ADC3_ISR_LOC)
{
    adc_sums[0] += Strain_ADC3_GetResult32();
    adc_count++;
}*/

uint16 pwm_cnt[4] = {0};
CY_ISR (PWM_1_TC_LOC)
{
    PWM_1_ReadStatusRegister();
    PWM_1_WriteCompare1(pwm_cnt[0]);
    PWM_1_WriteCompare2(pwm_cnt[1]);
}

CY_ISR (PWM_2_TC_LOC)
{
    PWM_2_ReadStatusRegister();
    PWM_2_WriteCompare1(pwm_cnt[2]);
    PWM_2_WriteCompare2(pwm_cnt[3]);
}

void Reset_ADC_Avg() {
#if STRAIN_MODE == STRAIN_AVERAGE
    adc_sums[0] = 0;
    adc_sums[1] = 0;
    adc_sums[2] = 0;
    adc_sums[3] = 0;
    adc_count = 0;
#endif
}

void SetServo(int i, int32 minutes) {
    pwm_cnt[i] = 1500 + minutes*1000/(90*60);
}

void DisableServo(int i) {
    pwm_cnt[i] = 0;
}

void DisableServos() {
    for (int i = 0; i < 4; i++)
        DisableServo(i);
}

void ReadStrain() {
    NewPacket(Packet_StrainReadings);
    WriteInt32(ms());
    for (int i = 0; i < 4; i++)
        WriteInt32(Strain_ADC_CountsTo_uVolts(adc_sums[i])/adc_count);
    
    SendPacket();
    Reset_ADC_Avg();
}

void OnConnStateChange(_Bool connected) {
    if (!connected)
        DisableServos();
}

void MainLoop() {
    uint32 lastStrain = ms();
    while(1) {
        ReadPackets();
        if (ms() - lastStrain >= 10) {
            lastStrain = ms();
            ReadStrain();
        }
    }
}

int main()
{
    Time_Start();
    DBG_Start();
    
    PWM_1_Init();
    PWM_2_Init();
    PWM_1_TC_StartEx(PWM_1_TC_LOC);
    PWM_2_TC_StartEx(PWM_2_TC_LOC);
    PWM_1_Enable();
    PWM_2_Enable();
    
    Strain_ADC_Start();
    Strain_ADC_IRQ_StartEx(Strain_ADC_ISR_LOC);
    Strain_ADC_StartConvert();
    
    /*PGA_Start();
    AMux_Start();
    AMux_Select(0);*/
    /*Strain_ADC2_Start();
    Strain_ADC2_IRQ_StartEx(Strain_ADC2_ISR_LOC);
    Strain_ADC2_StartConvert();*/
    /*Strain_ADC3_Start();
    Strain_ADC3_IRQ_StartEx(Strain_ADC3_ISR_LOC);
    Strain_ADC3_StartConvert();*/
    CyGlobalIntEnable;
    
    DBG_Log("Started");    
    MainLoop();
}