/* ========================================
 *
 * Copyright YOUR COMPANY, THE YEAR
 * All Rights Reserved
 * UNPUBLISHED, LICENSED SOFTWARE.
 *
 * CONFIDENTIAL AND PROPRIETARY INFORMATION
 * WHICH IS THE PROPERTY OF your company.
 *
 * ========================================
*/

#include "time.h"
#include <project.h>

uint32 secs = 0;

CY_ISR(isr_Time_Counter_OV) {
    secs++;
    uint8 stat = Time_Counter_STATUS;
}

int32 up_cnt() {
    return 1000000 - Time_Counter_ReadCounter();
}

void Time_Start() {
    Time_Counter_Start();
    Time_Counter_OV_StartEx(isr_Time_Counter_OV);
}

uint32 ms() {
    return secs * 1000 + up_cnt() / 1000;
}

uint64 us() {
    return secs * 1000000L + up_cnt();
}

/* [] END OF FILE */
