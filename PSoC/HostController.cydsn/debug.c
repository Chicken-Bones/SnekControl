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

#include "debug.h"

char debug_buf[256];

void DBG_PutInt32(int32 i) {
    DBG_PutChar(i >> 24);
    DBG_PutChar(i >> 16);
    DBG_PutChar(i >> 8);
    DBG_PutChar(i);
}

/* [] END OF FILE */