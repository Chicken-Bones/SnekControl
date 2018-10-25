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

#include "DBG.h"
#include "time.h"
#include <stdio.h>

extern char debug_buf[256];

#define DBG_Printf(...) { sprintf(debug_buf, __VA_ARGS__); DBG_PutString(debug_buf); }
#define DBG_Time() { uint32 _ms = ms(); DBG_Printf("[%lu.%03lu] ", _ms / 1000, _ms % 1000ul); }
#define DBG_Log(...) {DBG_Time(); DBG_Printf(__VA_ARGS__); DBG_PutString("\r\n"); }

void DBG_PutInt32(int32 i);

/* [] END OF FILE */