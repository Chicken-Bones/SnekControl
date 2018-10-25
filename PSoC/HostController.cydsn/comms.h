#include <stdint.h>
#include "debug.h"

#define Packet_KeepAlive 1
#define Packet_CommsReset 2
#define Packet_Latency 3
#define Packet_SetServos 4
#define Packet_DisableServos 5
#define Packet_StrainReadings 6
#define Packet_SetServos2 7

#define MAX_LEN 200
#define KeepAliveLength 5

extern _Bool connEstablished;

#define CommsError(...) { DBG_Log(__VA_ARGS__); CommsReset(); }

uint8 ReadByte();
int16 ReadInt16();
int32 ReadInt32();
void ReadPackets();
_Bool CheckReadError();

void NewPacket(uint8 id);
void WriteByte(uint8 b);
void WriteInt16(int16 i);
void WriteInt32(int32 i);
void SendPacket();

void HandlePacket();

void CommsReset();
void SetServo(int servo, int32 minutes);
void DisableServos();
void OnConnStateChange(_Bool connected);