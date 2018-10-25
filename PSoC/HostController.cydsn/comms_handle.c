#include "comms.h"
#include "crc.h"

_Bool connEstablished = 0;

void CommsReset() {
    if (connEstablished) {
        connEstablished = 0;
        
        NewPacket(Packet_CommsReset);
        SendPacket();
        
        OnConnStateChange(0);
    }
}

uint32 latency_start;
void HandlePacket() {
    uint8 id = ReadByte();
    if (CheckReadError())
        return;
    
    if (!connEstablished && id != Packet_KeepAlive)
        return;
    
    switch (id) {
		case Packet_KeepAlive: {
            int32 value = ReadInt32();
            if (CheckReadError())
                return;
            
            latency_start = ms();
            NewPacket(Packet_KeepAlive);
            WriteInt32(value ^ -1);
            SendPacket();

			if (!connEstablished) {
				connEstablished = 1;
                OnConnStateChange(1);
            }
			break;
        }
        case Packet_CommsReset: {
            CommsReset();
            break;
        }
        case Packet_Latency: {
            NewPacket(Packet_Latency);
            WriteInt16(ms() - latency_start);
            SendPacket();
            break;
        }
        case Packet_SetServos: {
            int pos0 = (int8)ReadByte();
            int pos1 = (int8)ReadByte();
            int pos2 = (int8)ReadByte();
            int pos3 = (int8)ReadByte();
            if (CheckReadError())
                return;
            
            SetServo(0, pos1*60);
            SetServo(1, pos1*60);
            SetServo(2, pos2*60);
            SetServo(3, pos3*60);
            
            NewPacket(Packet_SetServos);
            WriteInt32(ms());
            WriteByte(pos0);
            WriteByte(pos1);
            WriteByte(pos2);
            WriteByte(pos3);
            SendPacket();
            break;
        }
        case Packet_SetServos2: {
            int pos0 = ReadInt16();
            int pos1 = ReadInt16();
            int pos2 = ReadInt16();
            int pos3 = ReadInt16();
            if (CheckReadError())
                return;
            
            SetServo(0, pos1);
            SetServo(1, pos1);
            SetServo(2, pos2);
            SetServo(3, pos3);
            
            NewPacket(Packet_SetServos2);
            WriteInt32(ms());
            WriteInt16(pos0);
            WriteInt16(pos1);
            WriteInt16(pos2);
            WriteInt16(pos3);
            SendPacket();
            break;
        }
        case Packet_DisableServos: {
            DisableServos();
            break;
        }
        default:
            CommsError("Unknown Packet Id: %d", id);
    }
}