#include "comms.h"
#include "crc.h"

int num_read = 0;
uint8 packet[MAX_LEN + 4];
uint8 read_idx;

const uint8 *read_ptr;
const uint8 *read_limit;
const char* read_error_msg;

uint8 ReadOverflow(int len) {
    if (read_error_msg) return 1;
    if (read_ptr + len > read_limit) {
        read_error_msg = "Read Overflow";
        return 1;
    }
    return 0;
}

_Bool CheckReadError() {
    if (read_error_msg && connEstablished) CommsError("%s @ %s", read_error_msg, __func__);
    return read_error_msg;
}

uint8 ReadByte() {
    if (ReadOverflow(1)) return 0;
    return *read_ptr++;
}

int16 ReadInt16() {
    if (ReadOverflow(2)) return 0;
    int16 i = read_ptr[0] | (int16)read_ptr[1] << 8;
    read_ptr += 2;
    return i;
}

int32 ReadInt32() {
    if (ReadOverflow(4)) return 0;
    int32 i = read_ptr[0] | (int32)read_ptr[1] << 8 | (int32)read_ptr[2] << 16 | (int32)read_ptr[3] << 24;
    read_ptr += 4;
    return i;
}

void ReadPackets() {
    while (DBG_GetRxBufferSize() > 0) {
        uint8 b = DBG_ReadRxData();
        
        packet[num_read++] = b;
        if (num_read == 1 && packet[0] != '!')
            num_read = 0;
        
        if (num_read == 2 && connEstablished && (++read_idx) != packet[1]) {
            CommsError("Packet Drop. Got %d, Expected %d", packet[1], read_idx);
            num_read = 0;
        }
        
        uint8 len = packet[2];
        if (num_read == 3) {
            if (!connEstablished) {
                if (len != KeepAliveLength)
                    num_read = 0;
                else
                    read_idx = packet[1];
            }
            else if (len > MAX_LEN) {
                CommsError("Packet too long (%d)", len);
                num_read = 0;
            }
        }
        
        if (num_read > 3 && num_read == len + 4) {
            crc_t crc = crc_finalize(crc_update(crc_init(), packet + 1, len + 2));
            if (crc == packet[num_read - 1]) {
                read_ptr = packet + 3;
                read_limit = read_ptr + len;
                HandlePacket();
            }
            else if (connEstablished) {
                CommsError("CRC Failed. Recieved %02X, Calculated %02X", packet[num_read - 1], crc);
            }
            num_read = 0;
        }
    }
}