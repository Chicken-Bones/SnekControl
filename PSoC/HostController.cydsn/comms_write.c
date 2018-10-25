#include "comms.h"
#include "crc.h"

uint8 write_buf[MAX_LEN + 4];
uint8 write_idx;

uint8 *write_ptr;

void NewPacket(uint8 id) {
    write_buf[0] = '!';
    write_buf[1] = write_idx++;
    write_buf[3] = id;
    
    write_ptr = write_buf + 4;
}

void WriteByte(uint8 b) {
    *(write_ptr++) = b;
}

void WriteInt16(int16 i) {
    *(write_ptr++) = i;
    *(write_ptr++) = i >> 8;
}

void WriteInt32(int32 i) {
    *(write_ptr++) = i;
    *(write_ptr++) = i >> 8;
    *(write_ptr++) = i >> 16;
    *(write_ptr++) = i >> 24;
}

void SendPacket() {
    int len = (write_ptr - write_buf) - 3;
    if (len > MAX_LEN) {
        CommsError("Packet too long: %d", len);
        return;
    }
    
    write_buf[2] = len;
    crc_t crc = crc_finalize(crc_update(crc_init(), write_buf + 1, len + 2));
    WriteByte(crc);
    DBG_PutArray(write_buf, len + 4);
}