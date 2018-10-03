using System;
using System.IO;
using System.Text;
using System.Threading;
using DiffBotMasterControl;

namespace SnekControl
{
	public class SnekConnection
	{
		private class Packet : BinaryWriter
		{
			private const int MAX_LEN = 200;
			private readonly MemoryStream ms = new MemoryStream();

			public Packet(PacketId id)
			{
				OutStream = ms;

				ms.WriteByte((byte)'!');
				ms.WriteByte(0); //idx
				ms.WriteByte(0); //length
				ms.WriteByte((byte)id);
			}

			public byte[] GetBytes(byte idx)
			{
				ms.WriteByte(0); //crc
				ms.Close();

				var bytes = ms.ToArray();

				int len = bytes.Length - 4;
				if (len > MAX_LEN)
					throw new Exception($"Packet too long ({ms.Length}) > {MAX_LEN}");

				bytes[1] = idx;
				bytes[2] = (byte) len;
				bytes[bytes.Length - 1] = CRC.crc(bytes, 1, bytes.Length - 2);
				return bytes;
			}
		}

		private static Random rand = new Random();

		private const int KeepAliveLength = 5;
		private const int KeepAlivePeriod = 100; // ms
		private enum PacketId
		{
			KeepAlive = 1,
			CommsReset,
			Latency,
			SetServos,
			DisableServos,
			StrainReadings,
		}

		private SerialControl serial;

		private bool connEstablished;
		private byte recvIdx;
		private byte sendIdx;
		private int keepAlive;
		private bool keepAliveResponded;

		public bool Connected => serial != null;
		public int Latency { get; private set; }
		public double SnekTime { get; private set; }

		public event Action<double, double, double, double> TensionReading;
		public event Action<int, int, int, int> ServoReading;
		public event Action<bool> OnConnStateChanged;
		public event Action OnLatencyChanged;

		public void Connect(string port)
		{
			serial = SerialControl.Create(port);
			serial.AddThread(SerialProcessor, ThreadPriority.Highest);
			serial.AddThread(KeepAlive);

			Logger.Log("Connecting on "+port);
			serial.Open();
		}

		public void Disconnect()
		{
			Logger.Log("Closing " + serial.port.PortName);
			serial.Close();
			serial = null;
		}

		private void SendPacket(Packet packet)
		{
			try {
				lock (serial.port) {
					serial.SendPacket(packet.GetBytes(sendIdx++));
				}
			}
			catch (IOException e) {
				Logger.Log(e.Message);
				ResetConn();
			}
		}

		private void KeepAlive(SerialControl serial)
		{
			connEstablished = false;
			while (!serial.Closed) {
				if (connEstablished && !keepAliveResponded) {
					Logger.Log("Timeout");
					ResetConn();
				}
				SendKeepAlive();
				Thread.Sleep(KeepAlivePeriod);
			}
		}

		public void SetServos(int servo0, int servo1, int servo2, int servo3)
		{
			if (!Connected)
				return;

			var p = new Packet(PacketId.SetServos);
			p.Write((byte)servo0);
			p.Write((byte)servo1);
			p.Write((byte)servo2);
			p.Write((byte)servo3);
			SendPacket(p);
		}

		public void DisableServos()
		{
			if (!Connected)
				return;
			
			SendPacket(new Packet(PacketId.DisableServos));
		}

		private void SendKeepAlive()
		{
			keepAliveResponded = false;
			var p = new Packet(PacketId.KeepAlive);
			p.Write(keepAlive = rand.Next());
			SendPacket(p);
		}

		private void SendLatencyResp()
		{
			SendPacket(new Packet(PacketId.Latency));
		}

		private void SerialProcessor(SerialControl serial)
		{
#if SerialPortStream
			byte[] buf = new byte[2048];
			serial.port.InvokeEventsOnThreadPool = false;
			serial.port.DataReceived += (s, args) => {
				if (args.EventType == SerialData.Chars) { }
				int len = serial.port.Read(buf, 0, buf.Length);
				for (int i = 0; i < len; i++)
					ReadByte(buf[i]);
			};
#else
			try {
				var sb = new StringBuilder();
				while (true) {
					byte b = serial.ReadByte();
					if (b == '!') {
						ReadPacket(serial);
					}
					else if (!connEstablished) { }
					else if (b == '\r') { }
					else if (b == '\n') {
						var s = sb.ToString();
						sb.Clear();
						Logger.LogDirect(s);
					}
					else {
						sb.Append((char)b);
					}
				}
			}
			catch (OperationCanceledException) {}
#endif
		}

		private readonly byte[] packet = new byte[256];
		private int num_read;
		private void ReadByte(byte b)
		{
			packet[num_read++] = b;
			if (num_read == 1 && packet[0] != '!') {
				num_read = 0;
				return;
			}
			
			if (num_read == 2 && connEstablished && (++recvIdx) != packet[1]) {
				Logger.Log($"Packet Drop. Got {packet[1]}, Expected {recvIdx}");
				ResetConn();
				num_read = 0;
				return;
			}
        
			byte len = packet[2];
			if (num_read == 3 && !connEstablished) {
				if (len != KeepAliveLength) {
					num_read = 0;
					return;
				}

				recvIdx = packet[1];
			}
        
			if (num_read > 3 && num_read == len + 4) {
				byte crcCheck = CRC.crc_update(0, packet, 1, len + 2);
				byte crcRecv = packet[num_read - 1];
				if (crcCheck != crcRecv) {
					if (connEstablished) {
						Logger.Log($"CRC Failed. Recieved {crcRecv:02X}, Calculated {crcCheck:02X}");
						ResetConn();
					}
				}
				else {
					HandlePacket(new BinaryReader(new MemoryStream(packet, 3, len)));
				}

				num_read = 0;
			}
		}

		private void ReadPacket(SerialControl serial)
		{
			byte idx = serial.ReadByte();
			if (connEstablished) {
				if (idx != ++recvIdx) {
					Logger.Log($"Packet Drop. Got {idx}, Expected {recvIdx}");
					ResetConn();
					return;
				}
			}

			int len = serial.ReadByte();
			if (!connEstablished) {
				if (len != KeepAliveLength)
					return;

				recvIdx = idx;
			}
			
			var arr = new byte[len];
			serial.ReadBytes(arr);
			byte crcRecv = serial.ReadByte();
			byte crcCheck = CRC.crc_update(0, idx);
			crcCheck = CRC.crc_update(crcCheck, (byte)len);
			crcCheck = CRC.crc_update(crcCheck, arr, 0, arr.Length);
			if (crcRecv != crcCheck) {
				if (connEstablished) {
					Logger.Log($"CRC Failed. Recieved {crcRecv:02X}, Calculated {crcCheck:02X}");
					ResetConn();
				}

				return;
			}
			
			HandlePacket(new BinaryReader(new MemoryStream(arr)));
		}

		private void ResetConn()
		{
			if (connEstablished) {
				Logger.Log("Invalidating Connection");
				connEstablished = false;
				OnConnStateChanged?.Invoke(false);
			}
		}

		private void HandlePacket(BinaryReader r)
		{
			var id = (PacketId) r.ReadByte();
			if (!connEstablished && id != PacketId.KeepAlive)
				return;

			switch (id) {
				case PacketId.KeepAlive:
					int value = r.ReadInt32() ^ -1;
					if (value != keepAlive) {
						if (connEstablished) {
							Logger.Log($"Keep Alive Mismatch. {value:X8} {keepAlive:X8}");
							ResetConn();
						}
						return;
					}

					if (!connEstablished) {
						connEstablished = true;
						Logger.Log("Connection Established.");
						OnConnStateChanged?.Invoke(true);
					}

					keepAliveResponded = true;
					SendLatencyResp();
					break;
				case PacketId.CommsReset:
					ResetConn();
					break;
				case PacketId.Latency:
					Latency = r.ReadUInt16();
					OnLatencyChanged?.Invoke();
					break;
				case PacketId.SetServos:
					SnekTime = r.ReadInt32() / 1000d;
					ServoReading?.Invoke((sbyte)r.ReadByte(), (sbyte)r.ReadByte(), (sbyte)r.ReadByte(), (sbyte)r.ReadByte());
					break;
				case PacketId.StrainReadings:
					SnekTime = r.ReadInt32() / 1000d;
					TensionReading?.Invoke(r.ReadInt32() / 1000d, r.ReadInt32() / 1000d, r.ReadInt32() / 1000d, r.ReadInt32() / 1000d);
					break;
				default:
					if (connEstablished)
						Logger.Log($"Unknown Packet Id ({id})");
					break;
			}
		}
	}
}
