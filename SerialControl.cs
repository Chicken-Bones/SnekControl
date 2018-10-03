using System;
using System.Collections.Generic;
using System.Threading;

#if SerialPortStream
using RJCP.IO.Ports;
using Serial = RJCP.IO.Ports.SerialPortStream;
#else
using System.IO.Ports;
using Serial = System.IO.Ports.SerialPort;
#endif

namespace DiffBotMasterControl
{
	public class SerialControl
	{
		/// <summary>
		/// Creates a new DiffBotSerial instance with the serial parameters for the project
		/// </summary>
		public static SerialControl Create(string portName) {
			return new SerialControl(
#if SerialPortStream
				new Serial(portName, 115200, 8, Parity.None, StopBits.One)
#else
				new Serial(portName, 115200, Parity.None, 8, StopBits.One)
#endif
				);
		}
		
		public readonly Serial port;
		private readonly CancellationTokenSource close;
		private readonly CancellationToken readCt;
		private readonly List<Thread> threads = new List<Thread>();
		private bool started;

		public bool Closed => close.IsCancellationRequested;

		public SerialControl(Serial port) {
			this.port = port;
			close = new CancellationTokenSource();
			readCt = close.Token;
			port.ReadTimeout = 5;
		}

		public CancellationToken CancellationToken() {
			return close.Token;
		}

		public byte ReadByte() {
			while (true) {
				readCt.ThrowIfCancellationRequested();
				try {
					return (byte) port.ReadByte();
				}
				catch (TimeoutException) { }
				catch (Exception) {
					if (!port.IsOpen)
						Close();
					else
						throw;
				}
			}
		}

		/*private byte[] readBuf = new byte[4096];
		private int readLen;
		private int readOffset;

		public async Task<byte> ReadByte()
		{
			if (readOffset < readLen)
				return readBuf[readOffset++];
			
			try {
				readLen = await port.ReadAsync(readBuf, readOffset = 0, readLen, readCt);
			}
			catch (Exception) {
				readCt.ThrowIfCancellationRequested();
				throw;
			}

			if (readLen <= 0)
				throw new Exception("No data");
			
			return readBuf[readOffset++];
		}*/

		public void SendPacket(byte[] bytes) {
		    lock (port) {
		        port.Write(bytes, 0, bytes.Length);
		    }
		}

		public void ReadBytes(byte[] bytes) {
			for (int i = 0; i < bytes.Length; i++)
				bytes[i] = ReadByte();
		}

		public void Open() {
			port.Open();
			foreach(var thread in threads)
				thread.Start();

			started = true;
		}

		public void AddThread(Action<SerialControl> action, ThreadPriority priority = ThreadPriority.Normal) {
			AddThread(() => action(this), priority);
		}

		public void AddThread(Action action, ThreadPriority priority = ThreadPriority.Normal) {
			var thread = new Thread(() => action()) {
				IsBackground = true,
				Priority = priority
			};
			threads.Add(thread);
			if(started)
				thread.Start();
		}

		public void Close() {
			if (close.IsCancellationRequested)
				return;

			close.Cancel();
			foreach (var thread in threads)
				if(thread.IsAlive)
					thread.Join();

			port.Close();
			close.Dispose();
		}
	}
}