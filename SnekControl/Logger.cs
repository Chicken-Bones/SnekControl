using System;
using System.IO;
using System.Linq;

namespace SnekControl
{
	static class Logger
	{
		private static StreamWriter fileWriter;
		public static event Action<string> OnLogLine;

		public static void Open()
		{
			fileWriter = new StreamWriter("log.txt", false) {AutoFlush = true};
		}

		public static void Log(string s) => LogDirect($"[{DateTime.Now:HH:mm:ss}] {s}");

		public static void LogDirect(string s)
		{
			var lines = s.Split(new [] {"\r\n", "\n"}, StringSplitOptions.None);
			lock (fileWriter) {
				foreach (var line in lines) {
					fileWriter.WriteLine(line);
					OnLogLine?.Invoke(line);
				}
			}
		}

		public static void Log(Exception e) => Log(e.ToString());
	}
}
