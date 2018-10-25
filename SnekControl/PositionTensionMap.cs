using System;
using System.Collections.Generic;
using System.Windows;

namespace SnekControl
{
	public class PositionTensionMap
	{
		private IDictionary<(int, int), int> map = new Dictionary<(int, int), int>();

		public bool Lookup(Vector position, out int tension)
		{
			return map.TryGetValue(((int)Math.Round(position.X), (int)Math.Round(position.Y)), out tension);
		}

		public void Update(Vector position, int tension)
		{
			map[((int) Math.Round(position.X), (int) Math.Round(position.Y))] = tension;
		}
	}
}
