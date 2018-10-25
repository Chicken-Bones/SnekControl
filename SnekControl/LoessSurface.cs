using System;
using System.Windows;
using SharpDX;

namespace SnekControl
{
	public class LoessSurface
	{
		public struct Entry
		{
			public Vector2 pt;
			public float value;
			public int n;

			public void Add(Vector2 pt2, float value2)
			{
				if (n == 0) {
					pt = pt2;
					value = value2;
				}
				else {
					pt = (pt * n + pt2) / (n + 1);
					value = (value * n + value2) / (n + 1);
				}
				if (n < 100)
					n++;
			}
		}

		public readonly float radius;
		public readonly float gridSize;
		public readonly Rect domain;
		public readonly Vector2 origin;
		public readonly Entry[,] buckets;

		public LoessSurface(float gridSize, float radius, Rect domain)
		{
			this.gridSize = gridSize;
			this.radius = radius;
			this.domain = domain;
			origin = new Vector2((float) domain.X, (float) domain.Y);
			
			int bucketsX = (int) Math.Ceiling(domain.Width / gridSize);
			int bucketsY = (int) Math.Ceiling(domain.Height / gridSize);
			buckets = new Entry[bucketsX, bucketsY];
		}

		public Vector2 ToLocal(Vector2 v) => (v - origin) / gridSize;

		public void Add(Vector2 pt, float value)
		{
			var xy = ToLocal(pt);
			buckets[(int)xy.X, (int)xy.Y].Add(pt, value);

			settings?.SaveSurface(settingsKey, this);
		}
		
		public void Add(Vector pt, float value) => Add(new Vector2((float) pt.X, (float) pt.Y), value);
		public void Add(float x, float y, float value) => Add(new Vector2(x,  y), value);

		public float Sample(Vector2 pt)
		{
			float weighted_sum = 0;
			float weight_sum = 0;

			var v1 = ToLocal(pt - radius);
			var v2 = ToLocal(pt + radius);
			int x1 = (int)v1.X, y1 = (int)v1.Y;
			int x2 = (int) Math.Ceiling(v2.X), y2 = (int) Math.Ceiling(v2.Y);
			if (x1 < 0) x1 = 0;
			if (y1 < 0) y1 = 0;
			if (x2 > buckets.GetLength(0)) x2 = buckets.GetLength(0);
			if (y2 > buckets.GetLength(1)) y2 = buckets.GetLength(1);
			for (int i = x1; i < x2; i++)
			for (int j = y1; j < y2; j++) {
				var e = buckets[i, j];
				var d = (e.pt - pt).Length();
				if (e.n != 0 && d < radius) {
					var f = d / radius;
					var g = 1 - f * f * f;
					var weight = g * g * g;
					weight_sum += weight;
					weighted_sum += weight * e.value;
				}
			}

			return weighted_sum / weight_sum;
		}

		public float Sample(Vector pt) => Sample(new Vector2((float) pt.X, (float) pt.Y));
		public float Sample(float x, float y) => Sample(new Vector2(x, y));

		public void Clear()
		{
			for (int i = 0; i < buckets.GetLength(0); i++)
				for (int j = 0; j < buckets.GetLength(1); j++)
					buckets[i, j].n = 0;
		}

		private Settings settings;
		private string settingsKey;
		public void BindSettings(string settingsKey, Settings settings)
		{
			this.settingsKey = settingsKey;
			this.settings = settings;
			settings.LoadSurface(settingsKey, this);
		}
	}
}
