using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;

namespace SnekControl
{
	public class Graph
	{
		private Rect dataBounds;
		private bool recalcDataBounds;

		public Rect DataBounds {
			get {
				lock (this) {
					if (!recalcDataBounds) 
						return dataBounds;

					recalcDataBounds = false;
					if (points.Count == 0) {
						dataBounds = new Rect();
						return dataBounds;
					}
					
					var n = points.First;
					dataBounds = new Rect(n.Value, new Size(0, 0));
					while ((n = n.Next) != null)
						dataBounds.Union(n.Value);

					if (dataBounds.Height == 0) {
						dataBounds.Height = 2;
						dataBounds.Y -= 1;
					}

					if (dataBounds.Width == 0) {
						dataBounds.Width = 2;
						dataBounds.X -= 1;
					}
				}

				return dataBounds;
			}
		}

		public string Name { get; set; }
		public int Limit { get; set; } = int.MaxValue;

		public Point? Last {
			get {
				lock (this) {
					return points.Count > 0 ? (Point?)points.Last.Value : null;
				}
			}
		}

		public event Action OnPointsChanged;
		public Func<Point, bool> DropOldestCondition;

		private readonly LinkedList<Point> points = new LinkedList<Point>();
		public Graph()
		{
			OnPointsChanged += () => recalcDataBounds = true;

			for (int i = 0; i < 200; i++) {
				double x = i * Math.PI / 100f;
				AddPoint(new Point(x, Math.Sin(x)));
			}
		}

		public void AddPoint(Point point)
		{
			lock (this) {
				points.AddLast(point);
				while (points.Count > 0 && (points.Count > Limit || (DropOldestCondition?.Invoke(points.First.Value) ?? false)))
					points.RemoveFirst();
			}
			OnPointsChanged?.Invoke();
		}

		public void Clear()
		{
			lock (this) {
				points.Clear();
			}
			OnPointsChanged?.Invoke();
		}

		public void SetPoints(List<Point> list)
		{
			lock (this) {
				points.Clear();
				foreach (var p in list)
					points.AddLast(p);
			}
			OnPointsChanged?.Invoke();
		}

		public override string ToString() => Name ?? base.ToString();

		public IReadOnlyList<Point> ToList()
		{
			lock (this) {
				return points.ToArray();
			}
		}

		public bool SeekBackward(Func<Point, bool> selector) => SeekBackward(selector, out _);
		public bool SeekBackward(Func<Point, bool> selector, out Point pt)
		{
			lock (this) {
				var n = points.Last;
				while (n != null && !selector(n.Value))
					n = n.Previous;

				if (n == null) {
					pt = default;
					return false;
				}

				pt = n.Value;
				return true;
			}
		}
	}
}
