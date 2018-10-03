using System;
using System.Collections.Generic;
using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;

namespace SnekControl
{
	public class WPFGraph : FrameworkElement
	{
		public static readonly DependencyProperty MaxPointsProperty = DependencyProperty.Register(
			nameof(MaxPoints), typeof(int), typeof(WPFGraph), new PropertyMetadata(1000));

		public int MaxPoints {
			get => (int)GetValue(MaxPointsProperty);
			set => SetValue(MaxPointsProperty, value);
		}

		public static readonly DependencyProperty TickSpacingProperty = DependencyProperty.Register(
			nameof(TickSpacing), typeof(float), typeof(WPFGraph),
			new PropertyMetadata(100f));

		public float TickSpacing {
			get => (float) GetValue(TickSpacingProperty);
			set => SetValue(TickSpacingProperty, value);
		}

		public static readonly DependencyProperty GraphProperty = DependencyProperty.Register(
			nameof(Graph), typeof(Graph), typeof(WPFGraph), new PropertyMetadata(null, GraphPropertyChanged));

		public Graph Graph {
			get => (Graph) GetValue(GraphProperty);
			set => SetValue(GraphProperty, value);
		}

		private readonly Pen mainPen = new Pen(Brushes.Black, 1.0);

		private readonly Typeface typeface =
			new Typeface(new FontFamily("Segoe UI"),
				FontStyles.Normal,
				FontWeights.Normal,
				FontStretches.Normal);

		private bool renderQueued;
		private DateTime lastRender = DateTime.MinValue;
		private readonly TimeSpan renderDelay = TimeSpan.FromSeconds(1/60f);

		public WPFGraph()
		{
			mainPen.Freeze();
		}

		private static void GraphPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
		{
			var wpfGraph = (WPFGraph) d;
			if (e.OldValue is Graph oldGraph)
				oldGraph.OnPointsChanged -= wpfGraph.QueueRender;
			if (e.NewValue is Graph newGraph)
				newGraph.OnPointsChanged += wpfGraph.QueueRender;

			wpfGraph.QueueRender();
		}

		private readonly DrawingGroup backingStore = new DrawingGroup();
		protected override void OnRender(DrawingContext drawingContext)
		{
			base.OnRender(drawingContext);
			Render();
			drawingContext.DrawDrawing(backingStore);
		}

		private void Render()
		{
			lastRender = DateTime.Now;
			using (var drawingContext = backingStore.Open()) {
				if (Graph == null)
					return;

				var points = Graph.ToList();
				if (points.Count <= 0) 
					return;

				var dataBounds = Graph.DataBounds;
				drawingContext.PushTransform(new TranslateTransform(0.5, 0.5));
				DrawPoints(drawingContext, points, dataBounds);
				DrawHorizontalAxis(drawingContext, dataBounds);
				DrawVerticalAxis(drawingContext, dataBounds);
			}
		}

		private Point ToCanvas(Point dataPt, Rect dataBounds) => new Point(
				(dataPt.X - dataBounds.X) / dataBounds.Width * ActualWidth, 
				(1 - (dataPt.Y - dataBounds.Y) / dataBounds.Height) * ActualHeight);

		private FormattedText GetAxisLabelText(double value, double interval)
		{
			if (Math.Abs(value) < interval * 0.1)
				value = 0;

			return new FormattedText(
				value.ToString(CultureInfo.CurrentCulture),
				CultureInfo.CurrentCulture,
				FlowDirection.LeftToRight,
				typeface,
				12,
				Brushes.Black,
				null,
				TextFormattingMode.Display, 1f);
		}

		private void DrawPoints(DrawingContext drawingContext, IReadOnlyList<Point> points, Rect dataBounds)
		{
			var geom = new StreamGeometry();
			using (var gc = geom.Open())
			{
				gc.BeginFigure(ToCanvas(points[0], dataBounds), false, false);
				for (int i = 1; i < points.Count; i++)
					gc.LineTo(ToCanvas(points[i], dataBounds), true, false);
			}
			geom.Freeze();
			drawingContext.DrawGeometry(null, mainPen, geom);
		}

		private static readonly double[] ticks = {Math.Log10(1), Math.Log10(2), Math.Log10(5), Math.Log10(10)};
		private double GetTickIncrement(double dataSize, double drawSize)
		{
			var l = Math.Log10(dataSize * TickSpacing / drawSize);
			var r = l % 1; if (r < 0) r += 1;
			var tick = ticks.MinBy(t => Math.Abs(t - r));
			l = l - r + tick;
			return Math.Pow(10, l);
		}

		private void DrawHorizontalAxis(DrawingContext drawingContext, Rect dataBounds)
		{
			drawingContext.DrawLine(mainPen, 
				new Point(0, ActualHeight), 
				new Point(ActualWidth, ActualHeight));
			
			DrawTickLabels(dataBounds.Left, dataBounds.Right, ActualWidth, (d, ft) => {
				var pt = ToCanvas(new Point(d, dataBounds.Y), dataBounds);
				drawingContext.DrawLine(mainPen, pt, new Point(pt.X, pt.Y + 5));
				drawingContext.DrawText(ft, new Point(pt.X - ft.Width/2, pt.Y + 5));
			});
		}

		private void DrawVerticalAxis(DrawingContext drawingContext, Rect dataBounds)
		{
			drawingContext.DrawLine(mainPen, 
				new Point(0, 0), 
				new Point(0, ActualHeight));

			DrawTickLabels(dataBounds.Top, dataBounds.Bottom, ActualHeight, (d, ft) => {
				var pt = ToCanvas(new Point(dataBounds.X, d), dataBounds);
				drawingContext.DrawLine(mainPen, pt, new Point(pt.X - 5, pt.Y));
				drawingContext.DrawText(ft, new Point(pt.X - ft.Width - 5, pt.Y - ft.Height / 2));
			});
		}

		private void DrawTickLabels(double min, double max, double actualSize, Action<double, FormattedText> drawAction)
		{
			double range = max - min;
			min -= range * 0.01;
			max += range * 0.01;
			var tickIncrement = GetTickIncrement(range, actualSize);
			double d = min - min % tickIncrement;
			for (; d <= max; d += tickIncrement)
				if (d >= min)
					drawAction(d, GetAxisLabelText(d, tickIncrement));
		}

		private void QueueRender()
		{
			if (renderQueued)
				return;

			renderQueued = true;
			var delay = lastRender + renderDelay - DateTime.Now;
			if (delay < TimeSpan.Zero)
				delay = TimeSpan.Zero;

			var dt = new DispatcherTimer(DispatcherPriority.Render, Dispatcher);
			dt.Tick += (s, e) =>
			{
				dt.Stop();
				renderQueued = false;
				Render();
			};
			dt.Interval = delay;
			dt.Start();
		}
	}
}
