using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using HelixToolkit.Wpf.SharpDX;
using HelixToolkit.Wpf.SharpDX.Core2D;
using HelixToolkit.Wpf.SharpDX.Elements2D;
using HelixToolkit.Wpf.SharpDX.Model.Scene2D;
using SharpDX;
using SharpDX.Direct2D1;
using SharpDX.Mathematics.Interop;
using Point = System.Windows.Point;

namespace SnekControl
{
	/// <summary>
	/// Interaction logic for HelixGraph.xaml
	/// </summary>
	public partial class HelixGraph : UserControl
	{
		public static readonly DependencyProperty GraphProperty = DependencyProperty.Register(
			nameof(Graph), typeof(Graph), typeof(HelixGraph), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

		public Graph Graph {
			get => (Graph) GetValue(GraphProperty);
			set => SetValue(GraphProperty, value);
		}

		public Camera Camera { get; } = new PerspectiveCamera();
		public IEffectsManager EffectsManager { get; } = new DefaultEffectsManager();

		public HelixGraph()
		{
			InitializeComponent();
		}
	}

	public class HelixGraphRenderCore2D : PathRenderCore2D
	{
		private Graph graph;
		private Dispatcher dispatcher;

		public Graph Graph
		{
			get => graph;
			set
			{
				var old = graph;
				if(SetAffectsRender(ref graph, value)) {
					isGeometryChanged = true;
					if (old != null)
						old.OnPointsChanged -= OnPointsChanged;
					if (value != null)
						graph.OnPointsChanged += OnPointsChanged;
				}
			}
		}

		public HelixGraphRenderCore2D(Dispatcher dispatcher)
		{
			this.dispatcher = dispatcher;
		}

		private void OnPointsChanged()
		{
			dispatcher.Invoke(InvalidateRenderer);
			isGeometryChanged = true;
		}

		protected override void OnLayoutBoundChanged(RectangleF layoutBound)
		{
			isGeometryChanged = true;
		}

		protected override void OnRender(RenderContext2D context)
		{
			if (isGeometryChanged)
			{               
				RemoveAndDispose(ref geometry);
				if (graph == null)
					return;

				var points = graph.ToList();
				if (points.Count == 0)
					return;

				geometry = Collect(new PathGeometry1(context.DeviceResources.Factory2D));
				using (var sink = geometry.Open())
				{
					sink.SetFillMode(FillMode);
					DrawPoints(sink, points, graph.DataBounds);
					sink.Close();
				}
				isGeometryChanged = false;
			}
			base.OnRender(context);
		}

		private RawVector2 ToCanvas(Point dataPt, Rect dataBounds) => new RawVector2(
			(float)((dataPt.X - dataBounds.X) / dataBounds.Width * LayoutBound.Width) + LayoutBound.Left, 
			(float)((1 - (dataPt.Y - dataBounds.Y) / dataBounds.Height) * LayoutBound.Height) + LayoutBound.Top);

		private void DrawPoints(GeometrySink sink, IReadOnlyList<Point> points, Rect dataBounds)
		{
			sink.BeginFigure(ToCanvas(points[0], dataBounds), FigureBegin.Hollow);
			for (int i = 1; i < points.Count; i++)
				sink.AddLine(ToCanvas(points[i], dataBounds));

			sink.EndFigure(FigureEnd.Open);
		}
	}

	public class HelixGraphNode2D : ShapeNode2D
	{
		private Dispatcher dispatcher;

		public HelixGraphNode2D(Dispatcher dispatcher)
		{
			this.dispatcher = dispatcher;
		}

		public Graph Graph
		{
			set => ((HelixGraphRenderCore2D) RenderCore).Graph = value;
			get => ((HelixGraphRenderCore2D) RenderCore).Graph;
		}

		protected override ShapeRenderCore2DBase CreateShapeRenderCore() => new HelixGraphRenderCore2D(dispatcher);

		protected override bool OnHitTest(ref Vector2 mousePoint, out HitTest2DResult hitResult)
		{
			hitResult = null;
			return false;
		}
	}

	public class HelixGraphModel2D : ShapeModel2D
	{
		public static readonly DependencyProperty GraphProperty = DependencyProperty.Register(
			nameof(Graph), typeof(Graph), typeof(HelixGraphModel2D), new PropertyMetadata(null, GraphChanged));

		private static void GraphChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
		{
			var h = (HelixGraphModel2D) d;
			h.graphChanged = true;
			h.InvalidateRender();
		}

		private bool graphChanged;
		public Graph Graph {
			get => (Graph) GetValue(GraphProperty);
			set => SetValue(GraphProperty, value);
		}

		protected override SceneNode2D OnCreateSceneNode() => new HelixGraphNode2D(Dispatcher);

		protected override void OnUpdate(RenderContext2D context)
		{
			base.OnUpdate(context);
			if (graphChanged)
			{
				((HelixGraphNode2D) SceneNode).Graph = Graph;
				graphChanged = false;
			}
		}
	}
}
