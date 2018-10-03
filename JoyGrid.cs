using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace SnekControl
{
    public class JoyGrid : FrameworkElement
    {
	    public static readonly DependencyProperty StrainBrushProperty = DependencyProperty.Register(
		    nameof(StrainBrush), typeof(Brush), typeof(JoyGrid));

	    public Brush StrainBrush {
		    get => (Brush) GetValue(StrainBrushProperty);
		    set => SetValue(StrainBrushProperty, value);
	    }

	    /*public static readonly DependencyProperty HValueProperty = DependencyProperty.Register(
		    nameof(HValue), typeof(int), typeof(JoyGrid), new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));

	    public int HValue {
		    get => (int) GetValue(HValueProperty);
		    set => SetValue(HValueProperty, value);
	    }

	    public static readonly DependencyProperty VValueProperty = DependencyProperty.Register(
		    nameof(VValue), typeof(int), typeof(JoyGrid), new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));

	    public int VValue {
		    get => (int) GetValue(VValueProperty);
		    set => SetValue(VValueProperty, value);
	    }*/

        public static readonly DependencyProperty T0Property = DependencyProperty.Register(
            nameof(T0), typeof(float), typeof(JoyGrid), new FrameworkPropertyMetadata(0f, FrameworkPropertyMetadataOptions.AffectsRender));

        public float T0 {
            get => (float) GetValue(T0Property);
            set => SetValue(T0Property, value);
        }

        public static readonly DependencyProperty T1Property = DependencyProperty.Register(
            nameof(T1), typeof(float), typeof(JoyGrid), new FrameworkPropertyMetadata(0f, FrameworkPropertyMetadataOptions.AffectsRender));

        public float T1 {
            get => (float) GetValue(T1Property);
            set => SetValue(T1Property, value);
        }

        public static readonly DependencyProperty T2Property = DependencyProperty.Register(
            nameof(T2), typeof(float), typeof(JoyGrid), new FrameworkPropertyMetadata(0f, FrameworkPropertyMetadataOptions.AffectsRender));

        public float T2 {
            get => (float) GetValue(T2Property);
            set => SetValue(T2Property, value);
        }

        public static readonly DependencyProperty T3Property = DependencyProperty.Register(
            nameof(T3), typeof(float), typeof(JoyGrid), new FrameworkPropertyMetadata(0f, FrameworkPropertyMetadataOptions.AffectsRender));

        public float T3 {
            get => (float) GetValue(T3Property);
            set => SetValue(T3Property, value);
        }

	    public static readonly DependencyProperty TMaxProperty = DependencyProperty.Register(
		    nameof(TMax), typeof(float), typeof(JoyGrid), new PropertyMetadata(7f));

	    public float TMax {
		    get => (float) GetValue(TMaxProperty);
		    set => SetValue(TMaxProperty, value);
	    }

	    public static readonly DependencyProperty RangeProperty = DependencyProperty.Register(
		    nameof(Range), typeof(int), typeof(JoyGrid), new PropertyMetadata(90));

	    public int Range {
		    get => (int) GetValue(RangeProperty);
		    set => SetValue(RangeProperty, value);
	    }

	    public static readonly DependencyProperty JoyPointProperty = DependencyProperty.Register(
		    nameof(JoyPoint), typeof(Vector), typeof(JoyGrid), new FrameworkPropertyMetadata(new Vector(), 
			    FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));
		
	    public Point JoyPoint {
		    get => (Point) ((Vector) GetValue(JoyPointProperty) / Range);
		    set => SetValue(JoyPointProperty, (Vector)value * Range);
	    }

	    public static readonly DependencyProperty FeedbackPointProperty = DependencyProperty.Register(
		    nameof(FeedbackPoint), typeof(Vector), typeof(JoyGrid), new FrameworkPropertyMetadata(new Vector(), 
			    FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));
		
	    public Point FeedbackPoint {
		    get => (Point) ((Vector) GetValue(FeedbackPointProperty) / Range);
		    set => SetValue(FeedbackPointProperty, (Vector)value * Range);
	    }
		
	    private readonly Brush whiteBrush = Brushes.White;
	    private readonly Brush redBrush = Brushes.Red;
	    private readonly Brush blueBrush = Brushes.Blue;
	    private readonly Pen blackPen = new Pen(Brushes.Black, 1.0);
	    private readonly Pen grayPen = new Pen(Brushes.Gray, 1.0);

	    public JoyGrid()
	    {
			blackPen.Freeze();
		    grayPen.Freeze();
	    }

	    public Point Normalize(Point pt)
	    {
		    double size = Math.Min(ActualHeight, ActualWidth);
			return new Point((pt.X*2 - ActualWidth)/size, -(pt.Y*2 - ActualHeight)/size);
	    }

	    public Point GetCanvasPos(Point pt)
	    {
		    double size = Math.Min(ActualHeight, ActualWidth);
		    return new Point(pt.X * size/2 + ActualWidth/2, -pt.Y*size/2 + ActualHeight/2);
	    }

	    protected override void OnMouseDown(MouseButtonEventArgs e)
	    {
		    UpdateJoyPos(e.GetPosition(this));
		    CaptureMouse();
	    }

	    private void UpdateJoyPos(Point mousePos)
	    {
		    var pos = Normalize(mousePos);
		    if (pos.X < -1) pos.X = -1;
		    if (pos.X > 1) pos.X = 1;
		    if (pos.Y < -1) pos.Y = -1;
		    if (pos.Y > 1) pos.Y = 1;
		    JoyPoint = pos;
	    }

	    protected override void OnMouseMove(MouseEventArgs e)
	    {
			if (IsMouseCaptured)
				UpdateJoyPos(e.GetPosition(this));
	    }

	    protected override void OnMouseUp(MouseButtonEventArgs e)
	    {
		    base.OnMouseUp(e);
			ReleaseMouseCapture();
	    }

	    protected override void OnRender(DrawingContext drawingContext)
	    {
		    base.OnRender(drawingContext);
			
		    var bounds = new Rect(GetCanvasPos(new Point(-1, -1)), GetCanvasPos(new Point(1, 1)));
			drawingContext.DrawRectangle(Brushes.Transparent, null, bounds);

		    for (int i = -3; i <= 3; i++) {
			    float f = i / 3f;
			    var pen = i == 0 ? blackPen : grayPen;
			    drawingContext.DrawLine(pen, GetCanvasPos(new Point(-1, f)), GetCanvasPos(new Point(1, f)));
			    drawingContext.DrawLine(pen, GetCanvasPos(new Point(f, -1)), GetCanvasPos(new Point(f, 1)));
		    }

		    float offset = 8, width = 5;
		    DrawStrain(drawingContext, T1, Orientation.Horizontal, 
			    new Rect(new Point(bounds.Left, bounds.Top - offset), new Size(bounds.Width, width)));
		    DrawStrain(drawingContext, T0, Orientation.Horizontal, 
			    new Rect(new Point(bounds.Left, bounds.Bottom + offset - width), new Size(bounds.Width, width)));

		    DrawStrain(drawingContext, T2, Orientation.Vertical, 
			    new Rect(new Point(bounds.Left - offset, bounds.Top), new Size(width, bounds.Height)));
		    DrawStrain(drawingContext, T3, Orientation.Vertical, 
			    new Rect(new Point(bounds.Right + offset - width, bounds.Top), new Size(width, bounds.Height)));
			
		    DrawCursor(drawingContext, FeedbackPoint, blueBrush);
		    DrawCursor(drawingContext, JoyPoint, redBrush);
	    }

	    private void DrawCursor(DrawingContext drawingContext, Point pt, Brush brush)
	    {
		    var r = new Rect(GetCanvasPos(pt), new Size(7, 7));
		    r.X -= r.Size.Width/2;
		    r.Y -= r.Size.Height/2;
		    drawingContext.DrawRectangle(brush, null, r);
	    }

	    private void DrawStrain(DrawingContext drawingContext, float f, Orientation orientation, Rect r)
	    {
			f /=  TMax;
		    if (f > 1) f = 1;
		    if (f < 0) f = 0;

		    LinearGradientBrush brush = (LinearGradientBrush)StrainBrush;
		    if (orientation == Orientation.Vertical) {
			    brush = (LinearGradientBrush) StrainBrush.Clone();
			    brush.StartPoint = new Point(brush.StartPoint.Y, brush.StartPoint.X);
			    brush.EndPoint = new Point(brush.EndPoint.Y, brush.EndPoint.X);
		    }

		    drawingContext.DrawRectangle(brush, null, r);
		    Rect r1 = r, r2 = r;
		    if (orientation == Orientation.Horizontal) {
			    r1.Width *= 0.5 * (1 - f);
			    r2.Width = r1.Width;
			    r2.X = r.Right - r2.Width;
		    }
		    else {
			    r1.Height *= 0.5 * (1 - f);
			    r2.Height = r1.Height;
			    r2.Y = r.Bottom - r2.Height;
		    }
		    drawingContext.DrawRectangle(whiteBrush, null, r1);
		    drawingContext.DrawRectangle(whiteBrush, null, r2);
	    }
    }
}
