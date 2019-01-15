using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO.Ports;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using HelixToolkit.Wpf.SharpDX;

namespace SnekControl
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window, INotifyPropertyChanged
	{
		private readonly Settings settings = new Settings("snek.json");
		private readonly SnekConnection senkConn = new SnekConnection();
	    private readonly MotorControlStage stage0;

		public int Latency => senkConn.Latency;

		#region Graphs
		private Graph graph;
		public Graph Graph {
			get => graph;
			set => SetProp(ref graph, value);
		}

		public ObservableCollection<Graph> Graphs { get; } = new ObservableCollection<Graph>();

		private readonly Graph latencyGraph = new Graph {Name = "Latency"};

		private void InitGraphs()
		{
			Graphs.AddRange(stage0.tensionGraphs);
			Graphs.Add(stage0.minTensionGraph);
			Graphs.AddRange(stage0.motorGraphs);
			Graphs.Add(stage0.userInputGraph);
			Graphs.Add(latencyGraph);
			Graphs.Add(stage0.inputMagnitude);
			Graphs.AddRange(stage0.tensionInputs);
			Graph = Graphs[0];

			latencyGraph.DropOldestCondition = p => p.X < senkConn.SnekTime - 30;
		}
		#endregion
		
		#region HelixModels
		private HelixViewModel helixModel;
		public HelixViewModel HelixModel {
			get => helixModel;
			set => SetProp(ref helixModel, value);
		}

		public ObservableCollection<HelixViewModel> HelixModels { get; } = new ObservableCollection<HelixViewModel>();
		
		private readonly CurvatureViewModel curvatureViewModel = new CurvatureViewModel { Title = "Curvature"};

		private readonly LoessSurfaceViewModel tensionSurfaceViewModel = new LoessSurfaceViewModel { Title = "Tension" };

		private readonly LoessSurfaceViewModel[] tensionViewModels = {
			new LoessSurfaceViewModel {Title = "Tension Up", ZMultiplier = 3f},
			new LoessSurfaceViewModel {Title = "Tension Left", ZMultiplier = 3f},
			new LoessSurfaceViewModel {Title = "Tension Right", ZMultiplier = 3f}
		};

		private void InitHelixModels()
		{
			HelixModels.Add(curvatureViewModel);
			HelixModels.Add(tensionSurfaceViewModel);
			HelixModels.AddRange(tensionViewModels);
			HelixModel = HelixModels[0];

			// connect to data sources
			curvatureViewModel.snekStage = stage0;
			tensionSurfaceViewModel.Surface = stage0.positionTensionMap;
			for (int i = 0; i < tensionViewModels.Length; i++)
				tensionViewModels[i].Surface = stage0.cableTensionEstimation[i];

			// link cameras
			foreach (var t in tensionViewModels)
				t.Camera = tensionSurfaceViewModel.Camera;

			foreach (var s in HelixModels.OfType<LoessSurfaceViewModel>())
				s.GetSamplePos = () => stage0.CurrentPosition.ToVector2();

			if (DesignerProperties.GetIsInDesignMode(this))
				return;
			
			new Thread(HelixBackground) {
				IsBackground = true,
				Name = "Background Model Thread"
			}.Start(SynchronizationContext.Current);
		}
		#endregion

		public MainWindow()
		{
			InitializeComponent();

			stage0 = (MotorControlStage)motorControl1.DataContext;
			stage0.Index = 0;

			InitGraphs();
			InitHelixModels();

			if (DesignerProperties.GetIsInDesignMode(this))
				return;

			textBoxLog.Clear();
			Logger.Open();
			Logger.OnLogLine += OnLogLine;
			
			stage0.ConnectTo(senkConn);
			
			senkConn.OnConnStateChanged += OnConnStateChanged;
			senkConn.OnLatencyChanged += OnLatencyChanged;
			
			settings.Load();
			stage0.BindSettings(settings);

			var portScanTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
			portScanTimer.Tick += ScanPorts;
			portScanTimer.Start();
		}

		private void HelixBackground(object o)
		{
			var context = (SynchronizationContext) o;
			var sw = new Stopwatch();
			sw.Start();
			while (true) {
				HelixModel.UpdateBackground(context);
				Thread.Sleep((int)Math.Max(200 - sw.ElapsedMilliseconds, 0));
			}
		}

		public event PropertyChangedEventHandler PropertyChanged;
		protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null) => 
			PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
		
		private void SetProp<T>(ref T field, T value, [CallerMemberName] string propertyName = null)
		{
			if (!EqualityComparer<T>.Default.Equals(field, value)) {
				field = value;
				OnPropertyChanged(propertyName);
			}
		}


		private void ScanPorts(object sender, EventArgs e)
		{
			var ports = SerialPort.GetPortNames();
			foreach (var p in portComboBox.Items.Cast<string>().Except(ports).ToArray())
				portComboBox.Items.Remove(p);

			foreach (var p in ports.Where(p => !portComboBox.Items.Contains(p)))
				portComboBox.Items.Add(p);

			if (portComboBox.SelectedIndex < 0 && portComboBox.Items.Count > 0) {
				portComboBox.SelectedIndex = 0;
				if (!senkConn.Connected)
					connectButton_Click(null, null);
			}
		}

		private void OnLogLine(string s)
		{
			Dispatcher.BeginInvoke(new Action(() => {
				textBoxLog.AppendText(s + Environment.NewLine);
			}));
		}

		private void textBoxLog_TextChanged(object sender, TextChangedEventArgs e)
		{
			if (textBoxLog.LineCount > textBoxLog.MaxLines)
				textBoxLog.Text = textBoxLog.Text.Substring(textBoxLog.GetCharacterIndexFromLineIndex(textBoxLog.MaxLines / 2));

			textBoxLog.ScrollToEnd();
		}

		private void connectButton_Click(object sender, RoutedEventArgs e)
		{
			if (!senkConn.Connected) {
				senkConn.Connect((string) portComboBox.SelectedValue);
				portComboBox.IsEnabled = false;
				connectButton.Content = "Disconnect";
			}
			else {
				senkConn.Disconnect();
				portComboBox.IsEnabled = true;
				connectButton.Content = "Connect";
			}
		}

		private void OnLatencyChanged()
		{
			OnPropertyChanged("Latency");
			latencyGraph.AddPoint(new Point(senkConn.SnekTime, Latency));
		}

		private void OnConnStateChanged(bool connected)
		{
			if (connected) {
				foreach (var graph in Graphs)
					graph.Clear();
			}
		}
	}
}
