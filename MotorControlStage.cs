using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using SharpDX;
using Point = System.Windows.Point;

namespace SnekControl
{
    public class MotorControlStage : INotifyPropertyChanged
    {
        private double[] m = new double[3];
        private int[] mOffset = new int[3];
        private double[] t = new double[3];
		private double[] tOffset = new double[3];
	    private double[] s = new double[3];
        
        #region BindingProperties
	    public bool Connected => snekConn.Connected;
		
	    public int M0 { get => (int)Math.Round(m[0]); set => SetProp(ref m[0], value); }
	    public int M1 { get => (int)Math.Round(m[1]); set => SetProp(ref m[1], value); }
	    public int M2 { get => (int)Math.Round(m[2]); set => SetProp(ref m[2], value); }

	    public int O0 { get => mOffset[0]; set => SetProp(ref mOffset[0], value); }
	    public int O1 { get => mOffset[1]; set => SetProp(ref mOffset[1], value); }
	    public int O2 { get => mOffset[2]; set => SetProp(ref mOffset[2], value); }

        public double T0 => t[0];
        public double T1 => t[1];
        public double T2 => t[2];

	    public readonly Graph[] tensionGraphs = {
		    new Graph {Name = "Tension Up", Limit = 1000},
		    new Graph {Name = "Tension Left", Limit = 1000},
		    new Graph {Name = "Tension Right", Limit = 1000},
	    };

	    public readonly Graph[] motorGraphs = {
		    new Graph {Name = "Motor Up"},
		    new Graph {Name = "Motor Left"},
		    new Graph {Name = "Motor Right"},
	    };
	    public readonly Graph minTensionGraph = new Graph {Name = "Min Tension", Limit = 1000};
	    public readonly Graph userInputGraph = new Graph {Name = "User Input", Limit = 400};
	    public readonly Graph inputMagnitude = new Graph {Name = "Input Magnitude", Limit = 400};
		
		// three wire
	    public static Vector[] cableLocations = {
		    new Vector(Math.Sin(0), Math.Cos(0)),
		    new Vector(Math.Sin(-2 * Math.PI/3), Math.Cos(-2 * Math.PI/3)),
		    new Vector(Math.Sin( 2 * Math.PI/3), Math.Cos( 2 * Math.PI/3))
	    };

	    public Vector Position {
		    get => CalculatePosition(m);
		    set {
			    var mapT = positionTensionMap.Sample(value);
			    UpdatePositionalControl(value, float.IsNaN(mapT) ? Tension : mapT);
		    }
	    }

	    public double Tension {
		    get => (m[0] + m[1] + m[2]) / 3f;
		    set => UpdatePositionalControl(Position, value);
	    }

	    public Vector CurrentPosition => CalculatePosition(s);

	    private Vector CalculatePosition(double[] cables) =>
		    (cableLocations[0] * cables[0] + cableLocations[1] * cables[1] + cableLocations[2] * cables[2]) / 1.5;

	    private void UpdatePositionalControl(Vector pos, double t)
	    {
		    bool posChanged = pos != Position;
		    bool tChanged = t != Tension;

		    m[0] = (float)Vector.Multiply(cableLocations[0], pos) + t;
		    m[1] = (float)Vector.Multiply(cableLocations[1], pos) + t;
		    m[2] = (float)Vector.Multiply(cableLocations[2], pos) + t;

		    if (posChanged) OnPropertyChanged("Position");
			if (tChanged) OnPropertyChanged("Tension");
	    }

	    public int H {
		    get => (int) Math.Round(Position.X);
		    set => Position = new Vector(value, Position.Y);
	    }

	    public int V {
		    get => (int) Math.Round(Position.Y);
		    set => Position = new Vector(Position.X, value);
	    }

	    private bool _driveServos;
	    public bool DriveServos {
		    get => _driveServos;
		    set {
			    if (value == _driveServos)
				    return;

			    _driveServos = value;
			    OnPropertyChanged();

			    if (_driveServos) {
				    SendServoSignals();
				    ZeroIfNeeded();
			    }
			    else {
				    DisableServos();
			    }
		    }
	    }

	    private bool _tensionControlEnabled;
	    public bool TensionControlEnabled {
		    get => _tensionControlEnabled;
		    set {
			    if (value == _tensionControlEnabled)
				    return;

			    _tensionControlEnabled = value;
			    OnPropertyChanged();

			    if (_tensionControlEnabled)
				    StartTensionControl();
		    }
	    }

	    private bool _explorationEnabled;
	    public bool ExplorationEnabled {
		    get => _explorationEnabled;
		    set {
			    if (value == _explorationEnabled)
				    return;

			    _explorationEnabled = value;
			    OnPropertyChanged();

			    if (_explorationEnabled)
				    StartExploring();
		    }
	    }

		private int explorationPercent;
	    public int ExplorationPercent {
		    get => explorationPercent;
			set => SetProp(ref explorationPercent, value);
	    }
		
	    public double MinTension => t.Min();
	    public double MaxTension => t.Max();
	    public double TotalTension => t.Sum();
		
	    public bool LearnCableEstimations { get; set; }

	    private bool learnTensionDelay;
	    public bool LearnTensionDelay {
		    get => learnTensionDelay;
		    set {
			    if (SetProp(ref learnTensionDelay, value) && value)
				    ResetDelayEstimation();
		    }
	    }

	    public static readonly Rect posDomainRect = new Rect(-30, -30, 60, 60);
	    public readonly LoessSurface positionTensionMap = new LoessSurface(1, 8, posDomainRect);
	    public readonly LoessSurface[] cableTensionEstimation = {
		    new LoessSurface(1, 8, posDomainRect),
		    new LoessSurface(1, 8, posDomainRect),
		    new LoessSurface(1, 8, posDomainRect),
	    };

	    private Vector GetTensionInput(double[] t, Vector3 expectedTension) =>
		    -cableLocations[0] * (t[0] - expectedTension.X) +
		    -cableLocations[1] * (t[1] - expectedTension.Y) +
		    -cableLocations[2] * (t[2] - expectedTension.Z);

	    private Vector3 EstimateTensionAt(Vector pos) => new Vector3(
		    cableTensionEstimation[0].Sample(pos),
		    cableTensionEstimation[1].Sample(pos),
		    cableTensionEstimation[2].Sample(pos));
			

	    private double tensionDelay;
	    public double TensionDelay {
		    get => tensionDelay;
		    set => SetProp(ref tensionDelay, value);
	    }
	    public Vector TensionDelayedPosition => HistoricalPosition(TensionDelay);
	    public Vector3 ExpectedTension => EstimateTensionAt(TensionDelayedPosition);
	    public Vector TensionInput => GetTensionInput(t, ExpectedTension);

	    public int TensionDelayMs => (int)Math.Round(TensionDelay * 1000);

	    private Vector targetPosition;
	    public Vector TargetPosition {
		    get => targetPosition;
		    set {
			    if (SetProp(ref targetPosition, value))
				    MoveToTarget();
		    }
	    }

	    private bool isTargetting;
	    public bool IsTargetting {
		    get => isTargetting;
		    private set => SetProp(ref isTargetting, value);
	    }
		
	    private ICommand zeroTensionCommand;
	    public ICommand ZeroTensionCommand => zeroTensionCommand ?? (zeroTensionCommand = new DelegateCommand(_ => ZeroTensionReadings()));
		
	    private ICommand relearnTensionCommand;
	    public ICommand RelearnTensionCommand => relearnTensionCommand ?? (relearnTensionCommand = new DelegateCommand(_ => ResetTensionMap()));

	    private ICommand relearnCablesCommand;
	    public ICommand RelearnCablesCommand => relearnCablesCommand ?? (relearnCablesCommand = new DelegateCommand(_ => ResetCableEstimations()));
	    #endregion

	    #region INotifyPropertyChanged
	    public event PropertyChangedEventHandler PropertyChanged;

	    private readonly Dictionary<string, object> propertyValues = new Dictionary<string, object>();
	    protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
	    {
		    object curValue = GetType().GetProperty(propertyName).GetValue(this);
		    if (propertyValues.TryGetValue(propertyName, out var prevValue)) {
			    if (curValue.Equals(prevValue))
				    return;
		    }
			propertyValues[propertyName] = curValue;

		    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

		    if (propertyName == "Position") {
			    OnPropertyChanged("H");
			    OnPropertyChanged("V");
		    }

		    if (propertyName == "Tension" || propertyName == "Position") {
			    OnPropertyChanged("M0");
			    OnPropertyChanged("M1");
			    OnPropertyChanged("M2");
		    }

		    if (propertyName[0] == 'M' || propertyName[0] == 'O') {
			    if (DriveServos)
				    SendServoSignals();

			    if (propertyName[0] == 'M') {
				    OnPropertyChanged("Position");
				    OnPropertyChanged("Tension");
			    }

			    if (propertyName[0] == 'O')
				    settings?.SetOffsets(Index, mOffset);
		    }

		    if (propertyName == "TensionDelay")
			    OnPropertyChanged("TensionDelayMs");
	    }

	    private bool SetProp<T>(ref T field, T value, [CallerMemberName] string propertyName = null)
	    {
		    if (!EqualityComparer<T>.Default.Equals(field, value)) {
			    field = value;
			    OnPropertyChanged(propertyName);
				return true;
		    }

		    return false;
	    }
		#endregion

	    public int Index;
	    private Settings settings;
	    private SnekConnection snekConn;
	    private double lastConnChangeTime;

	    private int GetServoSignal(int i)
	    {
	        int s = (int)Math.Round(m[i]) + mOffset[i];
	        if (i <= 0)
	            s = -s;

		    if (s < -90) s = -90;
		    if (s > 90) s = 90;
            return s;
	    }
		
	    private int[] lastSent;
	    private double lastServoSignalTime = double.MinValue;
	    private void SendServoSignals()
	    {
		    if (!Connected || !DriveServos)
			    return;

		    if (lastSent == null)
			    lastSent = new[] {int.MinValue, int.MinValue, int.MinValue};
			
		    int signal0 = GetServoSignal(0);
		    int signal1 = GetServoSignal(1);
		    int signal2 = GetServoSignal(2);
		    if (signal0 == lastSent[0] && signal1 == lastSent[1] && signal2 == lastSent[2])
			    return;
			
		    snekConn.SetServos(0, signal0, signal1, signal2);
		    lastSent[0] = signal0;
		    lastSent[1] = signal1;
		    lastSent[2] = signal2;
		    lastServoSignalTime = snekConn.SnekTime;
	    }

	    private void DisableServos() => snekConn.DisableServos();

	    private const float MV_PER_KG = 22.8f;
        private void TensionReading(int i, double mV)
        {
            t[i] = (float) mV / MV_PER_KG - tOffset[i];
	        tensionGraphs[i].AddPoint(new Point(snekConn.SnekTime, t[i]));
            OnPropertyChanged("T"+i);
        }

	    private void TensionReading(double mv0, double mv1, double mv2, double mv3)
	    {
			// currently a wire swap
		    TensionReading(0, mv0);
		    TensionReading(1, mv2);
		    TensionReading(2, mv3);

		    minTensionGraph.AddPoint(new Point(snekConn.SnekTime, MinTension));
		    userInputGraph.AddPoint((Point)TensionInput);
		    UpdateDelayEstimation();
	    }

	    private void ServoReading(int servo0, int servo1, int servo2, int servo3)
	    {
		    s[0] = -servo1 - mOffset[0];
		    s[1] = servo2 - mOffset[1];
		    s[2] = servo3 - mOffset[2];
			OnPropertyChanged("CurrentPosition");
		    for (int i = 0; i < 3; i++) {
			    var g = motorGraphs[i];
			    //duplicate last graph point to make a step
			    var last = g.Last;
				if (last.HasValue)
					motorGraphs[i].AddPoint(new Point(snekConn.SnekTime, last.Value.Y));

			    motorGraphs[i].AddPoint(new Point(snekConn.SnekTime, s[i]));
		    }
	    }

	    public void BindSettings(Settings settings)
	    {
		    this.settings = settings;
		    mOffset = settings.GetOffsets(Index);
			for (int i = 0; i < 3; i++)
				OnPropertyChanged("O"+i);
			
		    positionTensionMap.BindSettings("positionTensionMap", settings);
		    cableTensionEstimation[0].BindSettings("cable0", settings);
		    cableTensionEstimation[1].BindSettings("cable1", settings);
		    cableTensionEstimation[2].BindSettings("cable2", settings);
	    }

	    public void ConnectTo(SnekConnection snek)
	    {
			if (snekConn != null)
				throw new Exception("Already Connected");

		    snekConn = snek;
		    snek.TensionReading += TensionReading;
		    snek.ServoReading += ServoReading;
		    snek.OnConnStateChanged += OnConnStateChanged;

		    foreach (var g in motorGraphs)
			    g.DropOldestCondition = p => p.X < snekConn.SnekTime - 30;
	    }

	    private void OnConnStateChanged(bool connected)
	    {
		    lastConnChangeTime = snekConn.SnekTime;

		    if (connected && DriveServos) {
			    lastSent = null;
				SendServoSignals();
			    ZeroIfNeeded();
		    }

		    OnPropertyChanged("Connected");
	    }

	    private class HistoryUnavailableException : Exception
	    { }
	    private void TensionHistory(int cable, float duration, out float mean, out float range)
	    {
			double startTime = snekConn.SnekTime - duration;
			if (!Connected || lastConnChangeTime > startTime)
				throw new HistoryUnavailableException();
			
		    int n = 0;
		    double sum = 0, min = double.MaxValue, max = double.MinValue;
		    tensionGraphs[cable].SeekBackward(pt => {
			    if (pt.X < startTime)
				    return true;

			    sum += pt.Y;
			    if (pt.Y < min) min = pt.Y;
			    if (pt.Y > max) max = pt.Y;
			    n++;
			    return false;
		    });
			
		    if (n == 0)
			    throw new HistoryUnavailableException();

		    mean = (float) (sum / n);
		    range = (float) (max - min);
	    }
		
	    public async void ZeroTensionReadings()
	    {

		    try {
			    Logger.Log("Releasing Tension");
			    for (int i = -1; i >= -15; i--) {
				    UpdatePositionalControl(new Vector(), i);
				    await Task.Delay(100);
			    }

			    Logger.Log("Settling");
			    var tMean = new float[3];
			    while (!IsSettled(tMean)) {
				    await Task.Delay(50);
			    }

			    for (int i = 0; i < 3; i++)
				    tOffset[i] += tMean[i];

			    Logger.Log("Tension Zeroed");
			    for (int i = -14; i <= 0; i++) {
				    UpdatePositionalControl(new Vector(), i);
				    await Task.Delay(100);
			    }
		    }
		    catch (HistoryUnavailableException) {
			    Logger.Log("Zeroing Cancelled");
		    }
	    }

	    public void ZeroIfNeeded()
	    {
		    if (tOffset.All(t => t == 0) && DriveServos)
				ZeroTensionReadings();
	    }

	    public Vector HistoricalPosition(double delay) =>
		    CalculatePosition(new[] {
			    HistoricalValue(motorGraphs[0], delay),
			    HistoricalValue(motorGraphs[1], delay),
			    HistoricalValue(motorGraphs[2], delay)
		    });

	    private double HistoricalValue(Graph graph, double delay)
	    {
		    double targetTime = snekConn.SnekTime - delay;
		    Point? prev_ = null, next_ = null;
		    graph.SeekBackward(p => {
			    if (p.X >= targetTime)
				    next_ = p;
			    if (!(p.X <= targetTime))
				    return false;

			    prev_ = p;
			    return true;
		    });

		    if (prev_ == null && next_ == null)
			    return double.NaN;
			
		    var prev = (prev_ ?? next_).Value;
		    var next = (next_ ?? prev_).Value;
			
		    if (next.X == prev.X)
			    return (next.Y + prev.Y) / 2f;

		    var f = (targetTime - prev.X) / (next.X - prev.X);
		    return f * (next.Y - prev.Y) + prev.Y;
	    }

	    private void ResetTensionMap()
	    {
		    positionTensionMap.Clear();
	    }

	    private bool tensionControlling;
	    private void StartTensionControl()
	    {
		    if (!tensionControlling) {
			    tensionControlling = true;
				new Thread(TensionControl) {
					Name = "Tension Control",
					Priority = ThreadPriority.Highest,
					IsBackground = true
				}.Start();
		    }
	    }

	    private void TensionControl()
	    {
		    try {
			    bool dataPointRecorded = false;

			    while (TensionControlEnabled) {
				    if (snekConn.SnekTime - lastServoSignalTime < 0.2) {
					    dataPointRecorded = false;
						Thread.Sleep(1);
					    continue;
				    }

				    if (MinTension < 1 && TotalTension < 12)
					    Tension++;
					else if (MinTension > 2)
					    Tension--;
					else if (!dataPointRecorded) {
					    positionTensionMap.Add(Position, (float)Tension);
					    dataPointRecorded = true;
				    }
					Thread.Sleep(1);
			    }
		    }
		    finally {
			    tensionControlling = false;
			    TensionControlEnabled = false;
		    }
	    }

	    private void ResetCableEstimations()
	    {
		    cableTensionEstimation[0].Clear();
		    cableTensionEstimation[1].Clear();
		    cableTensionEstimation[2].Clear();
	    }

	    private const double MAX_DELAY = 0.5;
		private readonly double[] delayEstimationErrors = new double[50];
	    private void UpdateDelayEstimation()
	    {
		    if (!LearnTensionDelay)
			    return;

			var sampledMotorHistory = new double[delayEstimationErrors.Length][];
			for (int j = 0; j < delayEstimationErrors.Length; j++)
				sampledMotorHistory[j] = new double[3];

		    for (int i = 0; i < 3; i++) {
			    int j = 0;
			    var delay = j * MAX_DELAY / delayEstimationErrors.Length;
			    var g = motorGraphs[i];
			    bool enoughHistory = g.SeekBackward(p => {
				    while (p.X <= snekConn.SnekTime - delay) {
					    sampledMotorHistory[j][i] = p.Y;
					    if (++j == delayEstimationErrors.Length)
						    return true;

					    delay = j * MAX_DELAY / delayEstimationErrors.Length;
				    }

				    return false;
			    });

			    if (!enoughHistory)
				    return;
		    }

			ResetDelayEstimation();
		    int minIndex = 0;
		    double min = double.MaxValue;
			var estimationCache = new Dictionary<Vector, Vector3>();
		    var points = new List<Point>();
		    for (int j = 0; j < delayEstimationErrors.Length; j++) {
			    var p = CalculatePosition(sampledMotorHistory[j]);
			    if (!estimationCache.TryGetValue(p, out var expectedTension))
				    expectedTension = estimationCache[p] = EstimateTensionAt(p);

			    var error = GetTensionInput(t, expectedTension).LengthSquared;
			    delayEstimationErrors[j] += error;
			    points.Add(new Point(-j * MAX_DELAY / delayEstimationErrors.Length, error));

			    if (delayEstimationErrors[j] < min) {
				    min = delayEstimationErrors[j];
				    minIndex = j;
			    }
		    }

		    inputMagnitude.SetPoints(points);
		    TensionDelay = minIndex * MAX_DELAY / delayEstimationErrors.Length;
	    }

	    private void ResetDelayEstimation()
	    {
		    for (int j = 0; j < delayEstimationErrors.Length; j++)
			    delayEstimationErrors[j] = 0;

		    TensionDelay = 0;
	    }

	    private bool isExploring;
	    private void StartExploring()
	    {
		    if (!isExploring) {
			    isExploring = true;
			    new Thread(Explore) {
				    Name = "Learn Control",
				    Priority = ThreadPriority.Highest,
				    IsBackground = true
			    }.Start();
		    }
	    }
		
	    private List<Vector> targetPositions = new List<Vector>();
	    private int targetPositionCount = 0;
	    private void Explore()
	    {
		    try {
			    var rand = new Random();
			    int radius = 20;
			    var tMean = new float[3];
			    while (ExplorationEnabled) {
				    if (IsTargetting || snekConn.SnekTime - lastServoSignalTime < 0.2 || !IsSettled(tMean, true)) {
					    Thread.Sleep(1);
					    continue;
				    }

				    if (LearnCableEstimations) {
					    cableTensionEstimation[0].Add(Position, tMean[0]);
					    cableTensionEstimation[1].Add(Position, tMean[1]);
					    cableTensionEstimation[2].Add(Position, tMean[2]);
				    }

				    if (targetPositions.Count == 0) {
					    Logger.Log("Starting Exploration Pass");
					    for (int i = -radius; i <= radius; i++) {
						    for (int j = -radius; j <= radius; j++) {
							    var v = new Vector(i, j);
								if (v.Length <= radius)
									targetPositions.Add(v);
						    }
					    }

					    targetPositionCount = targetPositions.Count;
				    }

				    int n = rand.Next(targetPositions.Count);
				    TargetPosition = targetPositions[n];
					targetPositions.RemoveAt(n);
				    ExplorationPercent = (targetPositionCount - targetPositions.Count) * 100 / targetPositionCount;

				    Thread.Sleep(1);
			    }
		    }
		    finally {
			    isExploring = false;
			    ExplorationEnabled = false;
		    }
	    }

	    private bool IsSettled(float[] mean, bool silent = false)
	    {
		    const float duration = 1f;
		    const float jitter = 0.12f;

		    try {
			    for (int i = 0; i < 3; i++) {
				    TensionHistory(i, duration, out mean[i], out var range);
				    if (range > jitter)
					    return false;
			    }
		    }
		    catch (HistoryUnavailableException) {
			    if (!silent)
				    throw;

			    return false;
		    }

		    return true;
	    }

	    private async void MoveToTarget()
	    {
		    lock (this) {
			    if (IsTargetting)
				    return;

			    IsTargetting = true;
		    }

		    while (true) {
			    var target = targetPosition;
			    var currentTarget = target;
			    var pos = Position;
			    var delta = target - pos;
			    if (delta.Length > 2) {
				    delta /= delta.Length;
				    currentTarget = delta + pos;
			    }

			    Position = currentTarget;
			    await Task.Delay(100);

			    lock (this) {
				    if (currentTarget == targetPosition) {
					    IsTargetting = false;
					    return;
				    }
			    }
		    }
	    }
    }
}
