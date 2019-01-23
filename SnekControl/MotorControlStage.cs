using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
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

	    public readonly Graph[] tensionInputs = {
		    new Graph {Name = "Input Up", Limit = 1000},
		    new Graph {Name = "Input Left", Limit = 1000},
		    new Graph {Name = "Input Right", Limit = 1000},
	    };
		
		// three wire
	    public static Vector[] cableLocations = {
		    new Vector(Math.Sin(0), Math.Cos(0)),
		    new Vector(Math.Sin(-2 * Math.PI/3), Math.Cos(-2 * Math.PI/3)),
		    new Vector(Math.Sin( 2 * Math.PI/3), Math.Cos( 2 * Math.PI/3))
	    };

	    public Vector ControlPosition {
		    get => CalculatePosition(m);
		    set {
			    var mapT = positionTensionMap.Sample(value);
			    UpdatePositionalControl(value, float.IsNaN(mapT) ? ControlTension : mapT);
		    }
	    }

	    public double ControlTension {
		    get => (m[0] + m[1] + m[2]) / 3f;
		    set => UpdatePositionalControl(ControlPosition, value);
	    }

	    public Vector CurrentPosition => CalculatePosition(s);

	    private Vector CalculatePosition(double[] cables) =>
		    (cableLocations[0] * cables[0] + cableLocations[1] * cables[1] + cableLocations[2] * cables[2]) / 1.5;

	    private void UpdatePositionalControl(Vector pos, double t)
	    {
		    bool posChanged = Math.Abs((pos - ControlPosition).LengthSquared) > 1e-8;
		    bool tChanged = Math.Abs(t - ControlTension) > 1e-5;

		    m[0] = (float)Vector.Multiply(cableLocations[0], pos) + t;
		    m[1] = (float)Vector.Multiply(cableLocations[1], pos) + t;
		    m[2] = (float)Vector.Multiply(cableLocations[2], pos) + t;

		    if (posChanged) OnPropertyChanged(nameof(ControlPosition));
			if (tChanged) OnPropertyChanged(nameof(ControlTension));
	    }

	    public int H {
		    get => (int) Math.Round(ControlPosition.X);
		    set => ControlPosition = new Vector(value, ControlPosition.Y);
	    }

	    public int V {
		    get => (int) Math.Round(ControlPosition.Y);
		    set => ControlPosition = new Vector(ControlPosition.X, value);
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

			    if (_explorationEnabled) {
					WanderingEnabled = false;
				    StartExploring();
				} else {
					LearnCableEstimations = false;
				}
		    }
	    }

		private int _explorationPercent;
	    public int ExplorationPercent {
		    get => _explorationPercent;
			set => SetProp(ref _explorationPercent, value);
	    }

	    private bool _wanderingEnabled;
	    public bool WanderingEnabled {
		    get => _wanderingEnabled;
		    set {
			    if (value == _wanderingEnabled)
				    return;

			    _wanderingEnabled = value;
			    OnPropertyChanged();

			    if (_wanderingEnabled) {
					ExplorationEnabled = false;
				    StartWandering();
				}
		    }
	    }

		private bool _recordData;
	    public bool RecordData {
		    get => _recordData;
			set => SetProp(ref _recordData, value);
	    }

		private bool _compliantMotion;
	    public bool CompliantMotion {
		    get => _compliantMotion;
			set => SetProp(ref _compliantMotion, value);
	    }

		public float CompliantThreshold {get; set;} = 0.8f;
		
	    public double MinTension => t.Min();
	    public double MaxTension => t.Max();
	    public double TotalTension => t.Sum();
		
	    public bool LearnCableEstimations { get; set; }

	    public static readonly Rect posDomainRect = new Rect(-30, -30, 60, 60);
	    public readonly LoessSurface positionTensionMap = new LoessSurface(1, 8, posDomainRect);
	    public readonly LoessSurface[] cableTensionEstimation = {
		    new LoessSurface(1, 8, posDomainRect),
		    new LoessSurface(1, 8, posDomainRect),
		    new LoessSurface(1, 8, posDomainRect),
	    };

		private Vector3 ExternalTension(double[] t, Vector3 expectedTension) =>
		    new Vector3((float)(t[0] - expectedTension.X), (float)(t[1] - expectedTension.Y), (float)(t[2] - expectedTension.Z));

	    private Vector GetTensionInput(Vector3 externalTension) =>
		    -cableLocations[0] * externalTension.X +
		    -cableLocations[1] * externalTension.Y +
		    -cableLocations[2] * externalTension.Z;

	    private Vector3 EstimateTensionAt(Vector pos) => new Vector3(
		    cableTensionEstimation[0].Sample(pos),
		    cableTensionEstimation[1].Sample(pos),
		    cableTensionEstimation[2].Sample(pos));
		
	    public Vector3 ExpectedTension => UseMLTension ?
			mlTensionModel.Eval(snekConn.SnekTime, motorGraphs) :
			EstimateTensionAt(CurrentPosition);

	    public Vector TensionInput { get; private set; }

		private bool _useMLTension;
		public bool UseMLTension { 
			get => _useMLTension;
			set => SetProp(ref _useMLTension, value);
		}

		private MLTensionModel mlTensionModel;

	    private Vector targetPosition;
	    public Vector TargetPosition {
		    get => targetPosition;
		    set {
			    if (SetProp(ref targetPosition, value))
					lock (targettingLock)
						IsTargetting = true;
		    }
	    }

	    private bool isTargetting;
	    public bool IsTargetting {
		    get => isTargetting;
		    private set => SetProp(ref isTargetting, value);
	    }
		
		private const float velocityLimitMax = 40;
		public float VelocityLimitMax { get; } = velocityLimitMax;

		private float velocityLimit = velocityLimitMax;
		public float VelocityLimit {
			get => velocityLimit;
			set => SetProp(ref velocityLimit, value);
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
		
	    protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null, LinkedList<string> changed = null)
	    {
			if (changed == null)
				changed = new LinkedList<string>();
			if (changed.Contains(propertyName))
				return;
			changed.AddLast(propertyName);

		    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

		    if (propertyName == nameof(ControlPosition)) {
			    OnPropertyChanged(nameof(H), changed);
			    OnPropertyChanged(nameof(V), changed);
		    }

		    if (propertyName == nameof(ControlTension) || propertyName == nameof(ControlPosition)) {
			    OnPropertyChanged(nameof(M0), changed);
			    OnPropertyChanged(nameof(M1), changed);
			    OnPropertyChanged(nameof(M2), changed);
		    }

		    if (propertyName[0] == 'M' || propertyName[0] == 'O') {
			    if (DriveServos)
				    SendServoSignals();

			    if (propertyName[0] == 'M') {
				    OnPropertyChanged(nameof(ControlPosition), changed);
				    OnPropertyChanged(nameof(ControlTension), changed);
			    }

			    if (propertyName[0] == 'O')
				    settings?.SetOffsets(Index, mOffset);
		    }
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

		const bool USE_SERVO_MINUTES = true;
	    private int GetServoSignal(int i)
	    {
			const int scale = USE_SERVO_MINUTES ? 60 : 1;
	        int s = (int)Math.Round(m[i]*scale) + mOffset[i]*scale;
	        if (i <= 0)
	            s = -s;

		    if (s < -90*scale) s = -90*scale;
		    if (s > 90*scale) s = 90*scale;
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
			
			if (USE_SERVO_MINUTES)
				snekConn.SetServos2(0, signal0, signal1, signal2);
			else
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

		int a = 0;
	    private void TensionReading(double mv0, double mv1, double mv2, double mv3)
	    {
			// currently a wire swap
		    TensionReading(0, mv0);
		    TensionReading(1, mv2);
		    TensionReading(2, mv3);

		    minTensionGraph.AddPoint(new Point(snekConn.SnekTime, MinTension));
			
			var expectedTension = ExpectedTension;
			var externalTension = ExternalTension(t, expectedTension);
			TensionInput = GetTensionInput(externalTension);
		    userInputGraph.AddPoint((Point)TensionInput);
			inputMagnitude.AddPoint(new Point(snekConn.SnekTime, TensionInput.Length));

			for (int i = 0; i < 3; i++)
				tensionInputs[i].AddPoint(new Point(snekConn.SnekTime, t[i] - expectedTension[i]));

			RecordDataPoint(snekConn.SnekTime, s, t, CurrentPosition, expectedTension, TensionInput);
			UpdateCompliantMotion(expectedTension, externalTension);
			MoveTowardsTarget();
	    }

		private void ServoReading(int servo0, int servo1, int servo2, int servo3)
	    {
			const float scale = USE_SERVO_MINUTES ? 60 : 1;
		    s[0] = -servo1/scale - mOffset[0];
		    s[1] = servo2/scale - mOffset[1];
		    s[2] = servo3/scale - mOffset[2];
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
			
			mlTensionModel = new MLTensionModel("data_analysis/model_s.onnx");
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
				ControlPosition = new Vector();
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
			if (snekConn == null)
			    return double.NaN;

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

				    if (MinTension < 1.5 && TotalTension < 12)
					    ControlTension++;
					else if (MinTension > 2)
					    ControlTension--;
					else if (!dataPointRecorded) {
					    positionTensionMap.Add(ControlPosition, (float)ControlTension);
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
		
		private const int explorationRadius = 20;
	    private List<Vector> targetPositions = new List<Vector>();
	    private int targetPositionCount = 0;
	    private void Explore()
	    {
		    try {
			    var rand = new Random();
			    var tMean = new float[3];
			    while (ExplorationEnabled) {
				    if (IsTargetting || snekConn.SnekTime - lastServoSignalTime < 0.2 || !IsSettled(tMean, true)) {
					    Thread.Sleep(1);
					    continue;
				    }

				    if (LearnCableEstimations) {
					    cableTensionEstimation[0].Add(ControlPosition, tMean[0]);
					    cableTensionEstimation[1].Add(ControlPosition, tMean[1]);
					    cableTensionEstimation[2].Add(ControlPosition, tMean[2]);
				    }

				    if (targetPositions.Count == 0) {
					    Logger.Log("Starting Exploration Pass");
					    for (int i = -explorationRadius; i <= explorationRadius; i++) {
						    for (int j = -explorationRadius; j <= explorationRadius; j++) {
							    var v = new Vector(i, j);
								if (v.Length <= explorationRadius)
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

		private const int retargettingPeriod = 10;
		private object targettingLock = new object();
	    private void MoveTowardsTarget()
	    {
			lock (targettingLock) {
				if (!IsTargetting)
					return;

				var maxStep = retargettingPeriod / 1000f * VelocityLimit;
				var target = targetPosition;
				var pos = ControlPosition;
				var delta = target - pos;
				if (delta.Length > maxStep) {
					delta *= maxStep / delta.Length;
					target = delta + pos;
				}

				ControlPosition = target;

				if (target == targetPosition)
					IsTargetting = false;
			}
	    }

	    private bool isWandering;
	    private void StartWandering()
	    {
		    if (!isWandering) {
			    isWandering = true;
			    new Thread(Wander) {
				    Name = "Wander",
				    Priority = ThreadPriority.Highest,
				    IsBackground = true
			    }.Start();
		    }
	    }
		
	    private void Wander()
	    {
			const int updatePeriod = 20;
		    try {
			    var rand = new Random();
				int modeChangePeriod = 15*1000;
				var lastModeChange = new Stopwatch();
				lastModeChange.Start();

				int jerkiness = 50; //0-100
				int sleepiness = 0; //chance to stop per second (0-100)
				Vector wanderPosition = new Vector();
				double currentAngle = 0;

			    while (WanderingEnabled) {

					if ((CurrentPosition - wanderPosition).Length < 2) {
						do {
							wanderPosition = new Vector(rand.Next(-explorationRadius, explorationRadius+1), rand.Next(-explorationRadius, explorationRadius+1));
						} while (wanderPosition.Length > explorationRadius);
					}

					if (rand.Next(50 * (1000/updatePeriod)) < sleepiness) {//50 instead of 100 because I wanted more sleeps
						TargetPosition = CurrentPosition;
						Thread.Sleep(rand.Next(200, 200 + (100-sleepiness)*30));//more absentmindedness, lower stop time
					}

					var targetDirection = wanderPosition - CurrentPosition;
					var targetAngle = Math.Atan2(targetDirection.Y, targetDirection.X);
					var degreesPerSecond = 90 * Math.Pow(10, jerkiness/100d);//degrees per second (90-900)
					var angleChangeCap = (degreesPerSecond * updatePeriod / 1000) * Math.PI / 180;
					var angleChange = targetAngle - currentAngle;
					if (Math.Abs(angleChange) > Math.PI)
						angleChange -= 2*Math.PI*Math.Sign(angleChange);
					if (Math.Abs(angleChange) > angleChangeCap)
						angleChange = Math.Sign(angleChange) * angleChangeCap;

					currentAngle += angleChange;
					targetDirection = new Vector(Math.Cos(currentAngle), Math.Sin(currentAngle)) * targetDirection.Length;
					var target = CurrentPosition + targetDirection;
					if (target.Length > explorationRadius)
						target *= explorationRadius / target.Length;

					TargetPosition = target;
					
					if (lastModeChange.ElapsedMilliseconds > modeChangePeriod)
					{
						jerkiness = rand.Next(100);
						sleepiness = rand.Next(100);
						VelocityLimit = (float)(rand.NextDouble() * 0.75 + 0.25) * VelocityLimitMax;
						Logger.Log($"Wandering (Jerkiness: {jerkiness}, Sleepiness: {sleepiness}, Velocity: {(int)VelocityLimit})");
						lastModeChange.Restart();
					}
				    Thread.Sleep(updatePeriod);
			    }
		    }
		    finally {
			    isWandering = false;
			    WanderingEnabled = false;
				VelocityLimit = VelocityLimitMax;
		    }
	    }

		private TextWriter dataWriter;
		private void RecordDataPoint(double snekTime, double[] servos, double[] tension, Vector currentPosition, Vector3 expectedTension, Vector inputPosition)
		{
			if (!RecordData)
				return;

			if (dataWriter == null)
			{
				dataWriter = new StreamWriter($"{DateTime.Now:yyyy-MM-dd HH-mm}.csv");
				dataWriter.WriteLine("time, motor1, motor2, motor3, tension1, tension2, tension3, input1, input2, input3, posX, posY, inputX, inputY");
			}
			var inputTension = new Vector3((float)tension[0], (float)tension[1], (float)tension[2]) - expectedTension;
			dataWriter.WriteLine($"{snekTime:0.000}, " +
				$"{servos[0]:0.00}, {servos[1]:0.00}, {servos[2]:0.00}, " +
				$"{tension[0]:0.000}, {tension[1]:0.000}, {tension[2]:0.000}, " +
				$"{inputTension[0]:0.000}, {inputTension[1]:0.000}, {inputTension[2]:0.000}, " +
				$"{currentPosition.X:0.00}, {currentPosition.Y:0.00}, " +
				$"{inputPosition.X:0.00}, {inputPosition.Y:0.00}");
		}

		private void UpdateCompliantMotion(Vector3 expectedTension, Vector3 externalTension) {
			if (!CompliantMotion)
				return;
			
			const float limitA = -10;
			const float limitB = 20;
			const float scale = -20f;

			bool any = false;
			for (int i = 0; i < 3; i++) {
				var deltaT = externalTension[i];
				if (expectedTension[i] < 1.5f) //ensure at cables remain tensioned
					deltaT += expectedTension[i] - 1.5f;

				if (float.IsNaN(deltaT) || deltaT > -CompliantThreshold && deltaT < CompliantThreshold)
					continue;

				var deltaM =  (deltaT - CompliantThreshold * Math.Sign(deltaT)) * scale;
				m[i] += deltaM / 100f;
				if (m[i] < limitA) m[i] = limitA;
				if (m[i] > limitB) m[i] = limitB;

				any = true;
			}

			if (any)
				OnPropertyChanged("ControlPosition");
		}
    }
}
