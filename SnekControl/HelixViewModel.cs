using System.Collections.ObjectModel;
using System.Threading;
using System.Windows.Media.Media3D;
using HelixToolkit.Wpf.SharpDX;
using HelixToolkit.Wpf.SharpDX.Model;
using Camera = HelixToolkit.Wpf.SharpDX.Camera;
using PerspectiveCamera = HelixToolkit.Wpf.SharpDX.PerspectiveCamera;

namespace SnekControl
{
	public class HelixViewModel : ObservableObject
	{
		public string Title { get; set; }

		public Camera Camera { get; set; } = new PerspectiveCamera();
		public IEffectsManager EffectsManager { get; } = new DefaultEffectsManager();
		public ObservableCollection<Element3D> Items { get; } = new ObservableCollection<Element3D>();

		private Vector3D modelUpDirection;
		public Vector3D ModelUpDirection {
			get => modelUpDirection;
			set => Set(ref modelUpDirection, value);
		}

		public HelixViewModel()
		{
			ModelUpDirection = new Vector3D(0, 1, 0);
			Camera.Position = new Point3D(0, 5, 5);
			Camera.LookDirection = new Vector3D(0, -5, -5);
		}

		public virtual void UpdateBackground(SynchronizationContext context) {}

		public override string ToString() => Title;
	}
}
