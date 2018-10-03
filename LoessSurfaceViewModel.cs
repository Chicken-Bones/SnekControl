using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using HelixToolkit.Wpf.SharpDX;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using HelixToolkit.Wpf.SharpDX.Core;
using SharpDX;
using SharpDX.Direct3D11;
using Color = System.Windows.Media.Color;
using Matrix = SharpDX.Matrix;
using MeshGeometry3D = HelixToolkit.Wpf.SharpDX.MeshGeometry3D;

namespace SnekControl
{
	public class LoessSurfaceViewModel : HelixViewModel
	{
		private LoessSurface surface;
		public LoessSurface Surface {
			get => surface;
			set {
				if (Set(ref surface, value))
					OnSurfaceChanged();
			}
		}

		public Color MinColor { get; set; } = Colors.Green;
		public Color MaxColor { get; set; } = Colors.Red;
		public Color PointColor { get; set; } = Colors.Blue;
		public float ZMultiplier { get; set; } = 1f;

		public Func<Vector2> GetSamplePos;

		private MeshGeometry3D _mesh;
		private InstancingMeshGeometryModel3D _pointsInstanceModel;
		private LineGeometryModel3D _sampleLine;
		private BillboardSingleText3D _sampleText;

		public LoessSurfaceViewModel()
		{
			Camera.Position = new Point3D(0, -50, 60);
			Camera.LookDirection = new Vector3D(0, 50, -50);
			ModelUpDirection = Camera.UpDirection = new Vector3D(0, 0, 1);
			Items.Add(new MeshGeometryModel3D {
				Geometry = _mesh = new MeshGeometry3D {
					IsDynamic = true
				},
				RenderWireframe = true,
				WireframeColor = Colors.Black,
				Material = new VertColorMaterial()
			});
			var mb = new MeshBuilder(true, false);
			mb.AddSphere(Vector3.Zero, 0.4f);
			Items.Add(_pointsInstanceModel = new InstancingMeshGeometryModel3D {
					CullMode = CullMode.Back,
					Material = PhongMaterials.White,
					Geometry = mb.ToMesh()
				});

			Items.Add(_sampleLine = new LineGeometryModel3D {
				Smoothness = 1,
				Thickness = 4,
				Color = Colors.White
			});

			Items.Add(new BillboardTextModel3D {
				Geometry = _sampleText = new BillboardSingleText3D {
					TextInfo = new TextInfo("", new Vector3(0, 1, 0)),
					FontColor = Colors.Gold.ToColor4(),
					FontSize= 24,
					//BackgroundColor = Colors.Transparent.ToColor4(),
					//Padding = new System.Windows.Thickness(2)
				},
				FixedSize = true
			});
		}

		private void OnSurfaceChanged()
		{
			float spacing = 10;
			int x1 = (int) (surface.domain.X / spacing);
			int x2 = (int) (surface.domain.Right / spacing);
			int y1 = (int) (surface.domain.Y / spacing);
			int y2 = (int) (surface.domain.Bottom / spacing);
			var grid = new LineBuilder();
			for (int x = x1; x <= x2; x++)
				grid.AddLine(new Vector3(x, y1, 0) * spacing, new Vector3(x, y2, 0) * spacing);

			for (int y = y1; y <= y2; y++)
				grid.AddLine(new Vector3(x1, y, 0) * spacing, new Vector3(x2, y, 0) * spacing);

			Items.Add(new LineGeometryModel3D {
				Color = Colors.Gray,
				Smoothness = 0.7,
				Geometry = grid.ToLineGeometry3D()
			});

			var indices = new IntCollection();
			int nx = Surface.buckets.GetLength(0);
			int ny = Surface.buckets.GetLength(1);
			for (int j = 0; j < ny; j++) {
				for (int i = 0; i < nx; i++) {
					indices.Add(j * (nx+1) + i);
					indices.Add((j+1) * (nx+1) + i);
					indices.Add((j+1) * (nx+1) + i + 1);
					indices.Add(j * (nx+1) + i);
					indices.Add((j+1) * (nx+1) + i + 1);
					indices.Add(j * (nx+1) + i + 1);
				}
			}
			_mesh.Indices = indices;
		}

		public override void UpdateBackground(SynchronizationContext context)
		{
			if (Surface == null)
				return;
			
			var positions = new Vector3Collection();
			var colours = new Color4Collection();
			float min = float.MaxValue, max = float.MinValue;
			for (var y = (float)Surface.domain.Y; y <= Surface.domain.Bottom; y += Surface.gridSize)
			for (var x = (float)Surface.domain.X; x <= Surface.domain.Right; x += Surface.gridSize) {
				var s = Surface.Sample(x, y);
				if (s < min) min = s;
				if (s > max) max = s;
				positions.Add(new Vector3(x, y, s * ZMultiplier));
			}

			colours.AddRange(positions.Select(p => LinearInterp(MinColor, MaxColor, (p.Z/ZMultiplier - min) / (max - min))));

			var instParams = new List<InstanceParameter>();
			var instTransforms = new List<Matrix>();
			foreach (var e in Surface.buckets) {
				if (e.n <= 0) 
					continue;

				instParams.Add(new InstanceParameter { DiffuseColor = PointColor.ToColor4() });
				instTransforms.Add(Matrix.Translation(e.pt.X, e.pt.Y, e.value * ZMultiplier));
			}

			if (GetSamplePos != null) {
				var pos = GetSamplePos();
				var s = Surface.Sample(pos);
				var c = LinearInterp(MinColor, MaxColor, (s - min) / (max - min));
				var z = s * ZMultiplier + 5;
				var lb = new LineBuilder();
				lb.AddLine(new Vector3(pos, 0), new Vector3(pos, z));
				var g = lb.ToLineGeometry3D();
				g.Colors = new Color4Collection { c, c };
				context.Post(_ => {
					_sampleLine.Geometry = g;
					_sampleText.TextInfo = new TextInfo($"{pos.X:0.#}, {pos.Y:0.#}, {s:0.##}", new Vector3(pos, z));
				}, null);
			}

			context.Send(_ => {
				_mesh.Positions = positions;
				_mesh.Colors = colours;
				_pointsInstanceModel.InstanceParamArray = instParams;
				_pointsInstanceModel.Instances = instTransforms;
			}, null);
		}

		private static Color4 LinearInterp(Color start, Color end, float f)
		{
			byte a = (byte) ((end.A - start.A)*f + start.A);
			byte r = (byte) ((end.R - start.R)*f + start.R);
			byte g = (byte) ((end.G - start.G)*f + start.G);
			byte b = (byte) ((end.B - start.B)*f + start.B);
			return new Color4(r / 255f, g / 255f, b / 255f, a / 255f);
		}
	}
}
