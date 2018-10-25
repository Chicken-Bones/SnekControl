using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using HelixToolkit.Wpf.SharpDX;
using HelixToolkit.Wpf.SharpDX.Core;
using SharpDX;
using SharpDX.Direct3D11;
using Geometry3D = HelixToolkit.Wpf.SharpDX.Geometry3D;
using MeshGeometry3D = HelixToolkit.Wpf.SharpDX.MeshGeometry3D;
using Quaternion = SharpDX.Quaternion;

namespace SnekControl
{
	public class CurvatureViewModel : HelixViewModel
	{
		public MotorControlStage snekStage;
		
		private MeshGeometryModel3D _model;
		private MeshGeometryModel3D _deflectionModel;
		private LineGeometryModel3D _refLines;
		public CurvatureViewModel()
		{
			Camera.LookDirection = new Vector3D(-100, -65, -180);
			Camera.Position = (Point3D) (-Camera.LookDirection + new Vector3D(0, 0, 50));

			Items.Add(_refLines = new LineGeometryModel3D {
				Color = Colors.White,
				Smoothness = 0.7
			});

			Items.Add(new LineGeometryModel3D {
				Color = Colors.Gray,
				Smoothness = 0.7,
				Geometry = BuildGrid()
			});

			var mat = PhongMaterials.Brass.CloneMaterial();
			mat.DiffuseColor = new Color4((Color3)mat.DiffuseColor, 0.9f);
			Items.Add(_model = new MeshGeometryModel3D {
				Material = mat,
				CullMode = CullMode.Back
			});

			mat = mat.CloneMaterial();
			mat.DiffuseColor = new Color4(1, 0, 0, 0.5f);
			Items.Add(_deflectionModel = new MeshGeometryModel3D {
				Material = mat,
				CullMode = CullMode.Back
			});
		}

		private static Geometry3D BuildGrid()
		{
			float spacing = 20;
			var grid = new LineBuilder();
			for (int x = -2; x <= 2; x++) {
				grid.AddLine(new Vector3(x, -2, 0) * spacing, new Vector3(x, 2, 0) * spacing);
				grid.AddLine(new Vector3(x, -2, 5) * spacing, new Vector3(x, -2, 0) * spacing);
			}

			for (int y = -2; y <= 2; y++)
				grid.AddLine(new Vector3(-2, y, 0) * spacing, new Vector3(2, y, 0) * spacing);

			for (int z = 1; z <= 5; z++)
				grid.AddLine(new Vector3(-2, -2, z) * spacing, new Vector3(2, -2, z) * spacing);

			return grid.ToLineGeometry3D();
		}

		public override void UpdateBackground(SynchronizationContext context)
		{
			if (snekStage == null)
				return;
			
			var tube = CreateTube(snekStage.CurrentPosition, out var endPoint);
			var deflectionTube = CreateTube(snekStage.CurrentPosition + snekStage.TensionInput, out _, true);
			
			var lb = new LineBuilder();
			lb.AddLine(Vector3.Zero, new Vector3(0, 0, endPoint.Z));
			lb.AddLine(new Vector3(0, 0, endPoint.Z), new Vector3(0, endPoint.Y, endPoint.Z));
			lb.AddLine(new Vector3(0, endPoint.Y, endPoint.Z), new Vector3(endPoint.X, endPoint.Y, endPoint.Z));
			var lineGeom = lb.ToLineGeometry3D();
			lineGeom.Colors = new Color4Collection(new [] {
				Colors.Blue.ToColor4(),
				Colors.Blue.ToColor4(),
				Colors.Green.ToColor4(),
				Colors.Green.ToColor4(),
				Colors.Red.ToColor4(),
				Colors.Red.ToColor4(),
			});

			context.Send(_ => {
				_model.Geometry = tube;
				_deflectionModel.Geometry = deflectionTube;
				_refLines.Geometry = lineGeom;
			}, null);
		}

		private MeshGeometry3D CreateTube(Vector pos, out Vector3 endPoint, bool slim = false)
		{
			var m = pos.Length;
			if (double.IsNaN(m)) {
				endPoint = default;
				return new MeshGeometry3D();
			}
			
			float tubeLength = 100;
			float tubeRadius = 6;
			if (slim) tubeRadius *= 0.98f;

			float cableToCenter = 8;
			float wheelRadius = 20;
			float cableDist = (float) (m * Math.PI / 180 * wheelRadius);
			float curvature = cableDist / (tubeLength * cableToCenter);
			
			var r = Quaternion.RotationAxis(Vector3.BackwardRH, (float)Math.Atan2(pos.Y, pos.X));
			var points = new List<Vector3>();
			int divisions = 100;
			Vector3 p;
			points.Add(p = Vector3.Zero);
			for (int i = 1; i <= divisions; i++) {
				float a = curvature * tubeLength * i / divisions;
				var t = new Vector3((float)Math.Sin(a), 0, (float)Math.Cos(a));
				p += t * tubeLength / divisions;
				points.Add(Vector3.Transform(p, r));
			}

			var mb = new MeshBuilder();
			mb.AddTube(points, tubeRadius * 2, 36, false, true, true);

			endPoint = points.Last();
			return mb.ToMesh();
		}
	}
}
