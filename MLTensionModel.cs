using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Onnx;
using SnekControl.Onnx;
using System;
using System.Diagnostics;
using System.IO;

namespace SnekControl
{
	public class MLTensionModel
	{
		const int seq_len = 100;
		const float dt = 1f/seq_len;

		public readonly Func<Matrix<float>, Matrix<float>> forward;

		static MLTensionModel() {
			Control.UseNativeMKL();
		}
		
		public MLTensionModel(string path) {
			using (var stream = File.OpenRead(path)) {
				var model = ModelProto.Parser.ParseFrom(stream);
				forward = OnnxForward.FromGraph(model.Graph);
			}
			Test();
		}

		internal SharpDX.Vector3 Eval(double snek_time, Graph[] motorGraphs) {
			var input = new float[seq_len, 3];
			for (int m = 0; m < 3; m++) {
				int i = 0;
				motorGraphs[m].SeekBackward(p => {
					while (p.X < snek_time - i*dt) {
						input[i, m] = (float)p.Y;
						if (++i == seq_len)
							return true;
					}
					return false;
				});

				if (i < seq_len)
					return new SharpDX.Vector3(float.NaN);
			}
			
			var v = forward(Matrix<float>.Build.DenseOfArray(input)).Row(0);
			return new SharpDX.Vector3(v[0], v[1], v[2]);
		}

		public void Test() {
			var testData = new float[seq_len * 3];
			for (int i = 0; i < testData.Length; i++)
				testData[i] = i * 1f / (testData.Length-1);
			
			var sw = new Stopwatch();

			Vector<float> v = null;
			for (int i = 0; i < 100; i++) {
				if (i == 50)
					sw.Start();

				var m = Matrix<float>.Build.DenseOfRowMajor(seq_len, 3, testData);
				v = forward(m).Row(0);
			}
			var ms = sw.Elapsed.TotalMilliseconds / 50;
			Logger.Log($"ML Network Test [{string.Join(", ", v.ToArray())}] in {ms:0.00}ms");
		}
	}
}
