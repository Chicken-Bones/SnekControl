using CNTK;
using SharpDX;
using System.Collections.Generic;
using System.Linq;

namespace SnekControl
{
	public class MLTensionModel
	{
		const int seq_len = 100;
		const float dt = 1f/seq_len;
		
		private Function func;
		private Variable inputVar;
		private Variable outputVar;
		
		public MLTensionModel(string path) {
			//func = Function.Load(path, DeviceDescriptor.CPUDevice, ModelFormat.ONNX);
			//inputVar = func.Arguments.Single();
			//outputVar = func.Output;

			//Test();
		}

		internal Vector3 Eval(double snek_time, Graph[] motorGraphs) {
			var inputBuf = new float[seq_len * 3];
			for (int m = 0; m < 3; m++) {
				int i = 0;
				motorGraphs[m].SeekBackward(p => {
					while (p.X < snek_time - i*dt) {
						inputBuf[i*3 + m] = (float)p.Y;
						if (++i == seq_len)
							return true;
					}
					return false;
				});

				if (i < seq_len)
					return new Vector3(float.NaN);
			}
			
			return Eval(inputBuf);
		}

		private Vector3 Eval(float[] inputBuf) {
			var inputView = new NDArrayView(inputVar.Shape, inputBuf, DeviceDescriptor.CPUDevice);
			var inputValue = Value.Create(inputVar.Shape, new[] { inputView } , DeviceDescriptor.CPUDevice);
			var inputDict = new Dictionary<Variable, Value> {{ inputVar, inputValue }};
			var outputDict = new Dictionary<Variable, Value> {{ outputVar, null }};
			func.Evaluate(inputDict, outputDict, DeviceDescriptor.CPUDevice);
			var outputData = outputDict[outputVar].GetDenseData<float>(outputVar).Single().ToArray();
			return new Vector3(outputData);
		}

		public void Test() {
			var testData = new float[seq_len * 3];
			for (int i = 0; i < testData.Length; i++)
				testData[i] = i * 2f / (testData.Length-1);

			Logger.Log("ML Network Test "+Eval(testData));
		}
	}
}
