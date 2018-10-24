using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Onnx;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<float>;

namespace SnekControl.Onnx
{
	public static class OnnxForward
	{
		private delegate void Forward(IDictionary<string, Matrix> outputs);
		private delegate Forward BuildForward(GraphProto graph, NodeProto node);

		private static IDictionary<string, BuildForward> builders = new Dictionary<string, BuildForward> {
			["Gemm"] = BuildGemm,
			["LSTM"] = BuildLSTM,
		};

		private static Vector Sigmoid(Vector v) => 1 / (1 + Vector.Exp(-v));

		private static Forward BuildLSTM(GraphProto graph, NodeProto node) {
			var unsqueezeIn = graph.Node.Single(n => n.OpType == "Unsqueeze" && n.Output.Single() == node.Input[0]);
			var input = unsqueezeIn.Input.Single();
			var output = node.Output[1];

			var W = GetConstant(graph, node.Input[1], squeeze: new[]{0});
			var R = GetConstant(graph, node.Input[2], squeeze: new[]{0});
			var B = GetConstant(graph, node.Input[3], squeeze: new[]{0}).Row(0);
			var H0 = GetConstant(graph, node.Input[5], squeeze: new[]{0}).Row(0);
			var C0 = GetConstant(graph, node.Input[6], squeeze: new[]{0}).Row(0);
			
			var hidden_dim = B.Count / 8;
			B = B.SubVector(0, 4 * hidden_dim) + B.SubVector(4 * hidden_dim, 4 * hidden_dim);
			W = W.Transpose();
			R = R.Transpose();

			return outputs => {
				var X = outputs[input];
				var H = H0;
				var C = C0;
				foreach (var x in X.EnumerateRows()) {
					var gates = x * W + H * R + B;
					var i = Sigmoid(gates.SubVector(0 * hidden_dim, hidden_dim));
					var o = Sigmoid(gates.SubVector(1 * hidden_dim, hidden_dim));
					var f = Sigmoid(gates.SubVector(2 * hidden_dim, hidden_dim));
					var c = Vector.Tanh(gates.SubVector(3 * hidden_dim, hidden_dim));
					C = Vector.op_DotMultiply(f, C) + Vector.op_DotMultiply(i, c);
					H = Vector.op_DotMultiply(o, Vector.Tanh(C));
				}
				outputs[output] = Matrix.Build.DenseOfRowVectors(H);
			};
		}

		private static Matrix GetConstant(GraphProto graph, string name, int[] squeeze = null) {
			var n = graph.Node.Single(n2 => n2.OpType == "Constant" && n2.Output[0] == name);
			var value = n.Attribute.Single(a => a.Name == "value");
			var dims = value.T.Dims.ToList();

			if (squeeze != null) {
				foreach (var d in squeeze) {
					if (dims[d] != 1)
						throw new Exception($"Squeeze non-singular dimension {d}:{dims[d]}");
					dims.RemoveAt(d);
				}
			}

			while (dims.Count < 2)
				dims.Insert(0, 1);

			if (dims.Count > 2)
				throw new Exception($"Tensor with dimension > 2 not supported [{string.Join(", ", dims)}]");

			int rows = (int)dims[0], cols = (int)dims[1];
			var arr = new float[rows * cols];
			using (var ms = new MemoryStream(value.T.RawData.ToByteArray())) {
				var reader = new BinaryReader(ms);
				// there's a row-major -> column major transpose in here
				for (int r = 0; r < rows; r++)
					for (int c = 0; c < cols; c++)
						arr[c * rows + r] = reader.ReadSingle();
			}
			
			return Matrix.Build.Dense(rows, cols, arr);
		}

		private static Forward BuildGemm(GraphProto graph, NodeProto node) {
			// warning, not even close to a full implementation
			var alpha = node.Attribute.Single(a => a.Name == "alpha").F;
			var beta = node.Attribute.Single(a => a.Name == "beta").F;

			
			var B = GetConstant(graph, node.Input[1]);
			var C = GetConstant(graph, node.Input[2]);
			if (node.Attribute.Any(a => a.Name == "transB" && a.I == 1))
				B = B.Transpose();

			return outputs => {
				var A = outputs[node.Input[0]];
				outputs[node.Output[0]] = alpha * A * B + beta * C;
			};
		}

		public static Func<Matrix, Matrix> FromGraph(GraphProto graph) {
			var nodeForwards = new List<Forward>();
			foreach (var node in graph.Node)
				if (builders.TryGetValue(node.OpType, out var builder))
					nodeForwards.Add(builder(graph, node));

			return input => {
				var outputs = new Dictionary<string, Matrix>{
					[graph.Input.Single().Name] = input
				};
				foreach (var f in nodeForwards)
					f(outputs);
				return outputs[graph.Output.Single().Name];
			};
		}
	}
}
