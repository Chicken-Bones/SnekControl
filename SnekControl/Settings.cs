using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SharpDX;

namespace SnekControl
{
	public class Settings
	{
		public readonly string path;
		private JObject json;

		public Settings(string path)
		{
			this.path = path;
		}

		public void Load()
		{
			try {
				json = JObject.Parse(File.ReadAllText(path));
				Logger.Log("Settings Loaded: "+path);
			}
			catch (Exception e) {
				Logger.Log("Failed to load settings: " + e.Message);
				Logger.Log("Using Defaults");
				json = new JObject();
			}
		}

		public void Save()
		{
			using (var w = new StreamWriter(path))
				json.WriteTo(new JsonTextWriter(w) {
					Formatting = Formatting.Indented
				});
			
			//Logger.Log("Settings Saved");
		}

		private bool saveQueued;
		private void QueueSave()
		{
			if (saveQueued)
				return;
			
			saveQueued = true;
			Task.Run(async () => {
				await Task.Delay(500);
				saveQueued = false;
				Save();
			});
		}

		private JToken GetStage(int i)
		{
			if (!json.TryGetValue("stage" + i, out var j))
				json["stage" + i] = j = new JObject();

			return j;
		}

		public int[] GetOffsets(int stageIndex)
		{
			return GetStage(stageIndex)["offsets"]?.ToObject<int[]>() ?? new int[4];
		}

		public void SetOffsets(int stageIndex, int[] offsets)
		{
			GetStage(stageIndex)["offsets"] = JToken.FromObject(offsets);
			QueueSave();
		}

		public void SaveSurface(string key, LoessSurface surf)
		{
			var arr = surf.buckets.Cast<LoessSurface.Entry>()
				.Where(e => e.n > 0)
				.Select(e => new float[] {e.pt.X, e.pt.Y, e.value})
				.ToArray();

			json[key] = JToken.FromObject(arr);
			QueueSave();
		}

		public void LoadSurface(string key, LoessSurface surf)
		{
			if (json.TryGetValue(key, out var jList))
				foreach (var v in jList.ToObject<float[][]>())
					surf.Add(v[0], v[1], v[2]);
		}
	}
}
