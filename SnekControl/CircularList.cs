using System;
using System.Collections;
using System.Collections.Generic;

namespace SnekControl
{
	[Obsolete]
	public class CircularList<T> : IList<T>, IReadOnlyList<T>
	{
		private int _offset;
		private int _version;

		private readonly List<T> _underlying = new List<T>();

		private int limit = 100;

		public CircularList()
		{ }

		public CircularList(int limit)
		{
			this.limit = limit;
		}

		public int Limit {
			get => limit;
			set {
				var tmp = ToArray();
				Clear();
				limit = value;
				AddRange(tmp);
			}
		}

		public IEnumerator<T> GetEnumerator()
		{
			var version = _version;
			for (int i = 0; i < Count; i++) {
				if (_version != version)
					throw new Exception("Collection Modified");

				yield return this[i];
			}
		}

		IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

		public void Add(T item)
		{
			if (Count == Limit) {
				_underlying[_offset] = item;
				_offset = (_offset + 1) % Count;
			}
			else {
				if (_offset != 0)
					throw new Exception("Inconsistent State");

				_underlying.Add(item);
			}
			_version++;
		}

		public void AddRange(IEnumerable<T> items)
		{
			if (Count == 0 && items is ICollection<T> collection && collection.Count <= Limit) {
				_underlying.AddRange(collection);
				_version++;
				return;
			}

			foreach (var t in items)
				Add(t);
		}

		public void Clear()
		{
			_underlying.Clear();
			_offset = 0;
			_version++;
		}

		public bool Contains(T item) => _underlying.Contains(item);
		public void CopyTo(T[] array, int arrayIndex)
		{
			_underlying.CopyTo(_offset, array, arrayIndex, Count - _offset);
			_underlying.CopyTo(0, array, arrayIndex + Count - _offset, _offset);
		}

		public T[] ToArray()
		{
			var arr = new T[Count];
			CopyTo(arr, 0);
			return arr;
		}

		public bool Remove(T item) => throw new NotImplementedException();

		public int Count => _underlying.Count;
		public bool IsReadOnly => false;
		public int IndexOf(T item) => (_underlying.IndexOf(item) + Count - _offset) % Count;
		public void Insert(int index, T item) => throw new NotImplementedException();
		public void RemoveAt(int index) => throw new NotImplementedException();

		public T this[int index] {
			get {
				if (index < 0 || index >= Count)
					throw new IndexOutOfRangeException(""+index);

				return _underlying[(index + _offset) % Count];
			}
			set {
				if (index < 0 || index >= Count)
					throw new IndexOutOfRangeException(""+index);

				_underlying[(index + _offset) % Count] = value;
			}
		}
	}
}
