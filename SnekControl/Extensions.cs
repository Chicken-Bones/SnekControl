using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Newtonsoft.Json.Linq;

namespace SnekControl
{
	public static class Extensions
	{
		public static TSource MinBy<TSource, U>(this IEnumerable<TSource> source, Func<TSource, U> selector) where U : IComparable<U>
		{
			if (source == null)
				throw new ArgumentNullException(nameof (source));

			var comparer = Comparer<U>.Default;
			var min = default(U);
			var value = default(TSource);
			bool any = false;

			foreach (var t in source) {
				var u = selector(t);
				if (!any || comparer.Compare(u, min) < 0) {
					min = u;
					value = t;
					any = true;
				}
			}

			if (!any)
				throw new InvalidOperationException("Sequence contains no elements");

			return value;
		}

		public static void AddRange<T>(this ObservableCollection<T> collection, IEnumerable<T> items)
		{
			foreach (var item in items)
				collection.Add(item);
		}

		//public static T ValueOrDefault<T>(this JToken jToken) => jToken?.Va
	}
}
