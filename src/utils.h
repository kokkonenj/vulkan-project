#pragma once
#include <functional>

namespace utils {

	template <typename T, typename... Rest>
	void hashCombine(size_t seed, const T& v, const Rest&... rest)
	{
		seed ^= std::hash<T>{}(v)+0x9e3779b9 + (seed << 6) + (seed >> 2);
		(hashCombine(seed, rest), ...);
	}

}