#pragma once

// Stack with a fixed maximum capacity with similar interface
// to std::vector for quick porting of existing code:
template <typename T, std::size_t Capacity>
class ArrayStack {
    std::size_t nexti;
    T store[Capacity];

public:
    ArrayStack() : nexti(0) {}
    ~ArrayStack() {}

    constexpr std::size_t capacity() const { return Capacity; }
    bool empty() const { return nexti == 0; }
    std::size_t size() const { return nexti; }
    void push_back(const T& value) {
      store[nexti] = value; nexti += 1;
    }
    void pop_back() { nexti -= 1; }
    const T& back() const { return store[nexti - 1]; }
    T& back() { return store[nexti - 1]; }
    void clear() { nexti = 0; }
    const T& operator[] (std::size_t i) const { return store[i]; }
};
